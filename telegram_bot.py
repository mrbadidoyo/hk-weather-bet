"""
Minimal Telegram command receiver for HK weather predictions.

Supported command:
  /predict YYYY-MM-DD

This module only receives Telegram updates and delegates to the existing
prediction pipeline in telegram_alert.py.
"""

import logging
import json
import re
import time
from datetime import datetime

import requests

from telegram_alert import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, PROCESSED_DATA_DIR, run, send_telegram

API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
POLL_TIMEOUT = 30
ERROR_SLEEP_SECONDS = 3
SNAPSHOT_PATH = PROCESSED_DATA_DIR / "prediction_snapshots.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_updates(offset=None):
    """Fetch incoming Telegram updates using long polling."""
    params = {
        "timeout": POLL_TIMEOUT,
        "allowed_updates": ["message"],
    }
    if offset is not None:
        params["offset"] = offset

    resp = requests.get(f"{API_BASE}/getUpdates", params=params, timeout=POLL_TIMEOUT + 5)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("ok", False):
        raise RuntimeError(f"Telegram getUpdates failed: {payload}")
    return payload.get("result", [])


def _load_snapshot_history(limit=2):
    """Load the most recent prediction snapshots from JSONL history."""
    if not SNAPSHOT_PATH.exists():
        return []

    snapshots = []
    try:
        with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    snapshots.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    if limit:
        return snapshots[-limit:]
    return snapshots


def load_latest_snapshot():
    """Return the latest stored prediction snapshot, if any."""
    snapshots = _load_snapshot_history(limit=1)
    return snapshots[-1] if snapshots else None


def load_previous_snapshot():
    """Return the snapshot immediately before the latest one, if any."""
    snapshots = _load_snapshot_history(limit=2)
    return snapshots[0] if len(snapshots) >= 2 else None


def reply_usage():
    """Send the fixed usage text."""
    send_telegram("Usage:\n/predict YYYY-MM-DD\n/recommend\n/why <temperature>\n/why high <temperature>\n/why low <temperature>\n/changes")


def _format_temp(value):
    try:
        return f"{int(round(float(value)))}°C"
    except (TypeError, ValueError):
        return "?°C"


def _format_percent(value):
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "?"


def _format_signed(value):
    try:
        return f"{float(value):+.1%}"
    except (TypeError, ValueError):
        return "?"


def _format_currency(value):
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "?"


def _contract_label(contract):
    return contract or "?"


def _snapshot_target(snapshot):
    return snapshot.get("target_date") or snapshot.get("metadata", {}).get("target_date") or "?"


def _snapshot_status(snapshot):
    return snapshot.get("status") or snapshot.get("metadata", {}).get("status") or "?"


def _get_contract_entry(snapshot, side, temperature):
    contracts = snapshot.get("contracts", {}).get(side, {})
    label = f"{int(temperature)}°C"
    return label, contracts.get(label)


def handle_recommend(snapshot):
    """Format the latest snapshot into a recommendation reply."""
    recs = snapshot.get("recommendations", {})
    lines = [
        f"📌 Recommendation Snapshot {_snapshot_target(snapshot)}",
        f"Status: {_snapshot_status(snapshot)}",
        "",
        f"HIGH Main Bet: {_contract_label(recs.get('high', {}).get('main_contract'))}",
        f"  Edge: {_format_signed(recs.get('high', {}).get('edge'))}",
        f"  EV: {_format_signed(recs.get('high', {}).get('ev'))}",
        f"  Kelly: {_format_currency(recs.get('high', {}).get('kelly'))}",
        "",
        f"LOW Main Bet: {_contract_label(recs.get('low', {}).get('main_contract'))}",
        f"  Edge: {_format_signed(recs.get('low', {}).get('edge'))}",
        f"  EV: {_format_signed(recs.get('low', {}).get('ev'))}",
        f"  Kelly: {_format_currency(recs.get('low', {}).get('kelly'))}",
        "",
    ]

    lottery = []
    high_lottery = _contract_label(recs.get('high', {}).get('lottery_contract'))
    low_lottery = _contract_label(recs.get('low', {}).get('lottery_contract'))
    if high_lottery and high_lottery != "?":
        lottery.append(f"HIGH {high_lottery}")
    if low_lottery and low_lottery != "?":
        lottery.append(f"LOW {low_lottery}")
    lines.append("Lottery Bet(s): " + (", ".join(lottery) if lottery else "None"))
    return "\n".join(lines)


def handle_why(snapshot, temperature, side=None):
    """Explain one contract from the latest snapshot using stored values only."""
    targets = []
    if side in ("high", "low"):
        targets.append(side)
    else:
        targets.extend(["high", "low"])

    lines = [f"🔎 Why {_format_temp(temperature)}", f"Snapshot: {_snapshot_target(snapshot)}", ""]
    found = False
    for current_side in targets:
        label, contract = _get_contract_entry(snapshot, current_side, temperature)
        if not contract:
            continue
        found = True
        lines.extend([
            f"{current_side.upper()} {label}",
            f"Classification: {contract.get('classification', 'Not Recommended')}",
            f"Forecast: {_format_temp(snapshot.get('forecast', {}).get('high' if current_side == 'high' else 'low'))}",
            f"Bias correction: {_format_signed(snapshot.get('forecast', {}).get('bias_high' if current_side == 'high' else 'bias_low'))}",
            f"Model probability: {_format_percent(contract.get('model_probability'))}",
            f"Market probability: {_format_percent(contract.get('market_probability'))}",
            f"Edge: {_format_signed(contract.get('edge'))}",
            f"EV: {_format_signed(contract.get('ev'))}",
            f"Kelly: {_format_currency(contract.get('kelly'))}",
            "",
        ])

    if not found:
        return f"No contract found for {_format_temp(temperature)} in the latest snapshot."

    return "\n".join(lines).rstrip()


def handle_changes(latest, previous):
    """Compare the latest two snapshots without recalculating anything."""
    if not previous:
        return "Historical comparison is not yet available. Run at least two predictions first."

    def _main(snapshot, side):
        return snapshot.get("recommendations", {}).get(side, {})

    def _contract(snapshot, side, kind):
        return _main(snapshot, side).get(f"{kind}_contract") or "?"

    def _metric(snapshot, side, key):
        return _main(snapshot, side).get(key)

    lines = [
        f"🕒 Changes {_snapshot_target(previous)} → {_snapshot_target(latest)}",
        "",
        f"Forecast HIGH: {_format_temp(previous.get('forecast', {}).get('high'))} → {_format_temp(latest.get('forecast', {}).get('high'))}",
        f"Forecast LOW: {_format_temp(previous.get('forecast', {}).get('low'))} → {_format_temp(latest.get('forecast', {}).get('low'))}",
        "",
        f"HIGH Main Bet: {_contract(previous, 'high', 'main')} → {_contract(latest, 'high', 'main')}",
        f"HIGH Lottery Bet: {_contract(previous, 'high', 'lottery')} → {_contract(latest, 'high', 'lottery')}",
        f"HIGH Market Price: {_format_percent(_metric(previous, 'high', 'market_probability'))} → {_format_percent(_metric(latest, 'high', 'market_probability'))}",
        f"HIGH Edge: {_format_signed(_metric(previous, 'high', 'edge'))} → {_format_signed(_metric(latest, 'high', 'edge'))}",
        f"HIGH EV: {_format_signed(_metric(previous, 'high', 'ev'))} → {_format_signed(_metric(latest, 'high', 'ev'))}",
        f"HIGH Kelly: {_format_currency(_metric(previous, 'high', 'kelly'))} → {_format_currency(_metric(latest, 'high', 'kelly'))}",
        "",
        f"LOW Main Bet: {_contract(previous, 'low', 'main')} → {_contract(latest, 'low', 'main')}",
        f"LOW Lottery Bet: {_contract(previous, 'low', 'lottery')} → {_contract(latest, 'low', 'lottery')}",
        f"LOW Market Price: {_format_percent(_metric(previous, 'low', 'market_probability'))} → {_format_percent(_metric(latest, 'low', 'market_probability'))}",
        f"LOW Edge: {_format_signed(_metric(previous, 'low', 'edge'))} → {_format_signed(_metric(latest, 'low', 'edge'))}",
        f"LOW EV: {_format_signed(_metric(previous, 'low', 'ev'))} → {_format_signed(_metric(latest, 'low', 'ev'))}",
        f"LOW Kelly: {_format_currency(_metric(previous, 'low', 'kelly'))} → {_format_currency(_metric(latest, 'low', 'kelly'))}",
    ]
    return "\n".join(lines)


def route_command(text):
    """Route a supported Telegram command to its snapshot-backed response."""
    text = (text or "").strip()
    if not text.startswith("/"):
        return None

    if re.fullmatch(r"/predict(?:@\w+)?\s+\d{4}-\d{2}-\d{2}", text):
        return None

    if re.fullmatch(r"/recommend(?:@\w+)?", text):
        snapshot = load_latest_snapshot()
        return handle_recommend(snapshot) if snapshot else "No prediction snapshot found yet. Run /predict YYYY-MM-DD first."

    why_match = re.fullmatch(r"/why(?:@\w+)?(?:\s+(high|low))?\s+(\d{1,2})", text, flags=re.IGNORECASE)
    if why_match:
        snapshot = load_latest_snapshot()
        if not snapshot:
            return "No prediction snapshot found yet. Run /predict YYYY-MM-DD first."
        side = why_match.group(1).lower() if why_match.group(1) else None
        temperature = int(why_match.group(2))
        return handle_why(snapshot, temperature, side=side)

    if re.fullmatch(r"/changes(?:@\w+)?", text):
        latest = load_latest_snapshot()
        previous = load_previous_snapshot()
        if not latest:
            return "No prediction snapshot found yet. Run /predict YYYY-MM-DD first."
        return handle_changes(latest, previous)

    return None


def handle_message(message):
    """Handle a single incoming Telegram message."""
    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = (message.get("text") or "").strip()

    # Keep the bot scoped to the existing alert chat.
    if str(chat_id) != str(TELEGRAM_CHAT_ID):
        return

    response = route_command(text)
    if response is not None:
        send_telegram(response)
        return

    if text.startswith("/") and not text.startswith("/predict"):
        reply_usage()
        return

    if not text.startswith("/predict"):
        return

    match = re.fullmatch(r"/predict(?:@\w+)?\s+(\d{4}-\d{2}-\d{2})", text)
    if not match:
        reply_usage()
        return

    try:
        target_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        reply_usage()
        return

    run(target_dates=[target_date])


def main():
    """Poll Telegram and dispatch supported commands."""
    logger.info("Starting Telegram receiver for /predict commands...")
    offset = None

    while True:
        try:
            for update in fetch_updates(offset=offset):
                offset = update.get("update_id", 0) + 1
                message = update.get("message")
                if message:
                    handle_message(message)
        except Exception as exc:
            logger.warning("Telegram polling error: %s", exc)
            time.sleep(ERROR_SLEEP_SECONDS)


if __name__ == "__main__":
    main()
