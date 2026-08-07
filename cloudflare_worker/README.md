# Polymarket Proxy — Cloudflare Worker

Proxy worker untuk bypass ISP blocking Polymarket di Indonesia.

## Cara Kerja

```
[Laptop Indonesia] → [Cloudflare Worker] → [Polymarket API]
     (diblokir)         (akses bebas)         (US server)
```

Worker menerima request di 2 route:
- `/gamma/*` → `gamma-api.polymarket.com/*`
- `/clob/*` → `clob.polymarket.com/*`

## Deploy (5 menit)

### 1. Install Wrangler CLI

```bash
npm install -g wrangler
```

### 2. Login ke Cloudflare

```bash
wrangler login
```

Browser akan terbuka untuk autentikasi.

### 3. Deploy Worker

```bash
cd cloudflare_worker/
wrangler deploy
```

Output akan menampilkan URL worker kamu:
```
Published polymarket-proxy
  https://polymarket-proxy.<your-subdomain>.workers.dev
```

### 4. Test Worker

Buka URL tersebut di browser atau:
```bash
curl https://polymarket-proxy.<your-subdomain>.workers.dev/health
```

Harus return:
```json
{"status":"ok","service":"polymarket-proxy","routes":["/gamma/*","/clob/*","/health"]}
```

### 5. Test API Proxy

```bash
# Test Gamma API
curl "https://polymarket-proxy.<your-subdomain>.workers.dev/gamma/events?slug=highest-temperature-in-hong-kong-on-august-7-2026"

# Test CLOB API
curl "https://polymarket-proxy.<your-subdomain>.workers.dev/clob/prices-history?market=CONDITION_ID&interval=all"
```

### 6. Konfigurasi Script

Edit `polymarket_scraper.py`, set `PROXY_URL`:

```python
PROXY_URL = "https://polymarket-proxy.<your-subdomain>.workers.dev"
```

Atau via environment variable:

```python
import os
PROXY_URL = os.environ.get("POLYMARKET_PROXY", "")
```

## Fitur

- **Rate limiting**: 60 request/menit per IP
- **CORS enabled**: Bisa dipanggil dari browser/Streamlit
- **Zero config**: Tidak perlu server, Cloudflare handle semuanya
- **Free tier**: 100,000 request/hari (cukup untuk hourly alerts)

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `wrangler: command not found` | `npm install -g wrangler` |
| Auth error | `wrangler login` ulang |
| 502 Bad Gateway | Polymarket API mungkin down, coba lagi |
| Rate limit (429) | Tunggu 1 menit, kurangi request frequency |

## Security (Optional)

Untuk membatasi akses hanya dari script kamu, tambahkan secret token:

1. Di `wrangler.toml`, uncomment `PROXY_TOKEN`
2. Di `worker.js`, tambahkan validasi token di header
3. Di `polymarket_scraper.py`, kirim token di header request
