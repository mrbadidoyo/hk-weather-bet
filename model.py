"""
Temperature Prediction Model - Ensemble of LightGBM + XGBoost + quantile regression
Predicts daily high and low temperatures with uncertainty estimates
"""
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

from config import MODEL_PARAMS, MODELS_DIR, TRAIN_END_YEAR

logger = logging.getLogger(__name__)


class TemperaturePredictor:
    """
    Ensemble model for predicting daily high/low temperatures.
    Uses LightGBM + XGBoost with quantile regression for uncertainty.
    """

    def __init__(self, target_name: str = "max_temp"):
        self.target_name = target_name
        self.models = {}
        self.quantile_models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def _get_feature_cols(self, df: pd.DataFrame, target_col: str) -> list[str]:
        """Get feature column names (exclude target and metadata)"""
        exclude = {
            target_col,
            "wu_max_temp_c", "wu_min_temp_c", "wu_avg_temp_c",
            "season",  # categorical
        }
        cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32, float, int]]
        return cols

    def prepare_data(self, df: pd.DataFrame, target_col: str):
        """Prepare training data with feature selection and cleaning"""
        feature_cols = self._get_feature_cols(df, target_col)
        self.feature_names = feature_cols

        # Drop rows with NaN in target
        mask = df[target_col].notna()
        # Also require some key features to be present
        for col in feature_cols[:5]:  # first 5 features must be non-NaN
            mask &= df[col].notna()

        X = df.loc[mask, feature_cols].copy()
        y = df.loc[mask, target_col].copy()

        # Fill remaining NaN in features
        X = X.ffill().bfill().fillna(0)

        logger.info(f"  Prepared data: {len(X)} samples, {len(feature_cols)} features")
        logger.info(f"  Target stats: mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.2f}, max={y.max():.2f}")

        return X, y

    def fit(self, df: pd.DataFrame, target_col: str):
        """Train the ensemble model"""
        import lightgbm as lgb
        import xgboost as xgb

        X, y = self.prepare_data(df, target_col)

        # Time-series cross-validation split
        n = len(X)
        train_end = int(n * 0.8)
        X_train, X_val = X.iloc[:train_end], X.iloc[train_end:]
        y_train, y_val = y.iloc[:train_end], y.iloc[train_end:]

        # 1. LightGBM point prediction
        logger.info("  Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(**MODEL_PARAMS["lightgbm"], random_state=42, verbose=-1)
        lgb_model.fit(
            X_train, y_train,
            eval_X=X_val, eval_y=y_val,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        self.models["lightgbm"] = lgb_model

        # 2. XGBoost point prediction
        logger.info("  Training XGBoost...")
        xgb_model = xgb.XGBRegressor(**MODEL_PARAMS["xgboost"], random_state=42, verbosity=0)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self.models["xgboost"] = xgb_model

        # 3. Quantile regression for uncertainty estimation
        logger.info("  Training quantile models (10th, 25th, 50th, 75th, 90th)...")
        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
        for q in quantiles:
            q_model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            )
            q_model.fit(
                X_train, y_train,
                eval_X=X_val, eval_y=y_val,
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
            )
            self.quantile_models[q] = q_model

        # 4. Evaluate
        self.is_fitted = True  # set early so _predict_ensemble works
        logger.info("  Evaluating on validation set...")
        val_preds = self._predict_ensemble(X_val)
        mae = mean_absolute_error(y_val, val_preds["point"])
        rmse = np.sqrt(mean_squared_error(y_val, val_preds["point"]))
        r2 = r2_score(y_val, val_preds["point"])
        logger.info(f"  Validation MAE: {mae:.2f}°C, RMSE: {rmse:.2f}°C, R²: {r2:.4f}")

        # Feature importance
        importances = lgb_model.feature_importances_
        top_features = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1], reverse=True
        )[:15]
        logger.info(f"  Top 15 features:")
        for feat, imp in top_features:
            logger.info(f"    {feat}: {imp}")

        self.is_fitted = True
        return {"mae": mae, "rmse": rmse, "r2": r2}

    def _predict_ensemble(self, X: pd.DataFrame) -> dict:
        """Generate ensemble predictions with uncertainty"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")

        X = X[self.feature_names].ffill().bfill().fillna(0)

        # Point predictions from each model
        lgb_pred = self.models["lightgbm"].predict(X)
        xgb_pred = self.models["xgboost"].predict(X)

        # Ensemble: weighted average (LightGBM typically performs better)
        point_pred = 0.6 * lgb_pred + 0.4 * xgb_pred

        # Quantile predictions
        quantile_preds = {}
        for q, model in self.quantile_models.items():
            quantile_preds[q] = model.predict(X)

        # Estimate uncertainty from quantile spread
        p10 = quantile_preds.get(0.10, point_pred - 2)
        p90 = quantile_preds.get(0.90, point_pred + 2)
        spread = p90 - p10

        return {
            "point": point_pred,
            "lgb_pred": lgb_pred,
            "xgb_pred": xgb_pred,
            "p10": p10,
            "p25": quantile_preds.get(0.25, point_pred - 1),
            "p50": quantile_preds.get(0.50, point_pred),
            "p75": quantile_preds.get(0.75, point_pred + 1),
            "p90": p90,
            "spread": spread,
        }

    def predict_distribution(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict temperature distribution for each day.
        Returns a DataFrame with mean, std, and quantile estimates.
        """
        preds = self._predict_ensemble(X)

        result = pd.DataFrame({
            "mean": preds["point"],
            "std": preds["spread"] / 3.28,  # approximate std from 10th-90th range
            "p10": preds["p10"],
            "p25": preds["p25"],
            "p50": preds["p50"],
            "p75": preds["p75"],
            "p90": preds["p90"],
        }, index=X.index if hasattr(X, 'index') else range(len(X)))

        return result

    def save(self, path: Path = None):
        """Save model to disk"""
        if path is None:
            path = MODELS_DIR / f"temp_predictor_{self.target_name}.joblib"
        joblib.dump({
            "models": self.models,
            "quantile_models": self.quantile_models,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
        }, path)
        logger.info(f"Model saved -> {path}")

    def load(self, path: Path = None):
        """Load model from disk"""
        if path is None:
            path = MODELS_DIR / f"temp_predictor_{self.target_name}.joblib"
        data = joblib.load(path)
        self.models = data["models"]
        self.quantile_models = data["quantile_models"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.target_name = data["target_name"]
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")


class HKWeatherEnsemble:
    """
    Complete ensemble predictor for HK daily high and low temperatures.
    Combines:
    1. ML models (LightGBM + XGBoost)
    2. HKO official forecast (when available)
    3. Climatological normals
    4. Persistence forecast
    """

    def __init__(self):
        self.max_predictor = TemperaturePredictor("max_temp")
        self.min_predictor = TemperaturePredictor("min_temp")
        self.hko_forecast = None

    def fit(self, feature_df: pd.DataFrame, target_max_col: str, target_min_col: str):
        """Train both high and low temperature predictors"""
        logger.info("=" * 60)
        logger.info("Training High Temperature Model")
        logger.info("=" * 60)
        max_metrics = self.max_predictor.fit(feature_df, target_max_col)

        logger.info("=" * 60)
        logger.info("Training Low Temperature Model")
        logger.info("=" * 60)
        min_metrics = self.min_predictor.fit(feature_df, target_min_col)

        return {"max_temp": max_metrics, "min_temp": min_metrics}

    def predict(
        self,
        feature_df: pd.DataFrame,
        hko_forecast_max: float = None,
        hko_forecast_min: float = None,
    ) -> dict:
        """
        Generate predictions for the given dates.
        Blends ML predictions with HKO forecast when available.
        """
        # ML predictions
        max_dist = self.max_predictor.predict_distribution(feature_df)
        min_dist = self.min_predictor.predict_distribution(feature_df)

        # Blend with HKO forecast if available
        if hko_forecast_max is not None:
            # Weight: 60% ML, 40% HKO forecast (HKO is quite accurate)
            max_dist["mean"] = 0.6 * max_dist["mean"] + 0.4 * hko_forecast_max
            max_dist["std"] *= 0.8  # reduce uncertainty when we have official forecast

        if hko_forecast_min is not None:
            min_dist["mean"] = 0.6 * min_dist["mean"] + 0.4 * hko_forecast_min
            min_dist["std"] *= 0.8

        return {
            "max_temp": max_dist,
            "min_temp": min_dist,
        }

    def predict_today(self, feature_df: pd.DataFrame, hko_forecast: dict = None) -> dict:
        """Predict today's high and low with all available information"""
        hko_max = None
        hko_min = None

        if hko_forecast:
            try:
                today_str = datetime.now().strftime("%Y%m%d")
                for day in hko_forecast.get("weatherForecast", []):
                    if day.get("forecastDate") == today_str:
                        hko_max = float(day["forecastMaxtemp"]["value"])
                        hko_min = float(day["forecastMintemp"]["value"])
                        break
            except (KeyError, TypeError, ValueError):
                pass

        # Get last row for prediction
        last_row = feature_df.tail(1)
        return self.predict(last_row, hko_max, hko_min)

    def save(self):
        """Save both models"""
        self.max_predictor.save()
        self.min_predictor.save()

    def load(self):
        """Load both models"""
        self.max_predictor.load()
        self.min_predictor.load()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Quick test with synthetic data
    dates = pd.date_range("2020-01-01", "2025-12-31")
    n = len(dates)
    doy = dates.dayofyear
    seasonal = 5 * np.sin(2 * np.pi * (doy - 80) / 365.25)

    sample = pd.DataFrame({
        "hko_max_temp": 27 + seasonal + np.random.normal(0, 2, n),
        "hko_min_temp": 22 + seasonal * 0.7 + np.random.normal(0, 1.5, n),
        "hko_mean_temp": 24.5 + seasonal * 0.85 + np.random.normal(0, 1.8, n),
        "hko_rh": 75 + 10 * np.sin(2 * np.pi * (doy - 120) / 365.25) + np.random.normal(0, 8, n),
        "wu_max_temp_c": 27 + seasonal + np.random.normal(0, 2.1, n),
        "wu_min_temp_c": 22 + seasonal * 0.7 + np.random.normal(0, 1.6, n),
    }, index=dates)

    from features import build_feature_matrix
    features = build_feature_matrix(sample)

    # Train
    ensemble = HKWeatherEnsemble()
    metrics = ensemble.fit(features, "wu_max_temp_c", "wu_min_temp_c")
    print(f"\nMetrics: {metrics}")

    # Predict last 7 days
    preds = ensemble.predict(features.tail(7))
    print(f"\nPredictions (last 7 days):")
    print(preds["max_temp"])
    print(preds["min_temp"])
