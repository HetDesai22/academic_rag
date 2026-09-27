"""Forecasting methods for per-topic question counts in the next exam period.

Each statistical method maps a history (counts over past periods) to a non-negative forecast.
The Random Forest is a pooled model across all topic series (lag features), used only when
enough observations exist. No deep learning: the series are short (about 10 periods).
"""
import numpy as np


def historical_mean(h, **_):
    return float(np.mean(h)) if len(h) else 0.0


def moving_average(h, window=3, **_):
    return float(np.mean(h[-window:])) if len(h) else 0.0


def _ses_fit_alpha(h):
    best, best_a = np.inf, 0.3
    for a in np.linspace(0.1, 0.9, 9):
        level, sse = h[0], 0.0
        for y in h[1:]:
            sse += (y - level) ** 2
            level = a * y + (1 - a) * level
        if sse < best:
            best, best_a = sse, a
    return best_a


def exp_smoothing(h, alpha=None, **_):
    h = np.asarray(h, dtype=float)
    if len(h) == 0:
        return 0.0
    a = alpha if alpha is not None else (_ses_fit_alpha(h) if len(h) >= 3 else 0.5)
    level = h[0]
    for y in h[1:]:
        level = a * y + (1 - a) * level
    return float(level)


def linear_trend(h, **_):
    h = np.asarray(h, dtype=float)
    if len(h) < 3:
        return historical_mean(h)
    slope, intercept = np.polyfit(np.arange(len(h)), h, 1)
    return float(max(0.0, intercept + slope * len(h)))


def seasonal_mean(h, seasons=None, target_season=None, **_):
    """Mean of past periods in the same exam session (Winter vs Summer)."""
    if seasons is None or target_season is None:
        return historical_mean(h)
    same = [y for y, s in zip(h, seasons) if s == target_season]
    return float(np.mean(same)) if same else historical_mean(h)


STAT_METHODS = {
    "historical_mean": historical_mean,
    "moving_average": moving_average,
    "exp_smoothing": exp_smoothing,
    "linear_trend": linear_trend,
    "seasonal_mean": seasonal_mean,
}


def features(h, seasons, target_season, subject_share):
    """Lag features for the pooled model (history h -> predict next)."""
    h = np.asarray(h, dtype=float)
    lag = lambda k: h[-k] if len(h) >= k else 0.0  # noqa: E731
    return [lag(1), lag(2), lag(3), h.mean() if len(h) else 0.0, h[-3:].mean() if len(h) else 0.0,
            exp_smoothing(h, 0.5), (h > 0).mean() if len(h) else 0.0, len(h),
            1.0 if target_season == "W" else 0.0, seasonal_mean(h, seasons, target_season), subject_share]


class PooledRandomForest:
    name = "random_forest"

    def __init__(self, seed=42):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)
        self.fitted = False

    @staticmethod
    def training_set(series, seasons_by_key, upto, min_hist=3):
        """series: key -> counts array; seasons_by_key: key -> session letters.
        Build (X, y) from targets at indices < upto."""
        X, y = [], []
        for key, s in series.items():
            seasons = seasons_by_key[key]
            share = s[:upto].sum() / max(1, sum(v[:upto].sum() for k, v in series.items() if k[0] == key[0]))
            for t in range(min_hist, min(upto, len(s))):
                X.append(features(s[:t], seasons[:t], seasons[t], share))
                y.append(s[t])
        return np.array(X), np.array(y)

    def fit(self, X, y):
        if len(y) == 0:
            return self
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict_one(self, h, seasons, target_season, share):
        if not self.fitted:
            return historical_mean(h)
        return float(max(0.0, self.model.predict([features(h, seasons, target_season, share)])[0]))
