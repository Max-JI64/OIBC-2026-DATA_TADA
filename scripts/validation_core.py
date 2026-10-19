"""S0 time, sample, left-join and local settlement contracts. No model imports."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KST = "Asia/Seoul"
ROUNDS = (10, 17)
KEY = ["site_id", "ts"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def times(values):
    result = pd.DatetimeIndex(pd.to_datetime(values))
    require(result.tz is not None, "Timezone-aware timestamps required")
    require(not result.hasnans, "Missing timestamp")
    return result.tz_convert(KST)


def hourly(values):
    result = times(values)
    require((result == result.floor("h")).all(), "Non-hourly target timestamp")
    return result


def clock(values):
    ts = hourly(values)
    start = ts - pd.Timedelta(hours=1)
    day = start.floor("D")
    return pd.DataFrame({"ts": ts, "start": start, "target_day": day,
                         "prediction_at": day - pd.Timedelta(days=1) + pd.Timedelta(hours=17)})


def weights(values):
    ts = hourly(values)
    return np.where((ts.hour >= 11) & (ts.hour <= 15), 3, 1)


@dataclass(frozen=True)
class Window:
    name: str
    start: str
    end: str
    train_start: str
    cutoff: str

    def axis(self):
        return pd.date_range(pd.Timestamp(self.start, tz=KST) + pd.Timedelta(hours=1),
                             pd.Timestamp(self.end, tz=KST), freq="h")

    def training(self, gen):
        start = times(gen.ts) - pd.Timedelta(hours=1)
        cutoff = pd.Timestamp(self.cutoff, tz=KST)
        require(cutoff < clock(self.axis()).prediction_at.min(), "Training cutoff leakage")
        return gen.loc[(start >= pd.Timestamp(self.train_start, tz=KST)) &
                       (times(gen.ts) <= cutoff)].copy()


WINDOWS = (
    Window("V1", "2024-01-01", "2025-01-01", "2022-01-01", "2023-12-31"),
    Window("V2", "2025-01-01", "2025-06-01", "2023-01-01", "2024-12-31"),
)


def unique(frame, keys):
    require(not frame[keys].isna().any().any(), f"Missing keys: {keys}")
    require(not frame.duplicated(keys).any(), f"Duplicate keys: {keys}")


def metadata(sites):
    unique(sites, ["site_id"])
    require(len(sites) == 253, "Expected 253 sites")
    require(set(sites.current_group) == {"A", "B"}, "Unexpected reference groups")
    require(np.isfinite(sites.capacity_kw).all() and sites.capacity_kw.gt(0).all(), "Invalid capacity")
    require(sites[["wa_station", "wb_station", "lat", "lon"]].notna().all().all(), "Missing metadata")
    require(np.isfinite(sites[["lat", "lon"]]).all().all(), "Invalid coordinates")
    capacity = sites.groupby("current_group").capacity_kw.sum()
    require(capacity.ge(20000).all(), "Reference capacity constraint failed")
    return capacity


def grid(sites, axis):
    axis = hourly(axis)
    require(axis.is_unique, "Duplicate grid times")
    unique(sites, ["site_id"])
    base = pd.MultiIndex.from_product([sites.site_id, axis], names=KEY).to_frame(index=False)
    return base.merge(sites, on="site_id", how="left", validate="many_to_one", sort=False)


def prepare_forecast(frame, station, features):
    """Keep exact hours only, never round; reject duplicate source keys."""
    frame = frame[[station, "round", "ts"] + list(features)].copy()
    frame["ts"] = times(frame.ts)
    unique(frame, [station, "round", "ts"])
    require(frame["round"].isin(ROUNDS).all(), "Unexpected forecast round")
    exact = frame.ts.eq(frame.ts.dt.floor("h"))
    return frame.loc[exact].copy(), int((~exact).sum())


def join_forecasts(base, fa, fb, r):
    require(r in ROUNDS, "Unexpected round")
    unique(base, KEY)
    original = pd.MultiIndex.from_frame(base[KEY])
    d = base.copy()
    for prefix, station, forecast in (("wa", "wa_station", fa), ("wb", "wb_station", fb)):
        part = forecast.loc[forecast["round"].eq(r)].drop(columns="round")
        unique(part, [station, "ts"])
        require((times(part.ts) == times(part.ts).floor("h")).all(), "Unfiltered non-hourly forecast")
        part = part.assign(**{prefix + "_row_present": True})
        d = d.merge(part, on=[station, "ts"], how="left", validate="many_to_one", sort=False)
        d[prefix + "_row_present"] = d[prefix + "_row_present"].eq(True)
    require(len(d) == len(base) and pd.MultiIndex.from_frame(d[KEY]).equals(original), "Join changed grid")
    c = clock(d.ts)
    for round_hour in (r,):
        issued = c.target_day - pd.Timedelta(days=1) + pd.Timedelta(hours=round_hour)
        require(issued.le(c.prediction_at).all(), "Forecast issuance after prediction time")
    return d


def truth_matrix(gen, ids, axis):
    """The only missing-truth policy is NaN, retained until a complete mask exists."""
    unique(gen, KEY)
    axis = hourly(axis)
    ids = list(ids)
    require(len(set(ids)) == len(ids), "Duplicate truth sites")
    part = gen.loc[gen.site_id.isin(ids) & gen.ts.isin(axis)]
    require(np.isfinite(part.energy).all(), "Nonfinite recorded energy")
    return part.pivot(index="ts", columns="site_id", values="energy").reindex(index=axis, columns=ids)


def group_totals(matrix, groups):
    require(matrix.columns.is_unique and matrix.index.is_unique, "Duplicate matrix keys")
    groups = groups.reindex(matrix.columns)
    require(groups.notna().all() and set(groups) == {"G1", "G2"}, "Missing group assignments")
    complete = matrix.notna().all(axis=1)
    require(np.isfinite(matrix.loc[complete].to_numpy()).all(), "Nonfinite complete values")
    total = pd.DataFrame(index=matrix.index[complete])
    for group in ("G1", "G2"):
        members = groups.index[groups.eq(group)]
        total[group] = matrix.loc[complete, members].sum(axis=1, min_count=len(members))
    return total, complete


def local_score(truth, prediction, groups, price, sample_axis):
    """Score a pre-fixed common-time sample. Never silently intersect predictions."""
    sample_axis = hourly(sample_axis)
    require(sample_axis.is_unique and len(sample_axis) > 0, "Empty/duplicate score sample")
    require(truth.index.equals(prediction.index) and truth.columns.equals(prediction.columns),
            "Prediction axes differ from the fixed truth grid")
    require(np.isfinite(prediction.to_numpy()).all() and prediction.ge(0).all().all(),
            "Predictions must cover the full grid with finite nonnegative values")
    actual, complete = group_totals(truth, groups)
    require(sample_axis.isin(actual.index).all(), "Fixed score sample has incomplete truth")
    pred, pred_complete = group_totals(prediction, groups)
    require(pred_complete.all(), "Missing predictions")
    unique(price, ["ts"])
    p = price.set_index("ts").reindex(sample_axis).smp
    require(np.isfinite(p).all() and p.ge(0).all(), "Missing/invalid score prices")
    k = weights(sample_axis)
    a = actual.loc[sample_axis]
    y = pred.loc[sample_axis]
    error = (y - a).abs()
    w = float(error.mul(p * k, axis=0).to_numpy().sum())
    revenue = float(a.sum(axis=1).mul(p).sum())
    denominator = float(2 * (p * k).sum())
    return {"times": len(sample_axis), "W_local": w, "R_local": revenue,
            "S_local": revenue - 0.10 * w, "price_k_sum": float((p * k).sum()),
            "weighted_group_mae_kwh": w / denominator if denominator else None,
            "group_mae_kwh": float(error.to_numpy().mean())}
