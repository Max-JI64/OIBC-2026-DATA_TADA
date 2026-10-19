"""Read-only timestamp precision check; does not correct or round source data."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eda" / "2026-10-19_basic_shape"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path = ROOT / "OIBC_2026_DATA" / "01_participant_dataset" / "train" / "forecast_wb.parquet"
    d = pd.read_parquet(path, columns=["ts", "round", "wb_station"])
    ts = d.ts
    offset = (ts.dt.minute.astype("int64") * 60_000_000_000 +
              ts.dt.second.astype("int64") * 1_000_000_000 +
              ts.dt.microsecond.astype("int64") * 1000 + ts.dt.nanosecond.astype("int64"))
    mask = offset != 0
    unique = pd.DatetimeIndex(ts.unique())
    distance = np.minimum(offset[mask].to_numpy(), 3_600_000_000_000 - offset[mask].to_numpy())
    nearest = unique.round("h")
    result = {"file": "train/forecast_wb.parquet", "rows": len(d),
              "off_hour_rows": int(mask.sum()), "off_hour_distinct_timestamps": int(ts[mask].nunique()),
              "off_hour_round_counts": {str(k): int(v) for k, v in d.loc[mask, "round"].value_counts().items()},
              "off_hour_stations": int(d.loc[mask, "wb_station"].nunique()),
              "distance_to_nearest_hour_ns_min": int(distance.min()),
              "distance_to_nearest_hour_ns_max": int(distance.max()),
              "timestamp_dtype": str(ts.dtype), "exact_unique_timestamps": len(unique),
              "hypothetical_rounded_unique_timestamps": int(nearest.nunique()),
              "rounding_used_only_for_diagnosis": True, "source_modified": False}
    result["components"] = {name: {"min": int(getattr(ts.dt, name).min()),
                                    "max": int(getattr(ts.dt, name).max())}
                            for name in ["minute", "second", "microsecond", "nanosecond"]}
    raw = pq.read_table(path, columns=["ts"])["ts"]
    # KST differs from UTC by an integer number of hours. Integer remainders
    # therefore check the source time unit without per-row timezone conversion.
    ticks_per_second = {"s": 1, "ms": 1000, "us": 1_000_000, "ns": 1_000_000_000}[raw.type.unit]
    raw_ticks = raw.cast(pa.int64()).to_numpy(zero_copy_only=False)
    remainder = np.remainder(raw_ticks, 3600 * ticks_per_second)
    result["arrow_off_hour_rows"] = int((remainder != 0).sum())
    result["arrow_ticks_from_hour_min"] = int(remainder.min())
    result["arrow_ticks_from_hour_max"] = int(remainder.max())
    result["arrow_tick_unit"] = raw.type.unit
    off = remainder[remainder != 0]
    result["arrow_distance_to_hour_ns_min"] = int(np.minimum(off, 3600 * ticks_per_second - off).min() * (1_000_000_000 // ticks_per_second))
    result["arrow_distance_to_hour_ns_max"] = int(np.minimum(off, 3600 * ticks_per_second - off).max() * (1_000_000_000 // ticks_per_second))
    result["diagnostic_timestamp_strings"] = ts[mask].astype(str).drop_duplicates().head(3).tolist()
    (OUT / "time_precision.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
