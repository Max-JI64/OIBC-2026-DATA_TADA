"""Investigate the existing dataset audit findings without changing source data."""
from pathlib import Path
import json
import pandas as pd

root = Path(__file__).resolve().parents[1]
data = root / "OIBC_2026_DATA" / "01_participant_dataset"
out = root / "데이터_확인"
report = json.loads((out / "데이터_검증결과.json").read_text(encoding="utf-8"))
d = pd.read_parquet(data / "train/observed_wa.parquet")
dups = d[d.duplicated(["wa_station", "ts"], keep=False)].sort_values(["wa_station", "ts"])
dups.to_csv(out / "A실측_중복키_원본행.csv", index=False, encoding="utf-8-sig")
wa = {"duplicate_key_groups":int(dups.groupby(["wa_station","ts"]).ngroups),
      "participating_rows":len(dups),"excess_identical_full_rows":int(dups.duplicated().sum()),
      "groups_with_conflicting_values":int(sum(len(g.drop_duplicates())>1 for _,g in dups.groupby(["wa_station","ts"]))),
      "wind_speed_gt_50":int((d.wa_wind_speed>50).sum())}
del d
d = pd.read_parquet(data / "train/generation.parquet")
neg = d[d.energy < 0]
neg.to_csv(out / "발전량_음수_원본행.csv", index=False, encoding="utf-8-sig")
energy = {"negative_rows":len(neg), "negative_sites":int(neg.site_id.nunique()),
          "negative_min":float(neg.energy.min()),"negative_max":float(neg.energy.max()),
          "missing_full_axis_pairs":report["files"]["train/generation.parquet"]["absent_pairs_on_full_training_axis"]}
del d
d = pd.read_parquet(data / "train/observed_wb.parquet", columns=["wb_cloud","wb_uv_idx","wb_vis","wb_wind_speed","wb_pressure"])
wb = {"cloud_gt_100":int((d.wb_cloud>100).sum()),"uv_gt_15":int((d.wb_uv_idx>15).sum()),
      "vis_gt_50":int((d.wb_vis>50).sum()),"wind_gt_200":int((d.wb_wind_speed>200).sum()),
      "pressure_lt_900":int((d.wb_pressure<900).sum())}
wa_cols = "wa_cloud wa_temp wa_temp_max wa_temp_min wa_humidity wa_ground_press wa_wind_speed wa_wind_dir wa_rain wa_snow".split()
wb_fc = "wb_temp wb_feel_temp wb_wet_bulb_temp wb_dew_point wb_wind_speed wb_wind_dir wb_wind_gust_spd wb_rel_hum wb_vis wb_ceiling wb_uv_idx wb_precip_prob wb_rain_prob wb_snow_prob wb_ice_prob wb_total_liq wb_rain wb_snow wb_ice wb_cloud".split()
wb_obs = "wb_temp wb_feel_temp wb_feel_temp_shade wb_rel_hum wb_dew_point wb_wind_dir wb_wind_speed wb_uv_idx wb_vis wb_cloud wb_ceiling wb_pressure wb_appr_temp wb_wind_chill_temp wb_precip_1h".split()
expected = {"sites.csv":"site_id current_group capacity_kw lat lon wa_station wb_station wa_dist wb_dist".split(),
            "smp.csv":["ts","smp","k"],"train/smp.csv":["ts","smp"],
            "train/generation.parquet":["site_id","ts","energy"]}
for split in ("train","test"):
    expected[f"{split}/forecast_wa.parquet"] = ["wa_station","round","ts"]+wa_cols
    expected[f"{split}/forecast_wb.parquet"] = ["wb_station","round","ts"]+wb_fc
expected["train/observed_wa.parquet"] = ["wa_station","ts"]+wa_cols
expected["train/observed_wb.parquet"] = ["wb_station","ts"]+wb_obs
schemas = {k:{"matches_readme_column_set":set(v)==set(report["files"][k]["columns"]),
              "missing":sorted(set(v)-set(report["files"][k]["columns"])),
              "extra":sorted(set(report["files"][k]["columns"])-set(v))} for k,v in expected.items()}
findings = {"observed_wa":wa,"generation":energy,"observed_wb":wb,"schemas":schemas}
(out / "추가확인.json").write_text(json.dumps(findings,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(findings,ensure_ascii=False,indent=2))
