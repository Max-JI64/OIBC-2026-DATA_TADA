"""Shared read-only panel views for relationship EDA."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from basic_shape_eda import ROOT,SOURCE,digest
from eda_plot_utils import write_gallery,check_gallery


def load_panel():
    sites=pd.read_csv(SOURCE/'sites.csv',encoding='utf-8-sig').set_index('site_id')
    gen=pd.read_parquet(SOURCE/'train/generation.parquet')
    assert not gen.duplicated(['site_id','ts']).any()
    start=gen.ts-pd.Timedelta(hours=1)
    sub=gen.loc[start.dt.year==2024]
    count=sub.groupby('site_id').size();ids=sorted(count[count==8784].index)
    assert len(ids)==148
    energy=sub.loc[sub.site_id.isin(ids)].pivot(index='ts',columns='site_id',values='energy').sort_index().reindex(columns=ids)
    assert energy.shape==(8784,148) and energy.notna().all().all() and energy.ge(0).all().all()
    cf=energy.div(sites.loc[ids,'capacity_kw'],axis=1)
    return sites,gen,energy,cf


def clock(index):
    start=index-pd.Timedelta(hours=1)
    return pd.DataFrame({'month':start.month,'hour':start.hour,'day':start.floor('D')},index=index)


def centered(frame):
    t=clock(frame.index)
    return frame-frame.groupby([t.month,t.hour],observed=True).transform('mean')


def hashes(files):return {p:digest(SOURCE/p) for p in files}


def finish(out,info,files,before,tables,gallery,title):
    assert hashes(files)==before
    info.update(sha256=before,finished_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes'))
    for name,frame in tables.items():frame.to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    (out/'summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    write_gallery(out,gallery,title)


def verify_base(out,report,files,figures):
    info=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert info['sha256']==hashes(files)
    assert '\u00b7' not in report.read_text(encoding='utf-8')
    return info,{**check_gallery(out,figures),'source_unchanged':True}


def save_check(out,check):
    (out/'verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    print(json.dumps(check))


def haversine(lat,lon):
    a=np.deg2rad(np.asarray(lat));b=np.deg2rad(np.asarray(lon))
    v=np.sin((a[:,None]-a)/2)**2+np.cos(a[:,None])*np.cos(a)*np.sin((b[:,None]-b)/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(v,0,1)))
