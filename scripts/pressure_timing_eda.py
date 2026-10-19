"""EDA-001: bounded descriptive check of supplied A pressure timing.

Three figures, one descriptive boundary candidate, one before/after comparison.
No source correction, model fitting, external lookup, or causal attribution.
"""
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from basic_shape_eda import ROOT, SOURCE, digest, fmt, md_table, log
from eda_plot_utils import save_plot, write_gallery, check_gallery

OUT = ROOT / 'data/eda/2026-10-19_pressure_timing'
REPORT = ROOT / 'reports/2026-10-19_17_A지표기압_변화시점_EDA.md'
FILES = ['sites.csv', 'train/forecast_wa.parquet', 'test/forecast_wa.parquet']
CONFIG = [('train', '학습 비교', '2024-11-01', '2025-06-01'),
          ('test', '평가', '2025-11-01', '2026-06-01')]
COLORS = ['#26718c', '#d7804f']
GALLERY = []


def load_period(split, label, lo, hi, stations):
    data = pd.read_parquet(SOURCE / f'{split}/forecast_wa.parquet',
                           columns=['wa_station', 'round', 'ts', 'wa_ground_press'])
    begin, end = pd.Timestamp(lo, tz='Asia/Seoul'), pd.Timestamp(hi, tz='Asia/Seoul')
    data = data.loc[data.ts.gt(begin) & data.ts.le(end)].copy()
    assert set(data.wa_station) == set(stations) and set(data['round']) == {10, 17}
    assert not data.duplicated(['wa_station', 'round', 'ts']).any()
    assert data.ts.eq(data.ts.dt.floor('h')).all()
    assert str(data.ts.dt.tz) == 'Asia/Seoul'
    assert np.isfinite(data.wa_ground_press).all()
    start = data.ts - pd.Timedelta(hours=1)
    data['date'] = start.dt.normalize()
    days = pd.date_range(begin, end - pd.Timedelta(days=1), freq='D')
    index = pd.MultiIndex.from_product([[10, 17], stations, days], names=['round', 'wa_station', 'date'])
    grouped = data.groupby(['round', 'wa_station', 'date'], observed=True).wa_ground_press
    daily = grouped.agg(mean='mean', minimum='min', maximum='max', rows='size', finite='count').reindex(index)
    daily[['rows', 'finite']] = daily[['rows', 'finite']].fillna(0).astype(int)
    daily['complete'] = daily.rows.eq(24) & daily.finite.eq(24)
    # Incomplete days are left missing, rather than mixed with complete days.
    daily['level'] = daily['mean'].where(daily.complete)
    daily = daily.reset_index()
    daily['period'] = label
    center = daily.groupby(['round', 'date'], observed=True)['level'].transform('median')
    daily['relative'] = daily.level - center
    audit = []
    for r in [10, 17]:
        z = daily.loc[daily['round'].eq(r)]
        original = data.loc[data['round'].eq(r), 'wa_ground_press']
        audit.append({'period': label, 'round': r, 'days': len(days), 'stations': len(stations),
                      'expected_rows': len(days)*len(stations)*24, 'actual_rows': int(z.rows.sum()),
                      'missing_rows': len(days)*len(stations)*24-int(z.rows.sum()),
                      'finite_rows': int(z.finite.sum()), 'complete_station_days': int(z.complete.sum()),
                      'incomplete_station_days': int((~z.complete).sum()),
                      'minimum': float(original.min()), 'maximum': float(original.max())})
    log(label + ' daily pressure complete')
    return daily, audit


def aggregate(daily):
    return daily.groupby(['period', 'round', 'date'], observed=True).agg(
        stations=('level', 'count'), mean_level=('level', 'mean'), median_level=('level', 'median'),
        level_q10=('level', lambda x: x.quantile(.1)), level_q90=('level', lambda x: x.quantile(.9)),
        minimum=('level', 'min'), maximum=('level', 'max')).reset_index()


def describe_change(daily):
    reference = daily.loc[daily.period.eq('학습 비교')].groupby(['round', 'wa_station'], observed=True).relative.mean()
    target = daily.loc[daily.period.eq('평가')].copy()
    base = pd.MultiIndex.from_frame(target[['round', 'wa_station']])
    target['reference_relative_mean'] = reference.reindex(base).to_numpy()
    target['relative_departure'] = target.relative - target.reference_relative_mean
    target = target.sort_values(['round', 'wa_station', 'date'])
    group = target.groupby(['round', 'wa_station'], observed=True)
    # The grid contains every calendar day, so shift(1) never bridges a gap.
    target['relative_day_change'] = group.relative.diff()
    changes = target.groupby(['round', 'date'], observed=True).agg(
        n_changes=('relative_day_change', 'count'),
        median_abs_change=('relative_day_change', lambda x: x.abs().median()),
        median_change=('relative_day_change', 'median'),
        change_q10=('relative_day_change', lambda x: x.quantile(.1)),
        change_q90=('relative_day_change', lambda x: x.quantile(.9)))
    # Keep candidates with the same full set of station changes in both rounds.
    score = changes.median_abs_change.where(changes.n_changes.eq(61)).unstack('round').dropna()
    score['mean_round_score'] = score[[10, 17]].mean(axis=1)
    candidate = score.mean_round_score.idxmax()
    before = target.loc[target.date.ge(candidate-pd.Timedelta(days=7)) & target.date.lt(candidate)]
    after = target.loc[target.date.ge(candidate) & target.date.lt(candidate+pd.Timedelta(days=7))]
    rows = []
    for r in [10, 17]:
        b = before.loc[before['round'].eq(r)].groupby('wa_station').agg(
            before=('relative', 'mean'), before_level=('level', 'mean'), before_days=('level', 'count'))
        a = after.loc[after['round'].eq(r)].groupby('wa_station').agg(
            after=('relative', 'mean'), after_level=('level', 'mean'), after_days=('level', 'count'))
        z = b.join(a, how='inner')
        z = z.loc[z.before_days.eq(7) & z.after_days.eq(7)].copy()
        z['relative_change'] = z.after - z.before
        z['level_change'] = z.after_level - z.before_level
        z['round'] = r
        rows.append(z.reset_index())
    pairs = pd.concat(rows, ignore_index=True)
    return target, changes.reset_index(), score.reset_index(), candidate, pairs


def plot_all(daily, summary, target, changes, candidate, before_after):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, r in zip(axes, [10, 17]):
        for label, color in [('학습 비교', '#939ba0'), ('평가', COLORS[0])]:
            z = summary.loc[summary.period.eq(label) & summary['round'].eq(r)].copy()
            dates = z.date if label == '평가' else z.date + pd.DateOffset(years=1)
            ax.plot(dates, z.mean_level, color=color, alpha=.35, lw=.8)
            ax.plot(dates, z.mean_level.rolling(7, min_periods=7).mean(), color=color, lw=1.8, label=label+' 후행 7일 평균')
        ax.axvline(candidate, color=COLORS[1], ls='--', label='상대 수준 변화 후보')
        ax.set_ylabel(f'{r}시 / 기압 hPa'); ax.legend(fontsize=9)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator()); axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m', tz=ZoneInfo('Asia/Seoul')))
    axes[-1].set_xlabel('구간 시작 날짜 (학습 비교는 같은 월/일로 1년 이동해 표시)')
    save_plot(fig, OUT, GALLERY, '01_levels', 'A 지표기압의 일별 수준과 주간 흐름',
              '지점 일평균을 동일 비중으로 평균. 얇은 선은 일별, 굵은 선은 후행 7일 평균. 해가 다른 실제 날씨까지 맞춘 비교가 아님.')
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    matrices = []
    for ax, r in zip(axes, [10, 17]):
        z = target.loc[target['round'].eq(r)].pivot(index='wa_station', columns='date', values='relative_departure').sort_index()
        matrices.append(z)
        im = ax.imshow(z, aspect='auto', cmap='RdBu_r', vmin=-60, vmax=60, interpolation='nearest')
        ticks = [i for i, d in enumerate(z.columns) if d.day == 1]
        ax.set_xticks(ticks, [z.columns[i].strftime('%Y-%m') for i in ticks]); ax.set_ylabel(f'{r}시 / 고유 A 지점 61개')
        ax.set_yticks([])
        ax.axvline(z.columns.get_loc(candidate), color='#222', ls='--', lw=1)
        fig.colorbar(im, ax=ax, label='학습 상대 평균 대비 차이 (hPa)', shrink=.8, extend='both')
    save_plot(fig, OUT, GALLERY, '02_relative', '같은 날 공통 수준을 뺀 지점별 변화',
              '지점 일평균에서 그날 61지점 중앙값을 빼고, 학습 비교 구간의 해당 지점 상대 평균을 차감. 기압 오차나 고도 효과의 추정 아님. 색 범위 밖 원값은 집계에 보존.')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for r, color in zip([10, 17], COLORS):
        z = changes.loc[changes['round'].eq(r)]
        axes[0].plot(z.date, z.median_abs_change, label=f'{r}시', color=color)
        z = before_after.loc[before_after['round'].eq(r)]
        values = np.sort(z.relative_change.to_numpy())
        axes[1].plot(values, np.arange(1, len(values)+1)/len(values), label=f'{r}시', color=color)
    axes[0].axvline(candidate, color='#555', ls='--'); axes[0].set_ylabel('전일 상대 변화 절댓값의 지점 중앙값 (hPa)')
    axes[0].xaxis.set_major_locator(mdates.MonthLocator()); axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m', tz=ZoneInfo('Asia/Seoul')))
    axes[0].tick_params(axis='x', rotation=30); axes[0].legend()
    axes[1].axvline(0, color='#999', lw=1); axes[1].set_xlabel('후 7일 - 전 7일 상대 수준 평균 (hPa)'); axes[1].set_ylabel('지점 누적 비중'); axes[1].legend()
    save_plot(fig, OUT, GALLERY, '03_boundary', '상대 수준의 변화 후보와 전후 한 차례 비교',
              '후보는 두 라운드의 일별 상대 변화 절댓값 중앙값을 평균한 값이 최대인 날. 왼쪽은 원인 판정이 아니고 오른쪽은 선택된 후보의 사후 기술통계.')


def write_report(info, audit, summary, target, changes, scores, candidate, before_after):
    date = candidate.strftime('%Y-%m-%d')
    prepost = []
    for r, z in before_after.groupby('round'):
        prepost.append([r, len(z), f'{z.level_change.mean():+.3f}', f'{z.relative_change.median():+.3f}',
                        f'{z.relative_change.quantile(.1):+.3f} ~ {z.relative_change.quantile(.9):+.3f}',
                        int(z.relative_change.lt(-5).sum()), int(z.relative_change.gt(5).sum())])
    tops = scores.nlargest(5, 'mean_round_score')
    months = target.assign(month=target.date.dt.strftime('%Y-%m')).groupby(['round', 'month']).agg(
        n=('level','count'), mean_level=('level','mean'), low=('relative_departure',lambda x:x.quantile(.1)),
        high=('relative_departure',lambda x:x.quantile(.9)))
    neighborhood = changes.loc[changes.date.ge(candidate-pd.Timedelta(days=1)) & changes.date.le(candidate+pd.Timedelta(days=1))]
    text = ['# A 지표기압 변화 시점 EDA', '', f"작성: {info['finished_at']}. 작업 ID: EDA-001. 실제 계산 완료.", '',
        '## 질문과 목적', '', '평가 구간의 A 지표기압 변화가 언제 어떤 형태로 나타났는지, 여러 지점과 두 예보 라운드에서 공통적인지 확인했다. 원인을 규명하거나 기압을 보정하는 작업으로 확장하지 않았다. 사전 범위는 [EDA 진행 계획](EDA_진행계획.md)에 기록했다.', '',
        '## 사용 범위와 수행 방법', '',
        '- A 예보 61지점의 10시/17시를 분리해 평가 2025-11-01~2026-05-31과 학습 비교 2024-11-01~2025-05-31을 읽었다. 날짜는 KST 구간 시작 ts-1시간 기준이다. 전체 파일의 다른 변수와 B 기압은 사용하지 않았다.',
        '- 한 지점/라운드/날짜에 유한값 24개가 모두 있어야 일평균을 사용했다. 빠진 기록과 NaN은 채우지 않았다. 원본 유한값을 크기 기준으로 제거하지 않았다. 일평균을 61지점 사이에서 동일 비중으로 평균하고 후행 7일 평균을 보았다.',
        '- 지점 일평균에서 같은 날/라운드의 지점 중앙값을 빼 상대 수준을 만들었다. 그 값에서 학습 비교 구간의 해당 지점 상대 평균을 뺀 차이를 그림으로 보았다. 공통 날씨 변동 일부를 줄이기 위한 기술통계이며, 서로 다른 지점의 실제 날씨나 고도/정의 차이를 통제한 결과가 아니다.',
        '- 후보 날짜는 지점별 상대 수준의 전일 차이 절댓값 중앙값을 각 라운드에서 계산하고, 두 라운드 평균이 가장 큰 날 하나로 정했다. 두 라운드 모두 61개 지점의 전일 차이가 있는 날만 후보로 사용했다. 이는 사전에 기록한 설명용 기준이며 통계적 변화점 검정이나 원인 판정이 아니다.',
        '- 후보 당일을 포함한 후 7일과 직전 7일을 한 차례 비교했다. 양쪽 7일이 완전한 지점만 사용했고 지점별 평균 차이의 분포를 보았다. 이 비교는 후보를 같은 자료에서 골랐기 때문에 독립적인 검증이 아니다.',
        '- 주별 집계는 각 구간 첫날부터 7일 단위이며 마지막 짧은 주의 일수와 유효 지점-날짜 수를 함께 보존했다. 그림은 계획대로 3개다. 외부 자료 조회, 모델 적합, 자동 보정과 삭제는 수행하지 않았다.', '',
        '## 키와 일별 가용성', '', md_table(['기간','라운드','날짜 수','지점 수','예상 시간 행','실제 행','없는 행','완전 지점-날짜','불완전 지점-날짜'],
        [[r['period'],r['round'],r['days'],r['stations'],fmt(r['expected_rows']),fmt(r['actual_rows']),r['missing_rows'],fmt(r['complete_station_days']),r['incomplete_station_days']] for r in audit]), '',
        '## 주요 관찰', '',
        '일별 수준과 지점별 상대 패턴의 변화는 **2026-01-23~24**에 집중됐다. 17시 예보는 1월 23일, 10시 예보는 1월 24일에 큰 전일 상대 변화가 나타났다. 이후 지점별로 달라진 상대 수준이 평가 구간 끝인 5월 31일까지 그림에서 이어졌다. 이는 관측된 시계열 모양이며 변경 원인이나 정확한 처리 시각을 확정한 결과는 아니다.', '',
        '하루만 전체 수준이 낮았다가 회복하는 모양과는 달랐다. 학습 상대 평균 대비 차이의 지점-날짜 10~90백분위는 11~12월 대략 -1~+1 hPa 안팎이었으나, 2~5월에는 대략 -16~+12 hPa로 넓어졌다. 같은 날의 지점 중앙값을 뺀 뒤에도 차이가 이어져 전체 지점의 공통 기압 변화 하나만으로 설명되지는 않는다. 지역별 실제 날씨와 자료 정의/가공 방식은 이 자료만으로 분리하지 못한다.', '',
        '전후 7일 원 기압 수준의 지점 평균 변화는 10시 -13.579 hPa, 17시 -16.104 hPa였다. 두 라운드의 공통 지점 수가 각각 59개와 61개로 다르며 실제 날씨 변화도 포함하므로, 두 값의 차이를 라운드 효과로 해석하지 않는다. 일부 지점의 상대 수준은 하락하고 일부는 상승했으므로 모든 지점이 같은 폭으로 변한 것도 아니다.', '',
        '## 변화 후보와 전후 비교', '', f'사전 기준으로 선택된 후보는 **{date}**다. 다음 표는 두 라운드 모두 61개 전일 변화가 있는 날짜 중 상대 변화 크기 상위 5개이며 hPa 단위다. 그날의 모든 지점이 같은 방향으로 변했다는 뜻은 아니다.', '',
        md_table(['날짜','10시','17시','두 라운드 평균'], [[r['date'].strftime('%Y-%m-%d'),f'{r[10]:.3f}',f'{r[17]:.3f}',f'{r.mean_round_score:.3f}'] for _,r in tops.iterrows()]), '',
        '완전한 공통 지점 조건 때문에 1월 24일의 큰 10시 변화는 후보 순위에 들어가지 않았다. 해당 날짜는 전일과 당일이 모두 완전한 지점이 59개였기 때문이다. 따라서 후보 하나를 두 라운드의 단일 전환일로 고정하지 않고, 저장된 일별 집계와 그림을 함께 읽어 시작 범위를 1월 23~24일로 기술했다. 후보를 다시 선택하거나 전후 비교를 추가하지 않았다.', '',
        md_table(['라운드','후보 주변 날짜','전일 차이 지점 수','상대 변화 절댓값 중앙값 hPa'],
        [[r['round'],r['date'].strftime('%Y-%m-%d'),r.n_changes,f'{r.median_abs_change:.3f}'] for _,r in neighborhood.iterrows()]), '',
        md_table(['라운드','공통 지점','원 수준 평균 변화 hPa','상대 수준 변화 중앙값 hPa','상대 변화 10~90백분위','상대 5 hPa 초과 하락','상대 5 hPa 초과 상승'],prepost), '',
        '전후 상대 변화는 지점 사이 분포가 변했는지를 보여준다. 중앙값이 작아도 한쪽 꼬리의 큰 변화가 남을 수 있으므로 중앙값 하나로 전체 변화가 없다고 해석하지 않는다. 5 hPa는 규모 설명용 문턱이며 오류 판정 기준이 아니다.', '',
        '## 월별 수준과 지점 차이의 지속', '', md_table(['라운드','평가 월','완전 지점-날짜','기압 평균 hPa','학습 상대 평균 대비 차이 10~90백분위 hPa'],
        [[r,m,fmt(z.n),f'{z.mean_level:.3f}',f'{z.low:+.3f} ~ {z.high:+.3f}'] for (r,m),z in months.iterrows()]), '',
        '월평균 수준은 실제 기상 변동과 지점 분포 변화를 함께 포함한다. 상대 평균 대비 차이의 분위수는 지점-날짜 분포이며 독립 표본 신뢰구간이 아니다.', '',
        '## 실행과 검증', '', f"정규 CPython 3.13으로 실제 계산을 실행했다. 원본 3개 파일의 해시가 실행 전후 같았다. 검증 기록과 그림 3개는 로컬 결과 폴더에 보관한다. 완료 시각: {info['finished_at']}.", '',
        '## 한계와 종료 판단', '', '원인 확정, 예보 오차나 모델 성능 평가는 하지 않았다. 날짜 후보는 상대 공간 패턴의 급한 변화 기준이므로 기압 수준이 처음 낮아진 날짜와 다를 수 있다. 전체 수준의 최초 변화는 일별 그림과 함께 해석한다. 학습 비교는 다른 해의 실제 날씨를 맞춘 대조군이 아니다.', '',
        '시작 구간, 지속과 지점/라운드 차이를 기록한 뒤 EDA-001을 종료한다. 추가로 독립된 질문이 생기면 먼저 진행 계획에 등록하며 후보를 더 찾거나 다른 원인 검증을 연속 실행하지 않는다.', '',
        '## 관련 코드와 로컬 결과', '', '- [분석 코드](../scripts/pressure_timing_eda.py)',
        '- [그림 3개](../data/eda/2026-10-19_pressure_timing/index.html)',
        '- [일별 전체 지점 집계](../data/eda/2026-10-19_pressure_timing/daily_summary.csv)',
        '- [주별 집계](../data/eda/2026-10-19_pressure_timing/weekly.csv)', '',
        '원본 또는 전처리 데이터 행을 보고서에 복사하지 않았다. 데이터가 담긴 모든 출력과 그림은 로컬 data/에만 보관한다.', '']
    REPORT.write_text('\n'.join(text),encoding='utf-8')


def verify(info):
    audit = pd.read_csv(OUT/'audit.csv', encoding='utf-8-sig')
    daily = pd.read_csv(OUT/'daily_station.csv', encoding='utf-8-sig')
    summary = pd.read_csv(OUT/'daily_summary.csv', encoding='utf-8-sig')
    weekly = pd.read_csv(OUT/'weekly.csv', encoding='utf-8-sig')
    scores = pd.read_csv(OUT/'candidate_scores.csv', encoding='utf-8-sig')
    before_after = pd.read_csv(OUT/'before_after.csv', encoding='utf-8-sig')
    assert len(audit)==4 and audit.days.eq(212).all() and audit.stations.eq(61).all()
    assert (audit.expected_rows-audit.actual_rows).eq(audit.missing_rows).all()
    assert (audit.complete_station_days+audit.incomplete_station_days).eq(audit.days*audit.stations).all()
    assert len(daily)==4*61*212 and len(summary)==4*212
    assert not daily.duplicated(['period','round','wa_station','date']).any()
    assert (daily.complete.eq(True)==(daily.rows.eq(24)&daily.finite.eq(24))).all()
    for (p,r),z in daily.groupby(['period','round']):
        a=audit.loc[audit.period.eq(p)&audit['round'].eq(r)].iloc[0]
        assert z.rows.sum()==a.actual_rows and z.complete.sum()==a.complete_station_days
        s=summary.loc[summary.period.eq(p)&summary['round'].eq(r)].set_index('date')
        assert np.allclose(z.groupby('date')['level'].mean().reindex(s.index),s.mean_level)
        w=weekly.loc[weekly.period.eq(p)&weekly['round'].eq(r)]
        assert w.n.sum()==z.level.notna().sum()
        assert np.isclose(np.average(w.mean_level,weights=w.n),z.level.mean())
    assert before_after.before_days.eq(7).all() and before_after.after_days.eq(7).all()
    assert np.allclose(before_after.after-before_after.before,before_after.relative_change)
    assert scores.loc[scores.mean_round_score.idxmax(),'date'][:10]==info['candidate']
    assert info['sha256']=={f:digest(SOURCE/f) for f in FILES}
    assert '\u00b7' not in REPORT.read_text(encoding='utf-8')
    result={**check_gallery(OUT,3),'station_days':len(daily),'aggregate_days':len(summary),'weekly_groups':len(weekly),
            'before_after_station_rounds':len(before_after),'source_unchanged':True,'daily_weekly_accounting':True}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    assert sys.version_info[:2]==(3,13)
    OUT.mkdir(parents=True,exist_ok=True)
    if '--verify-only' in sys.argv:
        verify(json.loads((OUT/'summary.json').read_text(encoding='utf-8')));return
    if '--report-only' in sys.argv:
        info=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
        tables={name:pd.read_csv(OUT/(name+'.csv'),encoding='utf-8-sig')
                for name in ['audit','daily_summary','target_relative','daily_change','candidate_scores','before_after']}
        for name in ['daily_summary','target_relative','daily_change','candidate_scores']:
            tables[name]['date']=pd.to_datetime(tables[name].date,utc=True).dt.tz_convert('Asia/Seoul')
        scores=tables['candidate_scores'].rename(columns={'10':10,'17':17})
        write_report(info,tables['audit'].to_dict('records'),tables['daily_summary'],tables['target_relative'],
                     tables['daily_change'],scores,pd.Timestamp(info['candidate'],tz='Asia/Seoul'),tables['before_after'])
        return
    before={f:digest(SOURCE/f) for f in FILES}
    stations=sorted(pd.read_csv(SOURCE/'sites.csv',encoding='utf-8-sig').wa_station.unique())
    assert len(stations)==61
    parts=[];audit=[]
    for split,label,lo,hi in CONFIG:
        d,a=load_period(split,label,lo,hi,stations);parts.append(d);audit.extend(a)
    daily=pd.concat(parts,ignore_index=True)
    summary=aggregate(daily)
    target,changes,scores,candidate,before_after=describe_change(daily)
    weeks=daily.copy()
    weeks['week']=weeks.groupby('period',observed=True).date.transform(lambda x:(x-x.min()).dt.days//7)
    weekly=weeks.groupby(['period','round','week'],observed=True).agg(
        first_date=('date','min'),last_date=('date','max'),days=('date','nunique'),n=('level','count'),mean_level=('level','mean')).reset_index()
    plot_all(daily,summary,target,changes,candidate,before_after)
    assert before=={f:digest(SOURCE/f) for f in FILES}
    info={'sha256':before,'candidate':candidate.strftime('%Y-%m-%d'),
          'finished_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='minutes')}
    for name,table in [('audit',pd.DataFrame(audit)),('daily_station',daily),('daily_summary',summary),
                       ('weekly',weekly),('target_relative',target),('daily_change',changes),('candidate_scores',scores),('before_after',before_after)]:
        table.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    (OUT/'summary.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    write_gallery(OUT,GALLERY,'A 지표기압 변화 시점 EDA')
    write_report(info,audit,summary,target,changes,scores,candidate,before_after)
    verify(info)
    log(str(REPORT))


if __name__=='__main__':
    main()
