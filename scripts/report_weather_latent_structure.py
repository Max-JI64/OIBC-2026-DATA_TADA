"""Write the DA-001 report from saved, verified calculations without refitting."""
import json
import sys

import numpy as np

from weather_latent_structure import OUT, REPORT

NAMES = {'wa_observed': 'A 실측', 'wa_10': 'A 예보 10시', 'wa_17': 'A 예보 17시',
         'wb_observed': 'B 실측', 'wb_10': 'B 예보 10시', 'wb_17': 'B 예보 17시'}
VIEWS = {'all': '전체 수준', 'within_day': '낮 편차'}


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def number(x, digits=3):
    return '미산출' if x is None else f'{x:.{digits}f}'


def main():
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    verification = json.loads((OUT / 'verification.json').read_text(encoding='utf-8'))
    header = (OUT / 'preregistered_report.md').read_text(encoding='utf-8').split('## 결과')[0]
    header = header.replace('상태: DA-001 진행. 아래 범위와 기준을 상관/요인 계산 전에 고정했다.',
                            '상태: DA-001 완료. 아래 계약은 상관/요인 계산 전에 고정했으며 변경 없이 실행했다.')
    body = ['## 결과', '',
        'B 예보의 낮 편차에서 운량/습도와 UV가 반대 방향으로 움직이는 첫 공통축은 연도를 바꿔도 반복됐다. '
        '그러나 두 번째 요인은 불안정했고 일부 재표집에서는 적합 자체가 실패했다. A의 7변수를 정한 3요인으로 설명하는 해도 부적절하거나 수렴하지 않았다. '
        '따라서 일부 관계의 공통축은 지지되지만 전체 기상 자료를 안정적인 소수 요인으로 대체할 근거는 확보하지 못했다.', '',
        '### 실제 사용 표본', '']
    rows = []
    for label, audits in summary['audit'].items():
        for a in audits:
            rows.append([NAMES[label], a['year'], a['stations'], a['balanced_dates'], f"{a['balanced_rows']:,}",
                         a['excluded_dates'], a['duplicate_key_rows'], sum(a['invalid_by_column'].values())])
    body += [table(['자료', '연도', '지점', '잔존 날짜', '전체 시간 행', '제외 날짜', '중복 키 행', '비유효 셀'], rows), '',
        '같은 자료의 낮 편차는 위 표 전체 시간 행의 1/3인 8시간을 사용했다. 비유효 셀은 임시 중복 충돌 결측을 포함하며 '
        '중복 행 수와 더해 별도의 제거 행 수로 해석하지 않는다. 한 지점의 주의값도 해당 날짜 전체를 제외하는 엄격한 균형 규칙을 사용했다. '
        '특히 A 10시 예보의 2024년 12월은 5일이 제외돼 주의값이 많은 날짜를 대표하지 못할 수 있다.', '',
        '### 적합성, 차원과 PCA 비교', '']
    rows = []
    for label, group in summary['primary'].items():
        for view, c in group.items():
            d = c['diagnostics']
            rows.append([NAMES[label], VIEWS[view], number(d['kmo']), c['selected'],
                         '적정 해' if c['factor']['valid'] else '부적절/수렴 미확인',
                         f"{100*d['pca_2']:.1f}%", f"{100*d['pca_3']:.1f}%"])
    body += [table(['자료', '뷰', 'KMO', '선택 요인 수', 'PA 결과', 'PCA 2개 누적', 'PCA 3개 누적'], rows), '',
        '12개 주 뷰의 상관행렬은 모두 정한 수치 안정성 기준을 통과했다. 그러나 PA 해의 적정성과는 다른 조건이다. '
        '주 뷰 12개 중 적정 해는 3개, 부적절하거나 수렴 기준을 통과하지 못한 해는 9개였다. '
        '여기서 적정 해는 공통성 범위와 수렴 검사를 통과했다는 뜻이며 해석과 기간별 안정성이 검증됐다는 뜻은 아니다. '
        'PCA는 전체 분산 요약이며 FA의 공통분산이나 발전량 설명력으로 읽지 않는다.', '',
        '### A에서는 이번 3요인 해를 해석할 수 없다', '',
        'A의 KMO는 전체 수준에서 0.512~0.521, 낮 편차에서 0.546~0.562로 사전 주의 기준 0.6보다 낮았다. '
        '귀무 비교에서는 세 차원이 선택됐지만 여섯 해 모두 공통성이 1을 넘는 변수가 나타났고 수렴 검사도 통과하지 못했다. '
        '이는 독자성이 음수가 되는 부적절해다. 잔차 상관이 작더라도 이를 유효한 잠재 요인으로 채택할 수 없다.', '',
        '이번 변수 구성과 정한 요인 수에서 안정적인 해를 얻지 못했다는 결론이다. A의 모든 변수 사이에 공통 정보가 없다거나 '
        '더 적은 요인이나 다른 추정 방법도 실패한다는 결론은 아니다. A를 임의로 몇 개 요인으로 치환하는 대신 '
        '공통 변동과 고유 변동을 구분할 추가 검토 후보를 DA-006에 남긴다.', '',
        '### B 예보는 첫 공통축과 불안정한 두 번째 축을 구분해야 한다', '',
        '아래 적재량은 적정 해를 얻은 세 뷰다. 요인 부호와 순서는 임의적이며 같은 번호가 서로 다른 자료의 동일한 물리 요인을 뜻하지 않는다. '
        'B 실측은 전체 수준, 두 예보는 낮 편차이므로 수치를 같은 조건의 공급사 우열 비교로 사용하지 않는다.', '']
    selections = [('wb_observed', 'all'), ('wb_10', 'within_day'), ('wb_17', 'within_day')]
    rows = []
    for i, name in enumerate(['기온', '상대습도', '운량', '풍속', 'UV', '강수(log1p)', '시정', '운고(log1p)']):
        rows.append([name] + [number(summary['primary'][lab][v]['factor']['loadings'][i][j])
                              for lab, v in selections for j in range(2)])
    body += [table(['변수', '실측 전체 F1', '실측 전체 F2', '10시 낮 F1', '10시 낮 F2', '17시 낮 F1', '17시 낮 F2'], rows), '',
        'B 17시 예보의 낮 편차 첫 축은 UV +0.808, 운량 -0.769, 습도 -0.520이 함께 나타났다. '
        '운고 +0.767과 시정 +0.434도 같은 축에 걸렸다. 달력 평균을 제거한 뒤에도 맑음/흐림에 동반하는 공통 변동이 남은 셈이다. '
        '이를 실제 일사량이나 구름의 인과 효과를 측정한 요인으로 확정하지는 않는다.', '',
        '이 첫 축의 2023년 Tucker 일치도는 10시 0.982, 17시 0.984였다. 반면 두 번째 축은 각각 0.432와 0.379였다. '
        '두 번째 축에서 절대 적재량 0.4 이상인 변수도 기온 하나뿐이었다. 기온에 가까운 이 축을 다른 변수까지 함께 설명하는 '
        '안정적인 공통요인으로 해석할 근거가 약하다.', '',
        '17시 예보 낮 편차에서 강우량의 공통성은 0.156, 풍속은 0.106이었다. 정한 두 요인이 이 변수들의 표준화된 변동 중 '
        '약 15.6%와 10.6%만 재현했다는 뜻이다. 운량/UV 축이 뚜렷하다는 이유로 강수나 바람을 같은 축으로 대체하면 '
        '관측된 별도 변동을 많이 잃을 수 있다. 이것이 그 변동의 발전량 예측 유용성을 입증한 것은 아니다.', '',
        '### 재현과 블록 안정성', '']
    rows = []
    for lab, v in selections:
        c = summary['primary'][lab][v]
        repeat = c['replication_2023'].get('congruence', [])
        b = c['bootstrap']
        rows.append([NAMES[lab] + '/' + VIEWS[v], number(c['factor']['rms_residual']),
                     '/'.join(number(x) for x in repeat), f"{b['success']}/{b['attempts']}", number(b['min_congruence_q05'])])
    body += [table(['자료/뷰', '잔차 RMS', '2023 F1/F2 일치도', '블록 적정 해', '블록 최저 일치도 5백분위'], rows), '',
        '블록 부트스트랩은 세 뷰에서 각 100회, 총 300회 시도했다. 실패한 적합은 구간 계산에서 제외하고 실패율을 별도로 보고했다. '
        '따라서 성공 표본에서의 높은 일치도만 보고 전체 추정이 안정적이라고 결론 내릴 수 없다. '
        '두 예보의 첫 축은 성공 재표집에서 높은 일치도를 보였지만 두 번째 축의 불안정성과 적합 실패를 함께 남긴다.', '',
        'B 실측 전체 수준의 두 요인은 2023년 재현과 잔차/블록 일치도 수치 기준을 통과했다. '
        '다만 블록 성공은 81%로 정한 90% 경고 기준에 미달했고 두 번째 축의 4분기 일치도는 0.793이었다. '
        '연간 전체 수준의 비교적 재현되는 요약 후보이며 모든 계절에 안정적인 구조로 확정하지 않는다.', '',
        'B 실측 낮 편차는 500회 반복 후 다음 공통성 갱신의 차이가 약 2.11e-5로 허용값 1e-6을 초과했다. '
        '이 해를 적정 해로 보고하거나 반복 상한을 사후 늘려 성공 결과로 바꾸지 않았다.', '',
        '### 시정/운고를 뺀 같은 표본의 민감도', '']
    rows = []
    for lab, group in summary['sensitivity'].items():
        for v, c in group.items():
            b = summary['primary'][lab][v]
            cong = c['replication_2023'].get('congruence', [])
            rows.append([NAMES[lab], VIEWS[v], number(b['diagnostics']['kmo']) + ' → ' + number(c['diagnostics']['kmo']),
                         f"{c['selected_raw']} → {c['selected']}", '적정 해' if c['factor']['valid'] else '수렴 미확인',
                         '/'.join(number(x) for x in cong) or '미산출'])
    body += [table(['자료', '뷰', 'KMO 8변수 → 6변수', '귀무 초과 차원 → 적합 수', '6변수 PA', '2023 F1/F2 일치도'], rows), '',
        'B 예보 낮 편차의 KMO는 시정/운고를 제외해도 10시 0.627, 17시 0.624로 남았다. '
        '두 요인의 연도 간 일치도도 10시 0.994/0.896, 17시 0.994/0.886이었다. '
        '이는 공통 구조가 시정/운고 두 변수에만 의존하지 않을 가능성을 지지한다. '
        '하지만 두 번째 요인에 강하게 걸리는 변수가 하나뿐인 경고는 남았고, 6변수 민감도에는 추가 블록 재표집을 하지 않았다. '
        '연도 일치도 개선을 곧바로 더 좋은 요인모형이나 예측 입력이라는 뜻으로 해석하지 않는다.', '',
        '모든 민감도는 같은 날짜와 행에서 두 열만 제외했다. 특정 집중값 행의 포함/제외 효과를 분리한 분석은 아니며 '
        '그 값이 결측 코드라는 결론이나 해당 행의 삭제 권고를 내리지 않는다.', '',
        '## 한계 및 다음 단계', '',
        '주축요인법과 직교 varimax, 정한 변수군에 한정된 결과다. 선형 공통요인으로 설명하기 어려운 비선형 관계, '
        '강수/강설의 0 집중과 희소성, 상관된 요인의 가능성은 남는다. 부적절해나 미수렴을 데이터 오류 또는 잠재 구조 부재로 단정하지 않는다. '
        '이번에는 사후 방법 교체와 요인 수 탐색을 이어가지 않았다.', '',
        '월 안의 날짜 순환이동은 공간과 일중 구조를 보존하지만 월 경계 및 이동 경계의 시간 연속성은 바뀐다. '
        '또한 유한 199회 기준에 가까운 고유값의 차원 수는 불안정할 수 있다. 7일 재표집도 관측 가능한 균형 날짜 순서를 기준으로 해 '
        '제외 날짜를 지나는 블록에서는 정확한 달력 7일이 아니다. 고정 지점 모집단과 이미 계산한 달력 평균에 조건부인 기술적 결과이며 '
        '모든 지점과 미래 연도에 대한 확증적 검정으로 일반화하지 않는다.', '',
        'DA-001은 정한 범위에서 종료한다. 전체 기상 정보를 소수 요인으로 대체하는 판단은 보류한다. '
        '다음 우선 후보는 DA-003이다. 반복된 맑음/흐림 공통 변동과 그 축으로 설명되지 않은 강수/바람을 함께 고려했을 때 '
        '발전량의 관계가 어떻게 다른지 설명 목적의 다변량 분석으로 다룰 수 있다. 실제 표본, 방법과 종료 기준은 시작 전에 별도로 고정한다. '
        'A의 더 적은 요인/변수군 검토는 DA-006 선택 후보로 남긴다.', '',
        '모델링 활용은 가설 단계다. 공통 축의 조건부 표현과 개별 변수의 남는 정보를 구분해 볼 수 있지만, '
        '새 발전량 모델 학습/튜닝/추론, 예측 파일 생성, V2/평가 구간 사용과 공식 제출은 모두 0회다. '
        '요인 추정과 예측 모델 학습을 혼동하지 않는다.', '',
        '## 실행과 검증', '',
        '정규 CPython 3.13.1의 승격 Start-Process 경로로 환경 점검, 합성 검증, 본 분석과 저장 결과 검증을 실행했고 '
        '모두 종료 코드 0으로 완료했다. 합성 검증 8개는 FFT와 직접 순환이동의 동치, 충분통계와 직접 상관의 동치, '
        '알려진 요인의 회복, 순서/부호 정렬, 잘못된 행렬 거부, 달력 편차와 귀무 행렬 대각, 블록 구성과 충돌 중복 날짜 제외를 확인했다.', '',
        f"저장 결과 검증에서는 주 뷰 {verification['primary_views']}개와 민감도 {verification['sensitivity_views']}개, "
        f"2024/2023 요인 재구성 {verification['factor_reconstructions']}건의 표본 수, KMO, 고유값, PCA, 공통성/독자성, "
        '잔차와 재현 일치도를 별도 수식으로 확인했다. 원본 6개 파일의 SHA-256은 실행 전후와 재검증 때 동일했고 '
        '분석 코드 해시는 실행 전후 동일했다. 부트스트랩 합계와 민감도의 같은 행 조건도 점검했다.', '',
        '그림 8개와 갤러리 링크 16개를 검사하고 모든 그림을 50% 크기의 4개 묶음 두 장에서 직접 확인했다. '
        '원본 크기 전체 픽셀을 검사했다는 뜻은 아니다. 적합 실패를 표시한 제목과 적재량, 안정성의 빈 막대가 구분됨을 확인했다.', '',
        '## 관련 코드와 로컬 결과', '',
        '- [분석 코드](../scripts/weather_latent_structure.py)',
        '- [합성 검증](../tests/test_weather_latent_structure.py)',
        '- [저장 결과 검증](../scripts/verify_weather_latent_structure.py)',
        '- [보고서 생성](../scripts/report_weather_latent_structure.py)',
        '- [계산 전 고정 보고서](../data/analysis/2026-10-21_weather_latent/preregistered_report.md)',
        '- [전체 수치와 실패 기록](../data/analysis/2026-10-21_weather_latent/summary.json)',
        '- [재검증 결과](../data/analysis/2026-10-21_weather_latent/verification.json)',
        '- [그림 갤러리](../data/analysis/2026-10-21_weather_latent/index.html)',
        '- [진행 계획](데이터분석_진행계획.md)', '',
        '방법 참고: [statsmodels Factor](https://www.statsmodels.org/stable/generated/statsmodels.multivariate.factor.Factor.html), '
        '[FactorResults](https://www.statsmodels.org/stable/generated/statsmodels.multivariate.factor.FactorResults.html). '
        '실제 설치 버전은 0.14.6이며 공식 웹 문서와 설치 소스의 PA 반복/회전 구현을 함께 확인했다.', '',
        '원본 및 전처리 행은 보고서에 복사하지 않았다. 충분통계, 요인 계수와 블록 구간 등 데이터 출력은 로컬 data/에만 보관하고 '
        '코드와 해석 보고서, 로그만 로컬 커밋한다.', '']
    text = header + '\n'.join(body)
    assert '\u00b7' not in text
    REPORT.write_text(text, encoding='utf-8')
    print(str(REPORT))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
