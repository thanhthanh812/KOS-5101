#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_descriptives.py
======================
survey_data.csv 를 읽어 논문(이선빈·김지현, 2025)의 기술통계와 비교한다.
  <표 1> 인적 사항 빈도,  <표 3> Cronbach's α,  <표 4> 쓰기 경험/요구 빈도,
  <표 5> 하위척도 M/SD,   <표 6> 상위·하위 1순위 문항 M/SD,
  <표 7> 상관,            <표 8>·<표 9> TOPIK 급수별 M/SD

사용법:  python verify_descriptives.py [survey_data.csv]
"""
import sys
import numpy as np
import pandas as pd
from generate_survey_data import (SUBSCALES, SUB_ORDER, ITEM_TARGETS, CORR_TARGET, GROUP_MEANS, GROUP_SDS,
                                  GENDER_COUNTS, NATION_COUNTS, TOPIK_COUNTS, PERIOD_COUNTS,
                                  EXP_COUNTS, DIFF_COUNTS, NEED_COUNTS, cronbach_alpha)

path = sys.argv[1] if len(sys.argv) > 1 else "survey_data.csv"
df = pd.read_csv(path)
likert = [c for c in df.columns if c[:2] in ("SE", "LW", "MC")]
assert df[likert].isin([1, 2, 3, 4, 5]).all().all(), "리커트 문항은 1~5 정수여야 함"
assert (df.dtypes == "int64").all(), "모든 열이 정수형이어야 함"
print(f"파일: {path}   N = {len(df)}   열 = {df.shape[1]}   리커트 문항 = {len(likert)}개 (모두 1~5 정수)\n")

def show(title, rows, cols):
    print(f"■ {title}")
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))
    print()

# <표 1> 빈도 --------------------------------------------------------------
rows = []
for var, counts in [("gender", GENDER_COUNTS), ("nationality", NATION_COUNTS),
                    ("topik", TOPIK_COUNTS), ("study_period", PERIOD_COUNTS), ("need", NEED_COUNTS)]:
    vc = df[var].value_counts()
    for k, n in counts.items():
        rows.append([var, k, n, int(vc.get(k, 0)), "OK" if vc.get(k, 0) == n else "X"])
for counts in (EXP_COUNTS, DIFF_COUNTS):
    for c, n in counts.items():
        rows.append([c, 1, n, int(df[c].sum()), "OK" if df[c].sum() == n else "X"])
show("<표 1>·<표 4> 빈도 비교", rows, ["변수", "값", "논문", "시뮬", "일치"])

# 하위척도 점수 ----------------------------------------------------------------
S = pd.DataFrame({s: df[SUBSCALES[s]["items"]].mean(axis=1) for s in SUB_ORDER})

# <표 3>, <표 5> ----------------------------------------------------------------
rows = []
for s in SUB_ORDER:
    t = SUBSCALES[s]
    a = cronbach_alpha(df[t["items"]].values)
    rows.append([s, t["M"], round(S[s].mean(), 2), t["SD"], round(S[s].std(ddof=1), 2),
                 t["alpha"], round(a, 3)])
# '전체' 행 = 논문에서는 하위척도 M, SD 의 단순 평균으로 계산됨
for name, subs in [("효능감 전체", SUB_ORDER[:3]), ("호감도 전체", SUB_ORDER[3:5]), ("상위인지 전체", SUB_ORDER[5:])]:
    tm = np.mean([SUBSCALES[s]["M"] for s in subs]); tsd = np.mean([SUBSCALES[s]["SD"] for s in subs])
    rows.append([name, round(tm, 2), round(S[subs].mean().mean(), 2), round(tsd, 2),
                 round(S[subs].std(ddof=1).mean(), 2), "", ""])
show("<표 5> 하위척도 M/SD 및 <표 3> Cronbach's α", rows,
     ["척도", "M(논문)", "M(시뮬)", "SD(논문)", "SD(시뮬)", "α(논문)", "α(시뮬)"])

# <표 6> ---------------------------------------------------------------------
rows = [[it, m, round(df[it].mean(), 2), sd, round(df[it].std(ddof=1), 3)] for it, (m, sd) in ITEM_TARGETS.items()]
show("<표 6> 상위·하위 1순위 문항", rows, ["문항", "M(논문)", "M(시뮬)", "SD(논문)", "SD(시뮬)"])
# 실제로 최고/최저인지 확인
im = df[likert].mean()
for grp, hi, lo in [("SE", "SE15", "SE04"), ("LW", "LW19", "LW18"), ("MC", "MC26", "MC08")]:
    g = im[[c for c in likert if c.startswith(grp)]]
    print(f"   {grp}: 최고 문항 = {g.idxmax()} ({g.max():.2f}), 최저 문항 = {g.idxmin()} ({g.min():.2f})"
          f"   → 논문: 최고 {hi}, 최저 {lo}  {'OK' if (g.idxmax()==hi and g.idxmin()==lo) else 'X'}")
print()

# <표 7> ---------------------------------------------------------------------
C = S.corr().values
print("■ <표 7> 상관 (시뮬 / 논문)")
lab = ["발상", "규칙", "자기조절", "긍정", "부정", "계획", "작성", "검토"]
for i in range(8):
    print(f"  {lab[i]:<5}", "  ".join(f"{C[i,j]:+.3f}/{CORR_TARGET[i,j]:+.3f}" for j in range(i)))
iu = np.triu_indices(8, 1)
print(f"  최대 절대 오차 = {np.abs(C[iu]-CORR_TARGET[iu]).max():.3f},  평균 절대 오차 = {np.abs(C[iu]-CORR_TARGET[iu]).mean():.3f}\n")

# <표 8>, <표 9> ---------------------------------------------------------------
from scipy import stats as sps
PAPER_F = {"ideation": (2.875, .060), "conventions": (1.741, .180), "selfreg": (0.268, .765), "like_pos": (1.827, .165),
           "like_neg": (14.918, .000), "planning": (1.524, .210), "drafting": (1.578, .197), "revising": (3.506, .017)}
rows = []
for s in SUB_ORDER:
    groups = [S.loc[df.topik == g, s].values for g in (1, 2, 3)]
    F, p = sps.f_oneway(*groups)
    for g, name in [(1, "초급"), (2, "중급"), (3, "고급")]:
        x = S.loc[df.topik == g, s]
        rows.append([s, name, len(x), GROUP_MEANS[s][g - 1], round(x.mean(), 2), GROUP_SDS[s][g - 1], round(x.std(ddof=1), 2),
                     PAPER_F[s][0] if g == 1 else "", round(F, 3) if g == 1 else "", PAPER_F[s][1] if g == 1 else "", round(p, 3) if g == 1 else ""])
    x = S.loc[df.topik == 0, s]
    rows.append([s, "없음", len(x), "-", round(x.mean(), 2), "-", round(x.std(ddof=1), 2), "", "", "", ""])
show("<표 8>·<표 9> TOPIK 급수별 M·SD 와 일원분산분석 F, p (긍정호감도·검토전략은 --priority 설정에 따라 다름, 생성 스크립트 주석 참조)",
     rows, ["척도", "급수", "N", "M(논문)", "M(시뮬)", "SD(논문)", "SD(시뮬)", "F(논문)", "F(시뮬)", "p(논문)", "p(시뮬)"])
