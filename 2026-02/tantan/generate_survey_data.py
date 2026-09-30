#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_survey_data.py
=======================
이선빈·김지현(2025), 「학부 유학생의 한국어 쓰기 효능감·호감도·상위인지 전략 사용 양상 연구」,
작문연구 65, 7-41 에 보고된 기술통계와 (거의) 일치하는 **가상 설문조사 데이터**를 생성한다.

설문 구성 (논문 <표 2>, <표 3>)
  - 기본 인적 사항 : 성별, 국적, TOPIK 급수, 한국어 학습 기간            (선택형)
  - 쓰기 경험/요구  : 대학에서 수행한 쓰기, 어려움을 느끼는 쓰기 과제, 필요한 교육 요소 (선택형, 복수응답)
  - 쓰기 효능감     : 16문항  SE01~SE16  (발상 1~5, 규칙 6~10, 자기조절 11~16)       5점 리커트
  - 쓰기 호감도     :  4문항  LW17~LW20  (긍정 17,19 / 부정 18,20)                 5점 리커트
  - 쓰기 상위인지 전략 : 31문항 MC01~MC31 (계획 1~10, 작성 중 11~20, 검토 21~31)      5점 리커트
  리커트 문항은 '전혀 아니다'=1 … '매우 그렇다'=5 의 정수값이다.

생성 원리
  1) 응답자별 8개 잠재요인(발상, 규칙, 자기조절, 긍정호감, 부정호감, 계획, 작성, 검토)을
     논문 <표 7>의 상관구조를 갖는 다변량 정규분포에서 뽑고, TOPIK 급수별 평균차(<표 8>, <표 9>)를 더한다.
  2) 문항 점수 = 문항 평균 + 부하량 × 잠재요인 + 문항 고유 오차  →  반올림 후 1~5로 절단.
  3) 반올림/절단 때문에 생기는 왜곡은 고정된 난수(seed)를 두고 파라미터(문항 평균, 요인 분산,
     오차 분산, 잠재 상관, 집단 오프셋)를 반복 보정하여 논문 수치에 맞춘다.

사용법
  python generate_survey_data.py                 # survey_data.csv 생성
  python generate_survey_data.py --out foo.csv --seed 7
  python generate_survey_data.py --priority table89 --out survey_data_table89.csv   # 표 8·9 우선 (아래 GROUP_SHRINK 설명 참조)
"""
import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. 논문에 보고된 목표치
# ---------------------------------------------------------------------------
N = 172

# <표 1> 연구 대상자 기본 정보
GENDER_COUNTS = {1: 52, 2: 120}                       # 1=남, 2=여
NATION_COUNTS = {1: 108, 2: 8, 3: 25, 4: 14, 5: 6, 6: 11}   # 베트남, 중국, 우즈베키스탄, 일본, 러시아, 기타
TOPIK_COUNTS = {0: 52, 1: 33, 2: 51, 3: 36}             # 없음, 초급(1~2급), 중급(3~4급), 고급(5~6급)
PERIOD_COUNTS = {1: 7, 2: 15, 3: 48, 4: 102}            # <6개월, 6개월~1년, 1년~1년6개월, ≥1년6개월

# <표 4> 쓰기 경험과 요구도 (복수 응답은 0/1 열, 요구 요소는 단일 선택)
EXP_COUNTS = {"exp_presentation": 54, "exp_report": 41, "exp_thesis": 12, "exp_exam": 71, "exp_other": 29}
DIFF_COUNTS = {"diff_presentation": 51, "diff_report": 34, "diff_thesis": 62, "diff_exam": 22, "diff_other": 14}
NEED_COUNTS = {1: 63, 2: 30, 3: 33, 4: 34, 5: 12}       # 전공내용/어휘, 학술글쓰기수업, 피드백수업, 즉각수정수업, 자료검색교육

# <표 3>, <표 5> 하위 척도: 문항, 평균, 표준편차(개인별 하위척도 평균점수의 SD), Cronbach's α
SUBSCALES = {
    "ideation":    dict(items=[f"SE{i:02d}" for i in range(1, 6)],   M=3.29, SD=0.73, alpha=0.901),
    "conventions": dict(items=[f"SE{i:02d}" for i in range(6, 11)],  M=3.33, SD=0.71, alpha=0.908),
    "selfreg":     dict(items=[f"SE{i:02d}" for i in range(11, 17)], M=3.35, SD=0.73, alpha=0.901),
    "like_pos":    dict(items=["LW17", "LW19"],                      M=3.27, SD=0.87, alpha=0.873),
    "like_neg":    dict(items=["LW18", "LW20"],                      M=3.17, SD=1.03, alpha=0.852),
    "planning":    dict(items=[f"MC{i:02d}" for i in range(1, 11)],  M=3.51, SD=0.73, alpha=0.847),
    "drafting":    dict(items=[f"MC{i:02d}" for i in range(11, 21)], M=3.51, SD=0.70, alpha=0.948),
    "revising":    dict(items=[f"MC{i:02d}" for i in range(21, 32)], M=3.57, SD=0.74, alpha=0.880),
}
SUB_ORDER = list(SUBSCALES)

# <표 6> 항목별 상위/하위 1순위 문항 (M, SD)
ITEM_TARGETS = {
    "SE15": (3.51, 0.880), "SE04": (3.20, 0.844),   # 효능감 상위1 / 하위1
    "LW19": (3.32, 0.909), "LW18": (3.15, 1.129),   # 호감도 상위1 / 하위1
    "MC26": (3.76, 0.400), "MC08": (3.31, 0.930),   # 상위인지 상위1 / 하위1
}
# 척도군별 문항 평균의 허용 범위 (표 6의 상위1·하위1 문항이 실제로 최고/최저가 되도록)
GROUP_RANGE = {"SE": (3.20, 3.51), "LW": (3.15, 3.32), "MC": (3.31, 3.76)}

# <표 7> 하위 척도 평균점수 간 Pearson 상관
CORR_TARGET = np.array([
    [1.000, 0.858, 0.797, 0.570, -0.566, 0.675, 0.654, 0.637],
    [0.858, 1.000, 0.789, 0.629, -0.533, 0.750, 0.702, 0.657],
    [0.797, 0.789, 1.000, 0.717, -0.437, 0.799, 0.765, 0.713],
    [0.570, 0.629, 0.717, 1.000, -0.258, 0.660, 0.625, 0.559],
    [-0.566, -0.533, -0.437, -0.258, 1.000, -0.344, -0.302, -0.285],
    [0.675, 0.750, 0.799, 0.660, -0.344, 1.000, 0.856, 0.755],
    [0.654, 0.702, 0.765, 0.625, -0.302, 0.856, 1.000, 0.832],
    [0.637, 0.657, 0.713, 0.559, -0.285, 0.755, 0.832, 1.000],
])

# <표 8>, <표 9> TOPIK 급수(초급/중급/고급)별 하위척도 평균. '없음' 집단은 전체 평균이 맞도록 유도된다.
GROUP_MEANS = {  # (초급 n=33, 중급 n=51, 고급 n=36)
    "ideation":    (3.58, 3.24, 3.26),
    "conventions": (3.56, 3.27, 3.41),
    "selfreg":     (3.44, 3.33, 3.41),
    "like_pos":    (3.41, 3.89, 3.39),
    "like_neg":    (2.59, 3.09, 3.79),
    "planning":    (3.61, 3.49, 3.55),
    "drafting":    (3.55, 3.46, 3.71),
    "revising":    (3.89, 3.85, 4.31),
}
# <표 8>, <표 9> TOPIK 급수(초급/중급/고급)별 하위척도 표준편차
GROUP_SDS = {
    "ideation":    (0.86, 0.63, 0.58),
    "conventions": (0.84, 0.60, 0.57),
    "selfreg":     (0.92, 0.69, 0.60),
    "like_pos":    (1.04, 0.72, 0.93),
    "like_neg":    (1.03, 0.90, 0.85),
    "planning":    (0.88, 0.62, 0.69),
    "drafting":    (0.85, 0.60, 0.63),
    "revising":    (0.99, 0.67, 0.80),
}
# 논문 <표 8>·<표 9>의 '긍정 호감도'와 '검토 전략' 행은 <표 5>의 전체 M·SD와 산술적으로 양립하지 않는다:
# 세 급수 집단의 M·SD를 그대로 두면 (집단 내 분산 + 집단 간 분산)만으로 전체 SD가 각각 ≥0.92, ≥0.95 가 되어
# <표 5>의 0.87, 0.74 를 넘는다('없음' 집단의 분산이 0이어도). 따라서 둘 중 하나를 골라야 한다.
#   priority="table5"  : <표 5>를 우선. 두 척도의 집단 차이(평균·SD 모두)는 방향만 유지하고 GROUP_SHRINK 배율로 축소. (기본)
#   priority="table89" : <표 8>·<표 9>를 우선. 두 척도의 급수별 M·SD를 논문대로 맞추고 전체 SD(표 5)는 포기.
GROUP_SHRINK = {"like_pos": 0.35, "revising": 0.35}


# ---------------------------------------------------------------------------
# 2. 보조 함수
# ---------------------------------------------------------------------------
def cronbach_alpha(X):
    X = np.asarray(X, float)
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / X.sum(axis=1).var(ddof=1))


def alpha_to_r(alpha, k):
    """Cronbach α → 평균 문항간 상관 (Spearman-Brown 역변환)"""
    return alpha / (k - alpha * (k - 1))


def nearest_corr(R, eps=1e-3):
    """대칭 행렬을 가장 가까운(고윳값 절단) 상관행렬로 투영"""
    R = (R + R.T) / 2
    w, V = np.linalg.eigh(R)
    w = np.clip(w, eps, None)
    R = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def exact_counts_vector(counts, rng):
    v = np.concatenate([np.full(n, k) for k, n in counts.items()])
    rng.shuffle(v)
    return v


def item_mean_targets():
    """하위척도 평균과 <표 6>의 상위/하위 1순위 문항을 만족하는 문항별 목표 평균"""
    t = {}
    for name, s in SUBSCALES.items():
        items = s["items"]
        fixed = {it: ITEM_TARGETS[it][0] for it in items if it in ITEM_TARGETS}
        free = [it for it in items if it not in fixed]
        rest_mean = (s["M"] * len(items) - sum(fixed.values())) / len(free)
        lo, hi = GROUP_RANGE[items[0][:2]]
        spread = min(0.10, rest_mean - lo - 0.02, hi - rest_mean - 0.02)
        offs = np.linspace(-spread, spread, len(free)) if len(free) > 1 else np.zeros(1)
        # 문항 순서에 따라 단조롭지 않도록 섞은 고정 순열
        perm = np.argsort(np.sin(np.arange(len(free)) * 2.7 + hash(name) % 7))
        for it, o in zip(free, offs[perm]):
            t[it] = rest_mean + o
        t.update(fixed)
    return t


# ---------------------------------------------------------------------------
# 3. 데이터 생성 (보정 루프 포함)
# ---------------------------------------------------------------------------
def generate(seed=20250618, n_iter=600, verbose=False, priority="table5"):
    assert priority in ("table5", "table89")
    rng = np.random.default_rng(seed)

    # --- 인적 사항 ---------------------------------------------------------
    topik = exact_counts_vector(TOPIK_COUNTS, rng)
    gender = exact_counts_vector(GENDER_COUNTS, rng)
    nation = exact_counts_vector(NATION_COUNTS, rng)
    # 학습 기간은 TOPIK 급수와 양의 상관을 갖도록 (급수 + 잡음)의 순위로 배정
    rank_score = topik + rng.normal(0, 1.2, N)
    order = np.argsort(rank_score)
    period = np.empty(N, int)
    period[order] = np.concatenate([np.full(n, k) for k, n in sorted(PERIOD_COUNTS.items())])

    # --- 잠재요인용 고정 난수 -----------------------------------------------
    all_items = [it for s in SUBSCALES.values() for it in s["items"]]
    J = len(all_items)
    item_sub = {it: i for i, s in enumerate(SUB_ORDER) for it in SUBSCALES[s]["items"]}
    sub_idx = np.array([item_sub[it] for it in all_items])
    Z = rng.standard_normal((N, 8))
    E = rng.standard_normal((N, J))

    # --- 초기 파라미터 -------------------------------------------------------
    tmean = item_mean_targets()
    mu = np.array([tmean[it] for it in all_items])              # 문항 평균
    lam = np.ones(J)                                            # 문항 부하량 배수
    sig = np.ones(J)                                            # 문항 고유오차 SD
    fsd = np.ones(8)                                            # 요인 SD
    for s_i, s in enumerate(SUB_ORDER):
        k = len(SUBSCALES[s]["items"])
        r = alpha_to_r(SUBSCALES[s]["alpha"], k)
        # 문항 SD ≈ 1.0 로 두고 요인분산 = r, 오차분산 = 1-r 로 시작
        fsd[s_i] = np.sqrt(r)
        sig[sub_idx == s_i] = np.sqrt(1 - r)
    for it, (m, sd) in ITEM_TARGETS.items():
        j = all_items.index(it)
        lam[j] = sd; sig[j] *= sd

    # 집단별 목표 평균·SD (없음 집단의 평균은 전체 평균이 맞도록 유도, SD는 목표 없음)
    n_g = np.array([TOPIK_COUNTS[g] for g in range(4)])
    gtarget = np.zeros((8, 4))                                  # 집단별 목표 평균
    gsd_target = np.full((8, 4), np.nan)                        # 집단별 목표 SD (없음 집단 = nan)
    sd_weight = np.ones(8)                                      # 전체 SD 목표의 가중치 (table89 모드에서 일부 0)
    for s_i, s in enumerate(SUB_ORDER):
        M = SUBSCALES[s]["M"]; SDall = SUBSCALES[s]["SD"]
        m123 = np.array(GROUP_MEANS[s]); sd123 = np.array(GROUP_SDS[s])
        m0 = (M * N - (n_g[1:] * m123).sum()) / n_g[0]
        m_all = np.concatenate([[m0], m123])
        shrink = GROUP_SHRINK.get(s, 1.0) if priority == "table5" else 1.0
        gtarget[s_i] = M + shrink * (m_all - M)
        gsd_target[s_i, 1:] = SDall + shrink * (sd123 - SDall)  # 축소 시 SD 차이도 같은 비율로 축소
        if priority == "table89" and s in GROUP_SHRINK:
            sd_weight[s_i] = 0.0                                # 전체 SD는 맞출 수 없으므로 목표에서 제외
    goff = gtarget - np.array([SUBSCALES[s]["M"] for s in SUB_ORDER])[:, None]   # 초기 오프셋
    gscale = np.ones((8, 4))                                    # 집단별 요인 SD 배율
    R = nearest_corr(CORR_TARGET.copy())

    ctarget = CORR_TARGET
    iu = np.triu_indices(8, 1)

    def simulate():
        F = Z @ np.linalg.cholesky(R).T
        lat = F * fsd * gscale[:, topik].T + goff[:, topik].T     # N×8 (집단별 배율·오프셋)
        Xc = mu + lam * lat[:, sub_idx] + sig * E
        return np.clip(np.rint(Xc), 1, 5).astype(int)

    def stats(X):
        S = np.column_stack([X[:, sub_idx == i].mean(axis=1) for i in range(8)])
        out = dict(
            item_mean=X.mean(axis=0), item_sd=X.std(axis=0, ddof=1),
            sub_sd=S.std(axis=0, ddof=1), corr=np.corrcoef(S.T),
            alpha=np.array([cronbach_alpha(X[:, sub_idx == i]) for i in range(8)]),
            gmean=np.array([[S[topik == g, i].mean() for g in range(4)] for i in range(8)]),
            gsd=np.array([[S[topik == g, i].std(ddof=1) for g in range(4)] for i in range(8)]),
        )
        return out, S

    tM = np.array([SUBSCALES[s]["M"] for s in SUB_ORDER])
    tSD = np.array([SUBSCALES[s]["SD"] for s in SUB_ORDER])
    tA = np.array([SUBSCALES[s]["alpha"] for s in SUB_ORDER])
    t_item_mean = np.array([tmean[i] for i in all_items])
    spec_j = [all_items.index(n) for n in ITEM_TARGETS]
    spec_sd = np.array([ITEM_TARGETS[n][1] for n in ITEM_TARGETS])

    def loss(o):
        """목표치와의 가중 오차 (반올림 때문에 진동하므로 최적 반복을 기억해 둔다)"""
        sub_mean = np.array([o["item_mean"][sub_idx == i].mean() for i in range(8)])
        gsd_err = np.nansum(np.abs(o["gsd"] - gsd_target))
        return (np.abs(sub_mean - tM).sum() * 4 + (sd_weight * np.abs(o["sub_sd"] - tSD)).sum() * 4
                + np.abs(o["alpha"] - tA).sum() * 4 + np.abs(o["corr"][iu] - ctarget[iu]).sum()
                + np.abs(o["item_sd"][spec_j] - spec_sd).sum() * 3
                + np.abs(o["item_mean"][spec_j] - t_item_mean[spec_j]).sum() * 3
                + np.abs(o["gmean"] - gtarget).sum() * 0.5 + gsd_err * 0.5)

    best = (np.inf, None)
    for it in range(n_iter):
        X = simulate()
        o, S = stats(X)
        L = loss(o)
        if L < best[0]:
            best = (L, (mu.copy(), lam.copy(), sig.copy(), fsd.copy(), goff.copy(), R.copy(), gscale.copy()))
        lr = 1.0 / (1.0 + it / 60.0)                 # 점차 작아지는 보정 폭
        # (a) 문항 평균
        mu += 0.8 * lr * (t_item_mean - o["item_mean"])
        # (b) 하위척도 SD 및 α : 요인/오차 비율과 전체 척도 조정
        for s_i, s in enumerate(SUB_ORDER):
            k = len(SUBSCALES[s]["items"]); m = sub_idx == s_i
            rt = alpha_to_r(tA[s_i], k)
            ro = np.clip(alpha_to_r(o["alpha"][s_i], k), 0.02, 0.98)
            adj = ((rt / (1 - rt)) / (ro / (1 - ro))) ** (0.25 * lr)
            scale = (tSD[s_i] / o["sub_sd"][s_i]) ** (0.7 * lr * sd_weight[s_i])
            fsd[s_i] *= scale * adj
            sig[m] *= scale / adj
        # (c) 특정 문항 SD
        for j, sd_t in zip(spec_j, spec_sd):
            f = (sd_t / o["item_sd"][j]) ** (0.5 * lr)
            lam[j] *= f; sig[j] *= f
        # (d) 잠재 상관
        R = nearest_corr(R + 0.6 * lr * (ctarget - o["corr"]))
        # (e) 집단 오프셋과 집단별 요인 SD 배율
        goff += 0.7 * lr * (gtarget - o["gmean"])
        for g in range(1, 4):
            gscale[:, g] *= (gsd_target[:, g] / o["gsd"][:, g]) ** (0.5 * lr)
        if priority == "table89":                     # 전체 SD 목표를 뺀 척도는 없음 집단 배율로 전체 SD를 최대한 보정
            for s_i in np.where(sd_weight == 0)[0]:
                gscale[s_i, 0] *= (tSD[s_i] / o["sub_sd"][s_i]) ** (0.5 * lr)
        if verbose and it % 100 == 0:
            err = np.abs(o["corr"][iu] - ctarget[iu]).max()
            print(f"iter {it:4d}  loss={L:.3f}  max|corr err|={err:.3f}  alpha={np.round(o['alpha'],3)}")

    mu, lam, sig, fsd, goff, R, gscale = best[1]
    X = simulate()
    o, S = stats(X)
    if verbose:
        print(f"best loss = {best[0]:.3f}")

    # --- 쓰기 경험 / 어려운 과제 / 요구 요소 -------------------------------
    z = (S - S.mean(0)) / S.std(0)
    eff = z[:, :3].mean(1)                        # 효능감 종합
    def pick(count, weight_z, beta=0.35):
        p = np.exp(beta * weight_z); p /= p.sum()
        idx = rng.choice(N, size=count, replace=False, p=p)
        v = np.zeros(N, int); v[idx] = 1
        return v
    exp_cols = {
        "exp_presentation": pick(54, z[:, 5]),
        "exp_report":       pick(41, z[:, 5]),
        "exp_thesis":       pick(12, (topik >= 2).astype(float)),
        "exp_exam":         pick(71, np.zeros(N)),
        "exp_other":        pick(29, np.zeros(N)),
    }
    diff_cols = {
        "diff_presentation": pick(51, -z[:, 3]),
        "diff_report":       pick(34, -z[:, 1]),
        "diff_thesis":       pick(62, -eff),
        "diff_exam":         pick(22, -z[:, 1]),
        "diff_other":        pick(14, np.zeros(N)),
    }
    need = exact_counts_vector(NEED_COUNTS, rng)

    df = pd.DataFrame({"id": np.arange(1, N + 1), "gender": gender, "nationality": nation,
                       "topik": topik, "study_period": period})
    for c, v in exp_cols.items(): df[c] = v
    for c, v in diff_cols.items(): df[c] = v
    df["need"] = need
    for j, it in enumerate(all_items):
        df[it] = X[:, j]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="survey_data.csv")
    ap.add_argument("--seed", type=int, default=20250618)
    ap.add_argument("--iter", type=int, default=600)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--priority", choices=["table5", "table89"], default="table5",
                    help="긍정 호감도·검토 전략에서 <표 5>(전체 M·SD)와 <표 8·9>(급수별 M·SD) 중 무엇을 우선할지")
    a = ap.parse_args()
    df = generate(seed=a.seed, n_iter=a.iter, verbose=a.verbose, priority=a.priority)
    df.to_csv(a.out, index=False, encoding="utf-8")
    print(f"saved {a.out}: {df.shape[0]} rows x {df.shape[1]} cols (all integer)")
    print(df.iloc[:5, :12].to_string(index=False))


if __name__ == "__main__":
    main()
