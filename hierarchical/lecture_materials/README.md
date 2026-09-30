# Gelman, Hill, & Yajima (2012) 강의 자료

**Why We (Usually) Don't Have to Worry About Multiple Comparisons**
*Journal of Research on Educational Effectiveness*, 5: 189–211.

학생들에게 **다회검정** (multiple comparisons; 多回檢定) 문제, Type S/M error, 그리고 베이지안 다층(hierarchical) 모형을 가르치기 위한 종합 자료입니다.

> **용어 안내**: 본 자료에서는 ``다회검정''을 표준 용어로 사용합니다. 한국 통계학회에서 통용되는 ``다중비교''와 동의어입니다.

---

## 자료 구성

```
lecture_materials/
├── README.md                            ← 이 문서
├── code/
│   ├── eight_schools_nc.stan            ← Eight Schools Stan 모델 (primary)
│   ├── ihdp_multilevel.stan             ← IHDP 다층모형 Stan 모델 (primary)
│   ├── 01_eight_schools_stan.py         ← Eight Schools 분석 — Stan/cmdstanpy [PRIMARY]
│   ├── 01_eight_schools_numpyro.py      ← Eight Schools 분석 — NumPyro
│   ├── 02_ihdp_simulation_stan.py       ← IHDP 시뮬레이션 — Stan/cmdstanpy [PRIMARY]
│   ├── 02_ihdp_simulation_numpyro.py    ← IHDP 시뮬레이션 — NumPyro
│   ├── 03_type_s_m_errors.py            ← Type S/M error 시뮬레이션
│   └── 04_shrinkage_zscore.py           ← Figure 4 (z-score shrinkage) 재현
├── figures/                             ← 모든 .png + 결과 .npz
├── latex/
│   ├── lecture_notes.tex                ← 11페이지 강의노트 본문
│   ├── lecture_notes.pdf                ← 컴파일된 PDF
│   ├── slides.tex                       ← 30+ 페이지 Beamer 슬라이드
│   └── slides.pdf                       ← 컴파일된 PDF
└── html/
    └── shrinkage_explorer.html          ← 인터랙티브 위젯 (τ 슬라이더)
```

## MCMC 도구 정책

본 강의의 **주력 MCMC 샘플러는 Stan (cmdstanpy)** 이다. Stan은:
- 베이지안 모형링의 사실상 표준
- 풍부한 진단/요약 도구 (Rhat, ESS, divergence check 등)
- 모형 정의가 매우 명시적

**보조 도구로 NumPyro** (JAX 기반 NUTS) 도 제공한다. 같은 모형의 NumPyro 버전을 비교하며 학생들이:
- 같은 NUTS 알고리즘이 다른 도구에서도 같은 결과를 주는지 확인
- 순수 파이썬 환경에서 작동하는 베이지안 추론 경험
- GPU/TPU 활용 가능성 탐색

두 도구의 결과는 (충분한 sample size에서) 사실상 일치한다.

## 사용 흐름

1. **사전 학습** — `lecture_notes.pdf`를 학생에게 미리 배포.
2. **수업** — `slides.pdf`로 강의. 마지막 30분은 `shrinkage_explorer.html`을 열어
   τ 슬라이더를 다같이 움직이며 shrinkage가 어떻게 변하는지 관찰.
3. **실습 1** — `01_eight_schools_stan.py` 실행해 Table 1 재현.
4. **실습 2** — `01_eight_schools_numpyro.py` 실행해 동일 결과가 다른 도구에서도 나오는지 확인.
5. **과제** — `02_ihdp_simulation_stan.py`의 `n_per_site=80`을 20으로 바꿔 실행. 어느 분석이 더 안정적인가?

## 환경 설정

```bash
# 공통 의존성
pip install numpy scipy pandas matplotlib

# Stan (주력)
pip install cmdstanpy
python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"

# NumPyro (보조)
pip install numpyro jax
```

## 코드 실행

```bash
cd code

# 1. Eight Schools — Stan (주력)
python3 01_eight_schools_stan.py

# 2. Eight Schools — NumPyro (보조, 비교용)
python3 01_eight_schools_numpyro.py

# 3. IHDP 시뮬레이션 — Stan (주력)
python3 02_ihdp_simulation_stan.py

# 4. IHDP 시뮬레이션 — NumPyro (보조)
python3 02_ihdp_simulation_numpyro.py

# 5. Type S/M 시뮬레이션
python3 03_type_s_m_errors.py

# 6. z-score shrinkage 그림
python3 04_shrinkage_zscore.py
```

## LaTeX 컴파일

```bash
cd latex
xelatex lecture_notes.tex   # 두 번 (목차 갱신)
xelatex lecture_notes.tex
xelatex slides.tex
```
한국어 + 한자 (多回檢定 등)를 포함하므로 `xelatex` + Noto Sans 