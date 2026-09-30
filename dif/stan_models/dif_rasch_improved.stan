data {
  int<lower=1> N;                        // 전체 학생 수
  int<lower=1> J;                        // 문항 수
  array[N, J] int<lower=0, upper=1> Y;   // 응답 행렬
  array[N] int<lower=0, upper=1> group;  // 0 = 참조 집단, 1 = 초점 집단
}

parameters {
  real mu_focal;                 // 초점 집단의 평균 능력 (Impact 추정)
  vector[N] theta;               // 개별 학생 능력
  vector[J] beta;                // 문항 난이도 (참조 집단 기준)
  vector[J] delta;               // uniform DIF (초점 집단의 난이도 변화량)
}

model {
  // priors
  mu_focal ~ normal(0, 1);       // 집단 간 능력 차이에 대한 사전 분포
  beta ~ normal(0, 3);
  delta ~ normal(0, 1);          // DIF 파라미터에 대한 정규화(weakly regularizing)

  // 집단별 능력 분포 모델링 (DIF와 Impact 분리의 핵심)
  for (i in 1:N) {
    if (group[i] == 0) {
      theta[i] ~ normal(0, 1);   // 참조 집단은 표준 정규 분포 (Anchor)
    } else {
      theta[i] ~ normal(mu_focal, 1); // 초점 집단은 mu_focal을 중심으로 분포
    }
  }

  // Likelihood
  for (i in 1:N) {
    for (j in 1:J) {
      Y[i, j] ~ bernoulli_logit(theta[i] - (beta[j] + delta[j] * group[i]));
    }
  }
}

generated quantities {
  matrix[N, J] log_lik;
  for (i in 1:N) {
    for (j in 1:J) {
      log_lik[i, j] = bernoulli_logit_lpmf(
        Y[i, j] | theta[i] - (beta[j] + delta[j] * group[i])
      );
    }
  }
}
