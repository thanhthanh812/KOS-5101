data {
  int<lower=1> N;
  int<lower=2> K;
  array[N] vector[K] X;           // 문항 응답
}
parameters {
  vector[K] mu;
  vector<lower=0>[K] lam;         // 요인부하량 (양수 제약)
  vector<lower=0>[K] psi;         // 문항 고유 표준편차
}
model {
  mu ~ normal(3, 2);
  lam ~ normal(0, 2);
  psi ~ normal(0, 2);
  X ~ multi_normal(mu, add_diag(lam * lam', square(psi)));   // Sigma = lam lam' + diag(psi^2)
}
generated quantities {
  real alpha;                     // Cronbach's α (Sigma 의 함수)
  real omega;                     // McDonald's ω
  {
    matrix[K, K] Sigma = add_diag(lam * lam', square(psi));
    alpha = K / (K - 1.0) * (1 - trace(Sigma) / sum(Sigma));
    omega = square(sum(lam)) / (square(sum(lam)) + sum(square(psi)));
  }
}
