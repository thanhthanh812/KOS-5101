data {
  int<lower=1> N;
  int<lower=2> K;
  array[N] vector[K] X;           // 관측 벡터
  real<lower=0> eta;              // LKJ 형상모수 (1 = 상관행렬 위의 균일분포)
}
parameters {
  vector[K] mu;
  vector<lower=0>[K] tau;         // 표준편차
  cholesky_factor_corr[K] L;      // 상관행렬의 촐레스키 인수
}
model {
  mu ~ normal(3, 2);
  tau ~ normal(0, 2);
  L ~ lkj_corr_cholesky(eta);
  X ~ multi_normal_cholesky(mu, diag_pre_multiply(tau, L));
}
generated quantities {
  matrix[K, K] Omega = multiply_lower_tri_self_transpose(L);      // 상관행렬
  real alpha;
  {
    matrix[K, K] Sigma = quad_form_diag(Omega, tau);              // diag(tau) Omega diag(tau)
    alpha = K / (K - 1.0) * (1 - trace(Sigma) / sum(Sigma));
  }
}
