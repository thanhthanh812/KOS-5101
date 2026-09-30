data {
  int<lower=1> N;
  int<lower=1> P;
  matrix[N, P] X;                 // 평균 중심화한 독립변수
  vector[N] y;
  real<lower=0> prior_sd;         // 회귀계수 사전분포의 표준편차
}
parameters {
  real a;
  vector[P] b;
  real<lower=0> sigma;
}
model {
  a ~ normal(0, 5);
  b ~ normal(0, prior_sd);
  sigma ~ normal(0, 2);
  y ~ normal_id_glm(X, a, b, sigma);      // y ~ normal(a + X * b, sigma)
}
