data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=2> G;
  matrix[N, S] y;
  array[N] int<lower=1, upper=G> g;     // 집단 지표 (1=초급, 2=중급, 3=고급)
}
parameters {
  matrix[G, S] mu;
  matrix<lower=0>[G, S] sigma;
}
model {
  to_vector(mu) ~ normal(3, 2);
  to_vector(sigma) ~ normal(0, 2);
  for (s in 1:S) y[:, s] ~ normal(mu[g, s], sigma[g, s]);   // mu[g, s]: 관측별 소속 집단의 평균
}
