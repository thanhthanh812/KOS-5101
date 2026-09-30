data {
  int<lower=1> N;
  int<lower=1> S;
  matrix[N, S] y;                 // 하위 척도 점수 (열마다 독립적으로 추정)
}
parameters {
  vector[S] mu;
  vector<lower=0>[S] sigma;
}
model {
  mu ~ normal(3, 2);              // 약한 정보 사전분포
  sigma ~ normal(0, 2);           // 하한 0 이므로 반정규
  for (s in 1:S) y[:, s] ~ normal(mu[s], sigma[s]);
}
