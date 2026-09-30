
data {
  int<lower=0> N;
  int<lower=0, upper=N> y;
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  theta ~ beta(3, 3);
  y ~ binomial(N, theta);
}
generated quantities {
  int<lower=0, upper=1> lt_045;
  lt_045 = theta < 0.45;
}
