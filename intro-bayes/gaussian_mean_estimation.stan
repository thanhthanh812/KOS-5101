
data {
  int<lower=1> N;
  array[N] real y;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 50);
  sigma ~ cauchy(0, 10);
  y ~ normal(mu, sigma);
}
