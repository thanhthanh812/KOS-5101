// tut_rasch_baseline.stan - IMPROVED VERSION
// Bayesian Rasch (1PL) model WITHOUT item-level DIF parameters,
// but WITH a free group-mean (mu_focal) so that group-level
// IMPACT is accounted for. This is the recommended PPMC baseline
// (Joo & Lee, 2022): discrepancies under this no-DIF / with-impact
// model isolate item-by-group misfit attributable to DIF.
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N, J] int<lower=0, upper=1> Y;
  array[N] int<lower=0, upper=1> group;     // 0 = reference, 1 = focal
}
parameters {
  real mu_focal;                            // focal-group ability mean
  vector[N] theta;
  vector[J] beta;
}
model {
  mu_focal ~ normal(0, 1);
  beta     ~ normal(0, 3);
  for (i in 1:N) {
    if (group[i] == 0) theta[i] ~ normal(0, 1);
    else               theta[i] ~ normal(mu_focal, 1);
  }
  for (i in 1:N)
    for (j in 1:J)
      Y[i, j] ~ bernoulli_logit(theta[i] - beta[j]);
}
generated quantities {
  // Replicated data set for PPMC.
  array[N, J] int Y_rep;
  for (i in 1:N)
    for (j in 1:J)
      Y_rep[i, j] = bernoulli_logit_rng(theta[i] - beta[j]);
}
