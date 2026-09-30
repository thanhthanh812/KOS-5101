// tut_rasch_dif.stan
// Bayesian Rasch (1PL) model WITH uniform DIF, separating impact and DIF.
//
//   logit P(Y_ij = 1) = theta_i - ( beta_j + delta_j * group_i )
//
// where group_i in {0 (reference), 1 (focal)}, delta_j is the uniform-DIF
// shift in difficulty for the focal group, and the focal group's mean
// ability mu_focal is estimated to keep group impact from contaminating
// the DIF estimates.
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N, J] int<lower=0, upper=1> Y;
  array[N] int<lower=0, upper=1> group;
}
parameters {
  real mu_focal;                          // focal group ability mean (impact)
  vector[N] theta;
  vector[J] beta;                         // baseline difficulty (reference)
  vector[J] delta;                        // uniform DIF (focal shift)
}
model {
  mu_focal ~ normal(0, 1);
  beta     ~ normal(0, 3);
  delta    ~ normal(0, 1);                // weakly regularising prior on DIF

  for (i in 1:N) {
    if (group[i] == 0) theta[i] ~ normal(0, 1);
    else               theta[i] ~ normal(mu_focal, 1);
  }
  for (i in 1:N)
    for (j in 1:J)
      Y[i, j] ~ bernoulli_logit(theta[i] - (beta[j] + delta[j] * group[i]));
}
generated quantities {
  matrix[N, J] log_lik;
  array[N, J] int Y_rep;
  for (i in 1:N)
    for (j in 1:J) {
      real eta = theta[i] - (beta[j] + delta[j] * group[i]);
      log_lik[i, j] = bernoulli_logit_lpmf(Y[i, j] | eta);
      Y_rep[i, j]  = bernoulli_logit_rng(eta);
    }
}
