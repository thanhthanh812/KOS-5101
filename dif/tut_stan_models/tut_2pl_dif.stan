// tut_2pl_dif.stan
// Bayesian 2PL IRT model WITH both uniform and non-uniform DIF.
//
//   logit P(Y_ij = 1) =
//       ( alpha_j + delta_a_j * group_i )
//     * ( theta_i - ( beta_j + delta_b_j * group_i ) )
//
// delta_b_j : uniform DIF (focal-group difficulty shift)
// delta_a_j : non-uniform DIF (focal-group discrimination shift)
// mu_focal   : focal group ability mean (impact)
data {
  int<lower=1> N;
  int<lower=1> J;
  array[N, J] int<lower=0, upper=1> Y;
  array[N] int<lower=0, upper=1> group;
}
parameters {
  real mu_focal;
  vector[N] theta;
  vector<lower=0>[J] alpha;               // baseline discrimination > 0
  vector[J] beta;                         // baseline difficulty
  vector[J] delta_b;                      // uniform DIF
  vector[J] delta_a;                      // non-uniform DIF (discrim. shift)
}
model {
  mu_focal ~ normal(0, 1);
  alpha    ~ lognormal(0, 0.5);
  beta     ~ normal(0, 3);
  delta_b  ~ normal(0, 1);
  delta_a  ~ normal(0, 1);     // 균일/비균일 DIF prior 동일 폭으로 완화

  for (i in 1:N) {
    if (group[i] == 0) theta[i] ~ normal(0, 1);
    else               theta[i] ~ normal(mu_focal, 1);
  }
  for (i in 1:N)
    for (j in 1:J) {
      real a_ij = alpha[j] + delta_a[j] * group[i];
      real b_ij = beta[j]  + delta_b[j] * group[i];
      Y[i, j] ~ bernoulli_logit(a_ij * (theta[i] - b_ij));
    }
}
generated quantities {
  matrix[N, J] log_lik;
  array[N, J] int Y_rep;
  for (i in 1:N)
    for (j in 1:J) {
      real a_ij = alpha[j] + delta_a[j] * group[i];
      real b_ij = beta[j]  + delta_b[j] * group[i];
      real eta  = a_ij * (theta[i] - b_ij);
      log_lik[i, j] = bernoulli_logit_lpmf(Y[i, j] | eta);
      Y_rep[i, j]  = bernoulli_logit_rng(eta);
    }
}
