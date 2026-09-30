// IHDP-like multilevel model (Gelman, Hill, Yajima 2012, eq. on p.195)
//
// y_i ~ Normal( gamma_{j[i]} + delta_{j[i]} * P_i , sigma_y^2 )
// gamma_j ~ Normal(mu_gamma, sigma_gamma^2)
// delta_j ~ Normal(mu_delta, sigma_delta^2)   <-- this is the key
//                                                shared distribution
//
// j[i] = site index of student i  (1..J)
// P_i  = treatment indicator (0/1)
//
// The classical "site by site" analysis is the limit sigma_delta -> infinity,
// i.e. completely separate delta_j's with no shared prior.  Bonferroni adds an
// ad-hoc width correction.  Here we instead *model* the across-site distribution
// of treatment effects, which yields partial pooling = shrinkage toward mu_delta.

data {
  int<lower=1> N;                      // number of students
  int<lower=1> J;                      // number of sites
  array[N] int<lower=1, upper=J> site; // site index for each student
  vector[N] P;                         // treatment indicator (0 / 1)
  vector[N] y;                         // outcome (test score)
}

parameters {
  real mu_gamma;
  real mu_delta;
  real<lower=0> sigma_gamma;
  real<lower=0> sigma_delta;
  real<lower=0> sigma_y;
  vector[J] gamma_raw;
  vector[J] delta_raw;
}

transformed parameters {
  vector[J] gamma = mu_gamma + sigma_gamma * gamma_raw;
  vector[J] delta = mu_delta + sigma_delta * delta_raw;
}

model {
  // Weakly informative priors
  mu_gamma     ~ normal(0, 100);
  mu_delta     ~ normal(0, 100);
  sigma_gamma  ~ normal(0, 50);   // half-normal
  sigma_delta  ~ normal(0, 50);   // half-normal -- between-site SD of treatment effect
  sigma_y      ~ normal(0, 50);
  gamma_raw    ~ std_normal();
  delta_raw    ~ std_normal();

  // Likelihood
  {
    vector[N] mu_i;
    for (i in 1:N) {
      mu_i[i] = gamma[site[i]] + delta[site[i]] * P[i];
    }
    y ~ normal(mu_i, sigma_y);
  }
}
