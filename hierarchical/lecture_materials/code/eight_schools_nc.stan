// Eight Schools (Rubin 1981) - Non-centered parameterization
// Gelman, Hill, Yajima (2012), Table 1
//
// Data: y_j (point estimate from school j) and sigma_j (its SE) for J schools.
// Model:
//   y_j  ~ Normal(theta_j, sigma_j)
//   theta_j = mu + tau * eta_j         (non-centered)
//   eta_j ~ Normal(0, 1)
//   mu    ~ Normal(0, 10)               (weakly informative)
//   tau   ~ Half-Normal(0, 10)          (weakly informative on between-school SD)
//
// Non-centered parameterization (Matt trick) prevents the funnel
// pathology that plagues hierarchical models when tau is small.

data {
  int<lower=1> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}

parameters {
  real mu;
  real<lower=0> tau;
  vector[J] eta;
}

transformed parameters {
  vector[J] theta = mu + tau * eta;
}

model {
  // Priors
  mu  ~ normal(0, 10);
  tau ~ normal(0, 10);   // truncated to >=0 -> half-normal
  eta ~ std_normal();

  // Likelihood
  y ~ normal(theta, sigma);
}

generated quantities {
  // Posterior predictive for each school's "true" effect difference
  // (we let downstream code compute pairwise contrasts from theta).
  real shrinkage_factor;
  // Average shrinkage: 1 means full shrinkage to mu, 0 means none.
  // For a normal hierarchical model with equal sigmas, the shrinkage
  // factor for school j is sigma_j^2 / (sigma_j^2 + tau^2).
  {
    vector[J] s;
    for (j in 1:J) {
      s[j] = square(sigma[j]) / (square(sigma[j]) + square(tau));
    }
    shrinkage_factor = mean(s);
  }
}
