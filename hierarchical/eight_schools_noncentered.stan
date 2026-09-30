
data {
  int<lower=0> J;
  vector[J] y;
  vector<lower=0>[J] sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  vector[J] eta;                    // eta_j ~ N(0,1)  표준 보조변수
}
transformed parameters {
  vector[J] theta = mu + tau * eta; // theta_j = mu + tau*eta_j  <=>  theta_j ~ N(mu, tau^2)
}
model {
  eta ~ std_normal();               // eta_j ~ N(0,1)
  y ~ normal(theta, sigma);         // y_j ~ N(theta_j, sigma_j^2)
}
