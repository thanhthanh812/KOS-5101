
data {
  int<lower=0> J;            // 학교 수 (number of schools)
  vector[J] y;               // 추정 효과 (estimated effects)
  vector<lower=0>[J] sigma;  // 알려진 표준오차 (known standard errors)
}
parameters {
  real mu;                   // 전체 평균  mu
  real<lower=0> tau;         // 학교 간 표준편차  tau
  vector[J] theta;           // 각 학교의 참 효과  theta_j
}
model {
  theta ~ normal(mu, tau);   // theta_j ~ N(mu, tau^2)        <- 계층 사전 (prior)
  y ~ normal(theta, sigma);  // y_j     ~ N(theta_j, sigma_j^2) <- 가능도 (likelihood)
  // mu, tau 사전 미지정 => p(mu,tau) propto 1  (균일/improper uniform)
}
