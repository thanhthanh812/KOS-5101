
data {
  int<lower=1> Nsubj;
  int<lower=1> Ncat;
  array[Nsubj] int<lower=0> z;                 // 안타
  array[Nsubj] int<lower=1> N;                 // 타석
  array[Nsubj] int<lower=1, upper=Ncat> c;     // 선수의 포지션 인덱스 (nested)
}
parameters {
  real<lower=0, upper=1> omega0;               // 전체 최빈 타격능력
  real<lower=0> kappa0MinusTwo;                // 전체 집중도 - 2
  vector<lower=0, upper=1>[Ncat] omega;        // 포지션별 최빈
  vector<lower=0>[Ncat] kappaMinusTwo;         // 포지션별 집중도 - 2
  vector<lower=0, upper=1>[Nsubj] theta;       // 선수별 타격확률
}
transformed parameters {
  real<lower=2> kappa0 = kappa0MinusTwo + 2;
  vector<lower=2>[Ncat] kappa = kappaMinusTwo + 2;
}
model {
  omega0 ~ beta(1, 1);
  kappa0MinusTwo ~ gamma(0.01, 0.01);
  omega ~ beta(omega0*(kappa0-2)+1, (1-omega0)*(kappa0-2)+1);     // 전체 안의 포지션
  kappaMinusTwo ~ gamma(0.01, 0.01);
  theta ~ beta(omega[c].*(kappa[c]-2)+1, (1-omega[c]).*(kappa[c]-2)+1);  // 포지션 안의 선수
  z ~ binomial(N, theta);                                        // 가능도
}
