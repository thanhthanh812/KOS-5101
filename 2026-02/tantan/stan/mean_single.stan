data {
  int<lower=1> N;                 // 관측된 것: 학생 수
  vector[N] y;                    //            학생들의 점수 (하위 척도 하나)
}
parameters {
  real mu;                        // 모르는 것: 모평균
  real<lower=0> sigma;            //            모표준편차 (0 이상)
}
model {
  mu ~ normal(3, 2);              // 자료를 보기 전의 믿음 (사전분포)
  sigma ~ normal(0, 2);           //   하한이 0 이므로 실제로는 반정규분포
  y ~ normal(mu, sigma);          // 자료가 생겨나는 방식 (우도): 점수 하나하나가 N(mu, sigma) 에서
}
