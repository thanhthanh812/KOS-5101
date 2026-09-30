data {
  int<lower=1> K;                 // 범주 수
  array[K] int<lower=0> n;        // 범주별 도수
  vector<lower=0>[K] a;           // 디리클레 사전분포 모수
}
parameters {
  simplex[K] theta;               // 비율 (합 = 1)
}
model {
  theta ~ dirichlet(a);           // 사전분포
  n ~ multinomial(theta);         // 우도
}
