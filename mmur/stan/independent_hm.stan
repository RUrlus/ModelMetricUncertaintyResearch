data {
  int<lower=1> N; // number of confusion matrices
  int y[N, 4]; // multinomial observations
  real<lower=0> alpha; // hyperprior for population distribution

  real<lower=0> gam_a;
  real<lower=0> gam_b;
}

parameters {
  simplex[4] pi_pop; // overarching distribution of CM probabilities
  simplex[4] pi[N]; // probabilities for a single confusion matrix
  real gamma;
}

model {
  pi_pop ~ dirichlet(rep_vector(alpha, 4));
  gamma ~ gamma(gam_a,gam_b)
  {
  for (n in 1:N) {
      pi[n] ~ dirichlet(gamma * pi_pop);
      y[n] ~ multinomial(pi[n]);
    }
  }
}