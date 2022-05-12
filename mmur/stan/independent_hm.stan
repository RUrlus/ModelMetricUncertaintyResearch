data {
  real<lower=0> alpha; // hyperprior for population distribution
  real<lower=0> gam_a; // prior parameters for gam
  real<lower=0> gam_b;

  int<lower=1> N; // number of confusion matrices
  int<lower=1> total_count; // total count
  int y[N, 4]; // multinomial observations
}

parameters {
  simplex[4] pi_pop; // overarching distribution of CM probabilities
  simplex[4] pi[N]; // probabilities for all N confusion matrices
  real<lower=0.000001> gam;
}

model {
  pi_pop ~ dirichlet(rep_vector(alpha, 4));
  gam ~ gamma(gam_a,gam_b);
  {
  for (n in 1:N) {
      pi[n] ~ dirichlet(gam * pi_pop);
      y[n] ~ multinomial(pi[n]);
    }
  }
}

generated quantities {
    vector[4] pi_hat = dirichlet_rng(pi_pop * gam);
    int y_hat[4] = multinomial_rng(pi_hat, total_count);
}