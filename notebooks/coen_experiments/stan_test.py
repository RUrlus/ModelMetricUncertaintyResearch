import stan
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ["CC"] = "/opt/rh/devtoolset-10/root/usr/bin/gcc"
os.environ["CXX"] = "/opt/rh/devtoolset-10/root/usr/bin/g++"



cm_code = """
data {
  int<lower=1> N; // number of confusion matrices
  int y[N, 4]; // multinomial observations
  real<lower=0> alpha; // hyperprior for population distribution

  real<lower=0> gam_a;
  real<lower=0> gam_b;
}

parameters {
  vector[4] pi_pop; // overarching distribution of CM probabilities
  simplex[N,4] pi; // probabilities for a single confusion matrix
  real gamma;
}

model {
  pi_pop ~ dirichlet(rep_vector(alpha, 4));
  gamma ~ gamma(gam_a,gam_b)
  {
  for (n in 1:N) {
      pi[n] ~ dirichlet(gamma * pi_pop);
      y[n] ~ multinomial(theta);
    }
  }
}
"""

regression_code = """
data {
  int<lower=1> N;
  vector[N] y;
}

parameters {
  real mu;
  real<lower=0> tau;
}

model {
  mu ~ normal(0, 5);
  tau ~ normal(0, 5);
  y ~ normal(mu, tau);
}
"""


y = np.random.normal(-1,3,size=(200))
print(y)
pooled_data_dict = {'N': len(y),
               'y': y}
posterior = stan.build(regression_code,data=pooled_data_dict,random_seed=1)
fit = posterior.sample(num_chains=4,num_samples=1000)
print(np.array(fit['mu']).shape)
plt.scatter(np.arange(4000),fit['mu'][0])
plt.show()