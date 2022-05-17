data {
    real<lower=0> alpha;
    int<lower=2> total_count;
    int y[4]; // multinomial observations
}

parameters {
    simplex[4] theta;
}

model {
    theta ~ dirichlet(rep_vector(alpha, 4));
    y ~ multinomial(theta);
}

generated quantities {
    int y_hat[4] = multinomial_rng(theta, total_count);
}