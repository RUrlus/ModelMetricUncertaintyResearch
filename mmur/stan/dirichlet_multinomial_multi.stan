data {
    int N; // num observations
    real<lower=0> alpha;
    int<lower=2> total_count;
    int y[N, 4]; // multinomial observations
}

parameters {
    simplex[4] theta;
}

model {
    theta ~ dirichlet(rep_vector(alpha, 4));
    {
        for (n in 1:N) {
            y[n] ~ multinomial(theta);
        }
    }
}

generated quantities {
    int y_hat[4] = multinomial_rng(theta, total_count);
}
