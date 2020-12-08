functions {
  real[] snc(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {

      real C = y[1];
      real eta[2] = theta[1:2];
      real beta[2] = theta[3:4];

      
      real dC_dt_pos = (eta[1] + eta[2] * t);
      real dC_dt_neg = (beta[1]) / (1 + beta[2] * C) * C;
      real dC_dt = dC_dt_pos - dC_dt_neg;
      return {dC_dt};
  }
}

data {
  int<lower=1> n_days;
  int<lower=1> n_mice;
  real t0;
  real ts[n_days];
  real cells[n_mice, n_days];
}

transformed data {
  real x_r[0];
  int x_i[0];
}
parameters {
  real<lower=0> eta[2];
  real<lower=0> beta[2];
  real<lower=0> sigma;
  real<lower=0> y_init[n_mice, 1];
}
transformed parameters{
  real y[n_mice, n_days, 1];
  {
    real theta[4];
    theta[1:2] = eta;
    theta[3:4] = beta;
    for (i in 1:n_mice) {
      y[i] = integrate_ode_rk45(snc, y_init[i], t0, ts, theta, x_r, x_i);
    }
  }
}

model {
  // Priors
  eta[1:2] ~ normal(0, 0.5);
  beta[1:2] ~ normal(0, 0.5);
  sigma ~ lognormal(-1, 1);
  
  // Sample cell counts for each mouse.
  for (i in 1:n_mice) {
    y_init[i] ~ lognormal(log(1), 1);
    cells[i] ~ lognormal(log(y[i, , 1]), sigma);
  }
}

generated quantities {
  real pred_cells[n_mice, n_days];
  real pred_cells_means[n_days];
  real log_likelihood = 0;
  
  for (i in 1:n_mice) {
    pred_cells[i] = lognormal_rng(log(y[i, , 1]), sigma);
    for (j in 1:n_days) {
      log_likelihood += lognormal_lpdf(cells[i, j] | log(y[i, j, 1]), sigma);
    }
  }
  for (i in 1:n_days) {
    pred_cells_means[i] = mean(pred_cells[:, i]);
  }
}
