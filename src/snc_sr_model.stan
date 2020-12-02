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
  int<lower=1> n_weeks;
  int<lower=1> n_mice;
  real y0[1];
  real t0;
  real ts[n_weeks];
  real cells[n_mice, n_weeks];
}

transformed data {
  real x_r[0];
  int x_i[0];
}
parameters {
  real<lower=0> eta[2];
  real<lower=0> beta[2];
  real<lower=0> eps_inv;
}
transformed parameters{
  real y[n_mice, n_weeks, 1];
  real eps = 1. / eps_inv;
  {
    real theta[4];
    theta[1:2] = eta;
    theta[3:4] = beta;
    for (i in 1:n_mice) {
      y[i] = integrate_ode_rk45(snc, y0, t0, ts, theta, x_r, x_i);
    }
  }
}

model {
  // Priors
  for (i in 1:2) {
    eta[i] ~ normal(0.5, 0.5);
  }
  for (i in 1:2) {
    beta[i] ~ normal(0.5, 0.5);
  }
  eps_inv ~ gamma(2, .1);
  
  // Sample cell counts for each mice.
  for (i in 1:n_mice) {
    for (j in 1:n_weeks) {
      cells[i, j] ~ normal(y[i, j, 1], eps);
    }
  }
}

generated quantities {
  real pred_cells[n_mice, n_weeks];
  real pred_cells_means[n_weeks];
  for (i in 1:n_mice) {
    for (j in 1:n_weeks) {
      pred_cells[i, j] = normal_rng(y[i, j, 1], eps);
    }
  }
  for (i in 1:n_weeks) {
    pred_cells_means[i] = mean(pred_cells[:, i]);
  }
}
