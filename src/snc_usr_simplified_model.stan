functions {
  real[] snc(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {

      real C = y[1];
      real eta = theta[1];
      real beta[2] = theta[2:3];

      
      real dC_dt_pos = eta;
      real dC_dt_neg = (beta[1] - beta[2] * t) * C;
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
  real<lower=0> eta;
  real<lower=0> beta[2];
  real<lower=.001> eps_inv[n_weeks];
}
transformed parameters{
  real y[n_mice, n_weeks, 1];
  real eps[n_weeks];
  {
    real theta[3];
    theta[1] = eta;
    theta[2:3] = beta;
    for (j in 1:n_weeks) {
      eps[j] = 1. / eps_inv[j];
    }
    
    for (i in 1:n_mice) {
      y[i] = integrate_ode_bdf(snc, y0, t0, ts, theta, x_r, x_i);
    }
  }
}

model {
  // Priors
  // eta ~ gamma(2, 0.1);
  // for (i in 1:2) {
  //   beta[i] ~ gamma(2., 0.5);
  // }
  eps_inv ~ gamma(2, 0.1);
  // 
  // Sample cell counts for each mouse.
  print("eta: ", eta, "beta: ", beta, "eps: ", eps)
  for (i in 1:n_mice) {
    for (j in 1:n_weeks) {
      cells[i, j] ~ normal(y[i, j, 1], eps[j]);
    }
  }
}

generated quantities {
  real pred_cells[n_mice, n_weeks];
  real pred_cells_means[n_weeks];
  for (i in 1:n_mice) {
    for (j in 1:n_weeks) {
      pred_cells[i, j] = normal_rng(y[i, j, 1], eps[j]);
    }
  }
  for (i in 1:n_weeks) {
    pred_cells_means[i] = mean(pred_cells[:, i]);
  }
}
