functions {
  real[] snc(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {

      real C = y[1];
      real eta[3] = theta[1:3];
      real beta[3] = theta[4:6];

      
      real dC_dt_pos = (eta[1] + eta[2] * t) * (1 + eta[3] * C);
      real dC_dt_neg = (beta[1] - beta[2] * t) / (1 + beta[3] * C) * C;
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
  real<lower=0> eta[3];
  real<lower=0> beta[3];
  real<lower=0> sigma;
  real<lower=0> y_init[n_mice, 1];
}
transformed parameters{
  real y[n_mice, n_weeks, 1];
  {
    real theta[6];
    theta[1:3] = eta;
    theta[4:6] = beta;
    for (i in 1:n_mice) {
      y[i] = integrate_ode_rk45(snc, y_init[i], t0, ts, theta, x_r, x_i);
    }
  }
}

model {
  // Priors
  eta[1:3] ~ normal(0.5, 0.5);
  beta[1:3] ~ normal(0.5, 0.5);
  sigma ~ lognormal(-1, 1);
  
  // Sample cell counts for each mouse.
  for (i in 1:n_mice) {
    y_init[i] ~ lognormal(log(1), 1);
    cells[i] ~ lognormal(log(y[i, , 1]), sigma);
  }
}

generated quantities {
  real pred_cells[n_mice, n_weeks];
  real pred_cells_means[n_weeks];
  real log_likelihood = 0;
  
  for (i in 1:n_mice) {
    pred_cells[i] = lognormal_rng(log(y[i, , 1]), sigma);
    for (j in 1:n_weeks) {
      log_likelihood += lognormal_lpdf(cells[i, j] | log(y[i, j, 1]), sigma);
    }
  }
  for (i in 1:n_weeks) {
    pred_cells_means[i] = mean(pred_cells[:, i]);
  }
}
