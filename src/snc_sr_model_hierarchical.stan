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
  int n_weeks_oos_young;
  int<lower=1> n_mice;
  real t0;
  real t0_oos_young;

  real ts[n_weeks];
  real cells[n_mice, n_weeks];
  real y0_oos_young[1];
}

transformed data {
  real x_r[0];
  int x_i[0];
}
parameters {
  real<lower=0> mu[4];
  real<lower=0> tau[4];
  real<lower=0> eta[n_mice, 2];
  real<lower=0> beta[n_mice, 2];
  real<lower=0> sigma;
  real<lower=0> y_init[n_mice, 1];
}

transformed parameters{
  real y[n_mice, n_weeks, 1];
  {
    real theta[n_mice, 4];
    theta[, 1:2] = eta;
    theta[, 3:4] = beta;
    for (i in 1:n_mice) {
      y[i] = integrate_ode_rk45(snc, y_init[i], t0, ts, theta[i, ], x_r, x_i);
    }
  }
}

model {
  // Priors
  mu ~ normal(0, 0.78);
  tau ~ normal(0, 1);

  sigma ~ lognormal(-1, 1);
  
  for (i in 1:n_mice) {
    eta[i] ~ normal(mu[1:2], tau[1:2]);
    beta[i] ~ normal(mu[3:4], tau[3:4]);
  }
  // Sample cell counts for each mouse.
  for (i in 1:n_mice) {
    y_init[i, 1] ~ lognormal(log(1), 1);
    cells[i] ~ lognormal(log(y[i, :, 1]), sigma);
  }
}

generated quantities {
  real pred_cells[n_mice, n_weeks];
  // real pred_cells_means[n_weeks];
  // real pred_cells_oos_young[n_weeks_oos_young];
  real log_likelihood = 0;
  // real oos_weeks_young[n_weeks_oos_young];
  // real theta[4];
  
  for (i in 1:n_mice) {
    pred_cells[i] = lognormal_rng(log(y[i, :, 1]), sigma);
    for (j in 1:n_weeks) {
      log_likelihood += lognormal_lpdf(cells[i, j] | log(y[i, j, 1]), sigma);
    }
  }
  // 
  // for (i in 1:n_weeks) {
  //   pred_cells_means[i] = mean(pred_cells[, i]);
  // }
  // 
  // Posterior predictive samples for young mice out-of-equilibrium example.
  // theta[1:2] = eta;
  // theta[3:4] = beta;
  // for (i in 1:n_weeks_oos_young) {
  //   oos_weeks_young[i] = t0_oos_young + i ;
  // }
  // pred_cells_oos_young = lognormal_rng(
  //   log(integrate_ode_rk45(
  //     snc, y0_oos_young, t0, oos_weeks_young, theta, x_r, x_i
  //   )[, 1]),
  //   sigma
  // );
}
