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
  real<lower=0> eps_inv;
}
transformed parameters{
  real y[n_mice, n_weeks, 1];
  real eps = 1. / eps_inv;
  {
    real theta[6];
    theta[1:3] = eta;
    theta[4:6] = beta;
    for (i in 1:n_mice) {
      y[i] = integrate_ode_rk45(snc, y0, t0, ts, theta, x_r, x_i);
    }
  }
}

model {
  // Priors
  for (i in 1:3) {
    eta[i] ~ normal(0.5, 0.5);
  }
  for (i in 1:3) {
    beta[i] ~ normal(0.5, 0.5);
  }
  eps_inv ~ gamma(2, 0);
  
  // Sample cell counts for each mice.
  for (i in 1:n_mice) {
    for (j in 1:n_weeks) {
      cells[i, j] ~ normal(y[i, j, 1], eps);
    }
  }
}
