functions {
  real[] snc(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {

      real C = y[1];
      real eta[2] = theta[3:4];
      real beta[2] = theta[1:2];

      
      real dC_dt_pos = eta[1] + eta[2] * t;
      real dC_dt_neg = (beta[1]) / (1 + beta[2] * C) * C;
      real dC_dt = dC_dt_pos - dC_dt_neg;
      return {dC_dt};
  }
}

data {
  int<lower=1> n_obs;
  int<lower=1> n_mice;
  int<lower=1> n_weeks;
  real t0;
  int tt[n_obs];
  real ts[n_weeks];
  real y0[1];
  real cells[n_obs];
}

transformed data {
  real x_r[0];
  int x_i[0];
}

parameters {
  real<lower=0> beta[2];
  real<lower=0> eta[2];
  real<lower=0> sigma;
}

transformed parameters{
  real y[n_weeks, 1];
  {
    real theta[4];
    theta[1:2] = beta;
    theta[3:4] = eta;
    y = integrate_ode_rk45(snc, y0, t0, ts, theta, x_r, x_i);
  }
}

model {
  beta[1] ~ normal(.8959, 0.4919);
  beta[2] ~ normal(0.9002, 0.4872);
  eta[1] ~ normal(0.2377, 0.1633);
  eta[2] ~ normal(0.0089, 0.0058);

  sigma ~ lognormal(-1, 1);

  for (i in 1:n_obs) {
    cells[i] ~ lognormal(log(y[tt[i], 1]), sigma);
  }
}

generated quantities {
  real pred_cells[n_weeks];
  real log_likelihood = 0;
  pred_cells = lognormal_rng(log(y[, 1]), sigma);
  
  for (j in 1:n_obs) {
    log_likelihood += lognormal_lpdf(cells[j] | log(y[tt[j], 1]), sigma);
  }
}
