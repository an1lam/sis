functions {
  real[] snc(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {

      real C = y[1];
      real eta[2] = theta[3:4];
      real beta[2] = theta[1:2];
      real ll = theta[5];
      
      real dC_dt_pos = eta[1] + eta[2] * ll;
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
  real ll_param[2];
}

transformed data {
  real x_r[0];
  int x_i[0];
}

parameters {
  real<lower=0> beta[2];
  real<lower=0> ll;
  real<lower=0> sigma;
  real<lower=0> eta[2];
}

transformed parameters{
  real y[n_weeks, 1];
  {
    real theta[5];
    theta[1:2] = beta;
    
    theta[3:4] = eta;
    theta[5] = ll;
    y = integrate_ode_rk45(snc, y0, t0, ts, theta, x_r, x_i);
  }
}

//                     mean se_mean     sd      2.5%       25%       50%       75%     97.5% n_eff   Rhat
// eta[1]            0.1716  0.0031 0.1361    0.0078    0.0683    0.1385    0.2391    0.5217  1980 0.9996
// eta[2]            0.5760  0.0090 0.4094    0.0995    0.2788    0.4742    0.7532    1.6505  2075 1.0004
// beta[1]           0.8865  0.0108 0.4623    0.2192    0.5354    0.8025    1.1538    1.9779  1837 0.9998
// beta[2]           1.0049  0.0105 0.5076    0.2418    0.6328    0.9254    1.2937    2.2022  2354 1.0001
// sigma             0.9035  0.0009 0.0489    0.8145    0.8690    0.8998    0.9355    1.0078  3160 1.0004
// alpha             0.0185  0.0002 0.0121    0.0038    0.0094    0.0153    0.0247    0.0489  2460 0.9995
// ll                0.1267  0.0018 0.1002    0.0055    0.0466    0.1050    0.1841    0.3660  3226 1.0004
// log_likelihood -281.1475  0.0492 1.7729 -285.1483 -282.2717 -280.8884 -279.7500 -278.5851  1296 1.0019
// lp__            -81.4398  0.0690 2.2988  -86.9365  -82.7057  -81.1206  -79.8297  -78.0002  1109 1.0050
model {
  // Priors drawn from the corresponding longitudinal model
  eta[1] ~ normal(.1716, .1361);
  eta[2] ~ normal(.5760, .4094);
  beta[1] ~ normal(0.8865, 0.4623);
  beta[2] ~ normal(1.0049, 0.5076);
  ll ~ normal(ll_param[1], ll_param[2]);

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
