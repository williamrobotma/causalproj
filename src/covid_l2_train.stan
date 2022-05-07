
// data {
//   int<lower = 0> N; // number of observations
//   int<lower = 0> K; // number of covariates
//   matrix[N, K]   a; // sensitive variables
//   real           ugpa[N]; // UGPA
//   int            lsat[N]; // LSAT
//   real           zfya[N]; // ZFYA
  
// }

// transformed data {
  
//  vector[K] zero_K;
//  vector[K] one_K;
 
//  zero_K = rep_vector(0,K);
//  one_K = rep_vector(1,K);

// }

// parameters {

//   vector[N] u;

//   real ugpa0;
//   real eta_u_ugpa;
//   real lsat0;
//   real eta_u_lsat;
//   real eta_u_zfya;
  
//   vector[K] eta_a_ugpa;
//   vector[K] eta_a_lsat;
//   vector[K] eta_a_zfya;
  
  
//   real<lower=0> sigma_g_Sq;
// }

// transformed parameters  {
//  // Population standard deviation (a positive real number)
//  real<lower=0> sigma_g;
//  // Standard deviation (derived from variance)
//  sigma_g = sqrt(sigma_g_Sq);
// }

// model {
  
//   // don't have data about this
//   u ~ normal(0, 1);
  
//   ugpa0      ~ normal(0, 1);
//   eta_u_ugpa ~ normal(0, 1);
//   lsat0     ~ normal(0, 1);
//   eta_u_lsat ~ normal(0, 1);
//   eta_u_zfya ~ normal(0, 1);

//   eta_a_ugpa ~ normal(zero_K, one_K);
//   eta_a_lsat ~ normal(zero_K, one_K);
//   eta_a_zfya ~ normal(zero_K, one_K);

//   sigma_g_Sq ~ inv_gamma(1, 1);

//   // have data about these
//   ugpa ~ normal(ugpa0 + eta_u_ugpa * u + a * eta_a_ugpa, sigma_g);
//   lsat ~ poisson(exp(lsat0 + eta_u_lsat * u + a * eta_a_lsat));
//   zfya ~ normal(eta_u_zfya * u + a * eta_a_zfya, 1);

// }


data {
  int<lower = 0> N; // number of observations
  int<lower = 0> Kp; // number of covariates
  int<lower = 0> Ks;
  int<lower = 0> Kt;

  matrix[N, Kp]   p; // sensitive variables
  matrix[N, Ks]   s; // sensitive variables
  matrix[N, Kt]   t; // sensitive variables
  matrix[N, Kogit statut]   t; // sensitive variables

  real           Y[N]; // UGPA
  real           o[N];
  int            T[N]; // LSAT
  
}

transformed data {
  
 vector[Kp] zero_Kp;
 vector[Kp] one_Kp;

 vector[Ks] zero_Ks;
 vector[Ks] one_Ks;

 vector[Kt] zero_Kt;
 vector[Kt] one_Kt;
 
 zero_Kp = rep_vector(0,Kp);
 one_Kp = rep_vector(1,Kp);

 zero_Ks = rep_vector(0,Ks);
 one_Ks = rep_vector(1,Ks);

 zero_Kt = rep_vector(0,Kt);
 one_Kt = rep_vector(1,Kt);

}

parameters {

  vector[N] u_t;
  vector[N] u_o;
  vector[N] u_y;

  real t0;
  real eta_u_t;

  real o0;
  real eta_u_o;

  real y0;
  real eta_u_y;
  
  vector[Kp+Ks] eta_in_t;
  vector[Ks+Kt] eta_in_o;
  vector[Kp+Ks+1] eta_in_y;
  
  
  real<lower=0> sigma_g_Sq;
}

transformed parameters  {
 // Population standard deviation (a positive real number)
 real<lower=0> sigma_g;
 // Standard deviation (derived from variance)
 sigma_g = sqrt(sigma_g_Sq);
}

model {
  
  // don't have data about this
  u_a ~ normal(0, 1);
  u_y ~ normal(0, 1);
  
  ugpa0      ~ normal(0, 1);
  eta_u_ugpa ~ normal(0, 1);
  lsat0     ~ normal(0, 1);
  eta_u_lsat ~ normal(0, 1);
  eta_u_zfya ~ normal(0, 1);

  eta_a_ugpa ~ normal(zero_K, one_K);
  eta_a_zfya ~ normal(zero_K, one_K);

  sigma_g_Sq ~ inv_gamma(1, 1);

  // have data about these
  ugpa ~ normal(ugpa0 + eta_u_ugpa * u + a * eta_a_ugpa, sigma_g);
  // lsat ~ poisson(exp(lsat0 + eta_u_lsat * u + a * eta_a_lsat));
  zfya ~ normal(eta_u_zfya * u + a * eta_a_zfya, 1);

}
