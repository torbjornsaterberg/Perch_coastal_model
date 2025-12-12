////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//                            age_model_missing_data_STAN                     //
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// This STAN program implements a simple cohort model based on abundance at age
// data. The model is described in Cadigan 2010 (Fish Ocean Canada) and 
// Cook 2013 (Fish Res). In contrast the aforeamentioned models this program
// implements mortality per age-class using a simplex vector describing how
// the average annual mortality is split between age-classes. This constraints
// the model.
//
// @ Torbjörn Säterberg(torbjorn.saterberg@slu.se)                    2024-06-27                    
//

data {
  int<lower=1> A; // number of age classes
  int<lower=1> Y; // number of years
  array[Y] int<lower=0, upper=1> Y_missing; // Indicator vector showing if data is missing for a given time index
  int <lower=0> n_missing; // number of years with no data
  array[Y] int<lower=0> E; // effort(i.e. number of net-nights per year)
  int<lower=1> n; // total sample number of data points
  array[n, A] int<lower=0> y; // survey indices for age a fish in year y
  array[A] real <lower=0>  Na1_init; // Initial abundance at time step 1
  real <lower=0> mean_log_N1; // initial recruitment value
  int<lower=0> N1_age; // age of initial age class
}

parameters {
  vector<lower=0>[Y] N1; // recruitment in other years (N1,y)
  vector<lower=0>[A] Na1; // abundances-at-age in year 1 (Na1)
  real mulogN1; // hyperparameter for the mean abundance of the youngest age-class
  real <lower=0> sdlogN1; // hyperparameter sd of the mean abundance of the youngest age-class 
  real <lower=0> phi; // scaling parameter(dispersion parameter) for the negative binomial distribution
  vector [A-1] logZ1; // vector with initial mortalities (at time t=1).
  real <lower=0, upper =1> phi_logZ_prior; // prior for AR1 logmortality innovations
  vector <lower = 0> [A-1] sd_logZ; // sd for logmortality innovations
  matrix [A-1, Y-1] z_logZ; // sd normal variates for non-centered parameterization of logZ process
}

transformed parameters {
matrix[A, Y] logN; // log abundance at age and year
matrix[A, Y-1] Z; // total mortality at age and year
matrix[A-1, A-1] cov_logZ; // covariance of logZ innovations
matrix[A-1, A-1] L_cov_logZ; // Cholesky factor of logZ innovation matrix
real phi_logZ; // AR1-coefficient of logZ-process
matrix[A, Y-1] logZ; // log of total mortality at age and year

// Transform AR(1) prior and define variance
phi_logZ = 2 * phi_logZ_prior - 1;      // maps [0,1] -> [-1,1] as phi_logZ_prior is Beta distributed i.e [0,1]

// Construct AR(1) covariance matrix across age
for (j in 1:(A-1)) {
  for (i in 1:(A-1)) {
    cov_logZ[i, j] = pow(phi_logZ, abs(i - j)) * sd_logZ[i] * sd_logZ[j];
  }
}
 // for (i in 1:(A-1)) {
 //   cov_logZ[i, i] += 1e-6;  // small jitter to make matrix PD
 // }
L_cov_logZ = cholesky_decompose(cov_logZ); // decompose covariance matrix

// initial value for logZ process
logZ[1:(A-1), 1] = logZ1;

// Random walk in time with AR(1) structure over age
for (t in 2:Y-1) {
  logZ[1:(A-1), t] = logZ[1:(A-1), t - 1] + L_cov_logZ * to_vector(z_logZ[1:(A-1), t - 1]);
}

// Assume that plusgroup mortality is the same as mortality for age-class below (A-1)
for (t in 1:(Y-1)) {
  logZ[A, t] = logZ[A-1, t];
}
// exponentiate mortalities
Z = exp(logZ);    
    
  // Initialize N1 (i.e recruitment to the gear)
    for(t in 1:Y){
    logN[1, t] = log(N1[t]);
    }
  
  // Initialize N-at-age for first year
    for(a in 2:A){
    logN[a, 1] = log(Na1[a]);
    }

  // Population dynamics
  for(t in 2:Y){
    for(a in 2:A) {
    if(a==A){ 
    logN[a, t] = log_sum_exp(
          logN[a-1, t-1] - Z[a-1, t-1],
          logN[a, t-1] - Z[a, t-1]  // if plus group add individuals staying in this group across years
        );
    }else{
    logN[a, t]=logN[a-1, t-1] -Z[a-1, t-1];
    }
    }
  }

// Check that matrix values are above machine precision  
  logN = fmax(logN, rep_matrix(-20, A, Y));

}

model {
  
//--------//
// Priors //
//--------//

// negative binomial dispersion parameter
phi ~ gamma(2, 1); 

// abundance of initial age-group   
mulogN1 ~ normal(mean_log_N1, 1); // hyperparameter
sdlogN1 ~ cauchy(0, 1); // hyperparameter

for (t in 1:Y) {
N1[t] ~ lognormal(mulogN1, sdlogN1); // group level parameter
}

// initial abundances year t=1
for (a in 1:A) {
Na1[a] ~ lognormal(log(Na1_init[a]), 1); // assume initial population size is related to initial abundance 
}

// initial mortality year t=1
logZ1 ~ normal(log(0.2),1);

// Standard normal for non-centered z
to_vector(z_logZ) ~ normal(0, 1);

// standard deviations for logZ random walk innovations
sd_logZ ~ normal(0, 1); // half-normal

// prior for AR1 coefficient of log mortality process
phi_logZ_prior ~ beta(2,2);

//------------//
// Likelihood //
//------------//
int pos; // initialize variable for ragged data structure
pos = 1; 
for (t in 1:Y) { // loop over year
if(Y_missing[t] == 0){
  for (a in 1:A) { // loop over age-groups
        segment(y[,a], pos, E[t]) ~ neg_binomial_2_log(logN[a, t], phi); // If you want to correct for mortality from beginning of year to time of survey   * exp(-t * Z[a, y])
  }
  pos += E[t];
}
}
}

generated quantities{
  vector<lower=0> [Y] R; // estimation of recruitment
  matrix[A, Y] N_obs; // simulated observation at age and year
  vector<lower=0> [Y] N_tot_obs; // Simulated observed total annual stock size 
  vector<lower=0> [Y] N_tot; // Simulated total annual stock size (deterministic process only)
  matrix[A, Y] N; // abundance at age and year
  array[n, A] int<lower=0> yrep; // simulated data for each observation
  array[n, A] real log_lik;
  vector [Y-1] Z_mean; // average annual mortality 
  real Z_mean_tot; // mean mortality for full time series
  vector [A] Z_hat; // mean mortality per age-class
  vector [A] Z_forecast; // 1-step-ahead forecast of mortality 
  vector [A] N_forecast; // 1-step-ahead forecast of Cohorts

  //---------------------------//
  // Estimate annual mortality //
  //---------------------------//
  for(a in 1:A){
  Z_hat[a] = mean(to_vector(Z[a,])); // mean age-class mortality
  }
  
  //---------------------------//
  // Estimate annual mortality //
  //---------------------------//
  for(t in 1:(Y-1)){
  Z_mean[t] = sum(Z[,t].*exp(logN[,t]))/sum(exp(logN[,t])); // abundance-weighted arithmetic mean
  }
  
  //-------------------------//
  // Estimate mean mortality //
  //-------------------------//
  for(t in 1:(Y-1)){
  Z_mean_tot = mean(Z_mean); // abundance-weighted arithmetic mean
  }

  
  //----------------------//
  // Estimate recruitment //
  //----------------------//
  R = rep_vector(0,Y); // initially set recruitment to zero for all cohorts
  if(N1_age==0){
      R=to_vector(exp(logN[1,]));
    }else{
  for(t in (N1_age+1):Y){
      R[t-N1_age]=exp(logN[1,t])*exp(sum(Z_mean[(t-N1_age):(t-1)]));  
    }
    }
    
  //------------------------------------------//
  // Population dynamics at observation level //
  //------------------------------------------//
  for(t in 1:Y) {
  for(a in 1:A){
    N_obs[a, t]=neg_binomial_2_log_rng(logN[a,t],phi);
  }
  }
  
  //-------------------------------------------//
  // Total stock dynamics at observation level //
  //-------------------------------------------//
  for(t in 1:Y) {
    N_tot_obs[t]=sum(N_obs[,t]);
  }
  
  //----------------------//
  // Total stock dynamics //
  //----------------------//
  for(t in 1:Y) {
    N_tot[t]=sum(exp(logN[,t]));
  }
  
  //--------------------------//
  // Stock size per age-class //
  //--------------------------//
  N=exp(logN); // 
  
  //---------------------------//
  // simulate each observation //
  //---------------------------//
  
 int pos; // initialize variable for ragged data structure
 pos = 1; 
  for (t in 1:Y) {
  if(Y_missing[t] == 0){
  for (a in 1:A) {
    for(i in pos:(pos+E[t]-1)){
        yrep[i, a] = neg_binomial_2_log_rng(logN[a, t], phi); 
    }
  }
  pos += E[t];
  }
  }
  
  //---------------//
  // Make forecast //
  //---------------//
  
  // Project mortality using correlated random walk process
  Z_forecast[1:(A-1)] = exp(multi_normal_cholesky_rng(logZ[1:(A-1), Y-1], L_cov_logZ));
  Z_forecast[A] = Z_forecast[A-1]; // assume plusgroup mortality is equal to the age-class below

  // 1-step-ahead forcast for cohorts
    for(a in 1:A) {
    if(a==1) {
    N_forecast[a]=0; // the first age-class can not be forecasted  
    } else if(a==A){ 
    // if plus group add individuals staying in this group across years
    N_forecast[a]=(exp(logN[a-1, Y])*exp(-Z_forecast[a-1]) + exp(logN[a, Y])*exp(-Z_forecast[a])); 
    }else{
    N_forecast[a]=exp(logN[a-1, Y])*exp(-Z_forecast[a-1]);
    }
    }
  
  //-----------------------------------------------------------------------//
  // calculated log-likelihood for each observation (for model comparison) //
  //-----------------------------------------------------------------------//
  int ind; // initialize variable for ragged data structure
  ind = 1; 
  
  for (t in 1:Y) {
  if(Y_missing[t] == 0){
  for (a in 1:A) {
   for(i in ind:(ind+E[t]-1)){
        log_lik[i, a] = neg_binomial_2_log_lpmf(y[i, a] | logN[a, t], phi); 
    }
  }
  ind += E[t];
  }
} 
}
