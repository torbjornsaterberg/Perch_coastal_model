

simulate_perch_model <- function(sim_params){             

  # input parameters
  n_age_classes <- sim_params$n_age_classes                   # Number of age-classes in the model
  t_max <- sim_params$t_max                                   # Number of time steps to simulate
  noise_vs_sinus_signal <- sim_params$noise_vs_sinus_signal   # How lareg should noise be in comparison to sinus signal
  mu_log_recruitment <- sim_params$mu_log_recruitment         # Mu of recruitment on log-scale
  sd_log_recruitment <- sim_params$sd_log_recruitment         # sd of recruitment on log-scale
  recruitment <- sim_params$recruitment                       # Recruitment to the first age-class
  N_init <- sim_params$N_init                                 # Initial abundances at time 1
  NBphi <- sim_params$NBphi                                   # Dispersion parameter for neg-binomial distribution
  n_gillnet_nights <- sim_params$n_gillnet_nights             # Number of gillnet nights per year    
  age_class_specific_mortality <- sim_params$age_class_specific_mortality # logical whether different mortality rates per age-class or mortality assumed to be the same for all age-classes
  logZ_mean <- sim_params$logZ_mean                           # Average logZ per age-class
  Z_CV <- sim_params$Z_CV                                     # CV of mortality on ordinary scale
  
################################################################################
#------------------------- Simulate recruitment -------------------------------#
################################################################################
recruitment <- rlnorm(t_max, mu_log_recruitment, sd_log_recruitment)  

################################################################################
#-------------------    Create mortality vectors   ----------------------------#
################################################################################  
# Time vector
t <- seq(2, t_max)
  
# Sinus parameters
A <- 0.5      # amplitude
f <- 0.1      # frequency (cycles per time unit)
phi <- runif(1, 0, 2*pi)   # random phase shift

# input
sigma_log <- sqrt(log(1+Z_CV^2)) # convert to sigma on log-scale

if(age_class_specific_mortality){
# Sinusoidal curve
logZ_sinus <- matrix(rep(A * sin(2 * pi * f * t + phi),each=n_age_classes-1), n_age_classes-1, t_max-1)
logZ_sinus <- logZ_sinus - rowMeans(logZ_sinus) # center sinus curve
signal_sd <- sd(logZ_sinus[1,])                 # sd for sinus signal
noise_sd <- signal_sd*noise_vs_sinus_signal     # scale noise applied to mortality in relation to sinus sd 
logZ_noise <- matrix(rnorm((t_max-1)*(n_age_classes-1), sd = noise_sd), n_age_classes-1, t_max-1) # randomly draw noise
logZ_sinus_noise <- logZ_noise + logZ_sinus     # add noise to sinus signal
sd_mult_fact <-  sigma_log/apply(logZ_sinus_noise,1,sd) 
logZ_sinus_noise_resc <- sweep(logZ_sinus_noise,1,sd_mult_fact,"*")
logZ_mean <- matrix(logZ_mean[1:(n_age_classes-1)], n_age_classes-1, t_max-1)
logZ <- logZ_mean + logZ_sinus_noise_resc  
logZ <- rbind(logZ,logZ[n_age_classes-1,])
Z <- exp(logZ)
} else {
  # Sinusoidal curve
  logZ_sinus <- matrix(rep(A * sin(2 * pi * f * t + phi),each=n_age_classes-1), n_age_classes-1, t_max-1)
  logZ_sinus <- logZ_sinus - rowMeans(logZ_sinus) # center sinus curve
  signal_sd <- sd(logZ_sinus[1,])                 # sd for sinus signal
  noise_sd <- signal_sd*noise_vs_sinus_signal     # scale noise applied to mortality in relation to sinus sd 
  logZ_noise <- matrix(rep(rnorm((t_max-1), sd = noise_sd), each=n_age_classes-1),n_age_classes-1, t_max-1) # randomly draw noise
  logZ_sinus_noise <- logZ_noise + logZ_sinus     # add noise to sinus signal
  sd_mult_fact <-  sigma_log/apply(logZ_sinus_noise,1,sd) 
  logZ_sinus_noise_resc <- sweep(logZ_sinus_noise,1,sd_mult_fact,"*")
  logZ_mean <- matrix(logZ_mean, n_age_classes-1, t_max-1)
  logZ <- logZ_mean + logZ_sinus_noise_resc  
  logZ <- rbind(logZ,logZ[n_age_classes-1,])
  Z <- exp(logZ)
  
}
################################################################################
#------------------- Simulate population dynamics -----------------------------#
################################################################################
N <- matrix(NA, n_age_classes,t_max)
N[1,] <- recruitment
N[2:n_age_classes,1] <- N_init[2:n_age_classes] 
for(i in c(2:t_max)){
  for(j in c(2:n_age_classes)){
     if(j < n_age_classes) {
      N[j,i] = N[j-1, i-1] * exp(-Z[j-1, i-1]) # standard age-class equation 
            } else {
      N[j,i] = N[j-1, i-1] * exp(-Z[j-1, i-1]) + N[j, i-1] * exp(-Z[j, i-1]) # plus group
  }
}  
}

################################################################################
# ------------------------ Add observation noise ------------------------------#
################################################################################
y <- data.frame() 
for(i in c(1:t_max)){
  y_temp <- matrix(NA, n_gillnet_nights, n_age_classes)
  for(j in c(1:n_age_classes)){
    y_temp[,j] <- rnegbin(n_gillnet_nights, N[j, i], NBphi) 
  }
  y <- rbind(y, y_temp)
}

################################################################################
# ------------------------ Make a STAN input list -----------------------------#
################################################################################
STAN_input <- list(A = n_age_classes,             # Number of age-classes
                   Y = t_max,                     # Number of years
                   Y_missing = rep(0,t_max),      # Indicator of years with missing data
                   n_missing = 0,                 # Number of years with missing data
                   years = c(2007:2024),
                   E = rep(n_gillnet_nights, t_max),
                   n = n_gillnet_nights*t_max,
                   N1_age = 2,
                   mean_log_N1 = mean(log(N[1,])),
                   Na1_init = colMeans(y[1:n_gillnet_nights,]),
                   y = as.matrix(y)
                   )

sim <- list(STAN_input = STAN_input, Z=Z)

# return STAN input list
return(sim)
}


