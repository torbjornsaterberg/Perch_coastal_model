
#----------------------------------------#
# Generate initial values for STAN-model #
#----------------------------------------#
#
# Outputs initial values, based on priors, for all parameters in model.
#
generate_inits_list <- function(data, chains = 4) {
  inits_list <- vector("list", chains)
  
  for (i in seq_len(chains)) {
    inits_list[[i]] <- list(
      phi = rgamma(1, 2, 1),
      mulogN1 = rnorm(1, data$mean_log_N1, 1),
      sdlogN1 = abs(rcauchy(1, 0, 1)),
      N1 = rlnorm(data$Y, data$mean_log_N1, 1),
      Na1 = rlnorm(data$A, log(data$Na1_init), 1),
      logZ1 = rnorm(data$A-1, log(0.2), 1),
      phi_logZ_prior = rbeta(1, 2, 2),
      sd_logZ = abs(rnorm(data$A-1, 0, 1)),
      z_logZ = array(rnorm((data$A-1) * (data$Y - 1), 0, 1), dim = c(data$A - 1, data$Y - 1))
    )
  }
  
  return(inits_list)
}
