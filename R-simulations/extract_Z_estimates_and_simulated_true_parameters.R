

extract_Z_estimates_and_simulated_true_parameters <- function(fit, sim_Z){

################################################################################
#-------------------------- Time varying mortality ----------------------------#
################################################################################
  
# Extract time varying mortality estimates from STAN-fit object
Z <- fit$summary(c("Z")) %>% 
    dplyr::select(c(variable, median, q5, q95)) %>% 
    mutate(age_class = str_extract(variable, "(?<=\\[)[0-9]+"), # Assign age_class to row index 
           time = str_extract(variable, "(?<=,)[0-9]+")) %>%  # Assign years to column index
    rename(Z_est = median, Z_est_q5 = q5, Z_est_q95 = q95) %>%
    add_column(Z_type="time_varying") %>%
    dplyr::select(Z_type, age_class, time, Z_est, Z_est_q5, Z_est_q95, -variable)
  
# Extract simulated time varying mortality rates    
sim_Z <- as_tibble(t(sim_Z)) %>% 
      rename_with(~ as.character(seq_along(.))) %>%
      add_column(time=rownames(.)) %>%
      pivot_longer(cols=!time, names_to="age_class", values_to="Z_true")
    
# Append true mortality values to estimates
Z <- Z %>% 
    left_join(sim_Z,join_by(age_class, time)) 
    

################################################################################
#--------------------------- Mean mortality -----------------------------------#
################################################################################

# Extract mean mortality estimates from STAN-fit object
Z_hat <- fit$summary("Z_hat") %>% 
      mutate(age_class = str_extract(variable, "(?<=\\[)[0-9]+")) %>%
      rename(Z_est=median, Z_est_q5 = q5, Z_est_q95 = q95) %>%
      add_column(Z_type="mean_mortality") %>%
      dplyr::select(Z_type, age_class, Z_est, Z_est_q5, Z_est_q95, -variable)

# Extract mean mortality rates from simulated data            
sim_Z_hat <- sim_Z %>% 
  group_by(age_class) %>% 
  summarize(Z_true = mean(Z_true))    

# Append true mortality values to estimates
Z_hat <- 
Z_hat %>% 
  left_join(sim_Z_hat,join_by(age_class))  


################################################################################
# --------------------- Merge mortality estimates -----------------------------#
################################################################################
Z <- bind_rows(Z, Z_hat)
Z <- Z %>% 
  mutate(bias = Z_est-Z_true,                                     # Bias - difference between estimated and observed value
         rel_bias = (Z_est-Z_true)/Z_true,                        # Relative bias - difference between estimated and observed value in relation to obs
         coverage = (Z_true > Z_est_q5) & (Z_true < Z_est_q95),   # Whether observed value is covered by 90% CI of posterior
         range = Z_est_q95 - Z_est_q5,                            # Posterior range q95-q5
         rel_range = (Z_est_q95 - Z_est_q5)/Z_est)                # Posterior relative range (q95-q5) / posterior median

return(Z)      
} 