#---------------------------#
# Restructure STAN fit data #
#---------------------------#
restruct_STAN_fit <- function(fit, STAN_input_list){
  
  # Some variables from STAN_input_list that are needed
  A <- STAN_input_list$A 
  N1_age <- STAN_input_list$N1_age # initial age-class considered
  age_class <- names(STAN_input_list$Na1_init) #character
  year <- STAN_input_list$years
  Y <- STAN_input_list$Y
  plusage <- A + N1_age - 1
  
  # N - expected number of individuals per age-class
  N <- fit$summary("N") %>% 
    mutate(age_class = factor(str_extract(variable, "(?<=\\[)[0-9]+"), 
                              levels = 1:A,        # Extract row index
                              labels = age_class), # Assign age_class to row index 
           year = factor(str_extract(variable, "(?<=,)[0-9]+"), 
                         levels = 1:Y,  # Extract column index
                         labels = year)  # Assign years to column index
    ) %>%
    mutate(age_class=as.character(age_class), # change from factor to character
           year=as.integer(as.character(year))) %>% # change from factor to character
    mutate(year_class=case_when(age_class=="plusgroup" ~ year - plusage, # Assign year class
                                age_class!="plusgroup" ~ year - as.numeric(age_class))) %>% 
    relocate(year,year_class,age_class,.after=variable) # rearrange order of columns
  
  
  # N - estimate of number of individuals per age-class
  N_obs <- fit$summary("N_obs") %>% 
    mutate(age_class = factor(str_extract(variable, "(?<=\\[)[0-9]+"), 
                              levels = 1:A,        # Extract row index
                              labels = age_class), # Assign age_class to row index 
           year = factor(str_extract(variable, "(?<=,)[0-9]+"), 
                         levels = 1:Y,  # Extract column index
                         labels = year)  # Assign years to column index
    ) %>%
    mutate(age_class=as.character(age_class), # change from factor to character
           year=as.integer(as.character(year))) %>% # change from factor to character
    mutate(year_class=case_when(age_class=="plusgroup" ~ year - plusage, # Assign year class
                                age_class!="plusgroup" ~ year - as.numeric(age_class))) %>% 
    relocate(year,year_class,age_class,.after=variable) # rearrange order of columns
  
  # N_tot - number of individuals per year
  N_tot <- bind_cols(as.data.frame(year), fit$summary("N_tot")) %>%
    relocate(year,.after=variable)
  
  # Recruitment indicies
  R <- bind_cols(as.data.frame(year), fit$summary("R")) %>%
    relocate(year,.after=variable) %>%
    add_column(min_age=N1_age,.after=year)
  
  # Annual mortality per age-class
  Z <- fit$summary("Z") %>% 
    mutate(age_class = factor(str_extract(variable, "(?<=\\[)[0-9]+"), 
                              levels = 1:A,        # Extract row index
                              labels = age_class), # Assign age_class to row index 
           year = factor(str_extract(variable, "(?<=,)[0-9]+"), 
                         levels = 1:(Y-1),  # Extract column index
                         labels = year[1:Y-1])  # Assign years to column index
    ) %>%
    mutate(age_class=as.character(age_class), # change from factor to character
           year=as.integer(as.character(year))) %>% 
    mutate(year_class=case_when(age_class=="plusgroup" ~ year - plusage, # Assign year class
                                age_class!="plusgroup" ~ year - as.numeric(age_class))) %>% 
    relocate(year,age_class,year_class, .after=variable) # rearrange order of columns
  
  # abundance-weighted arithmetic mean mortality
  Z_mean <- 
    bind_cols(data.frame(year=year[1:Y-1]), fit$summary("Z_mean")) %>%
    relocate(year,.after=variable)
  
  # Mean abundance-weighted arithmetic mean mortality across whole time series
  Z_mean_tot <- fit$summary("Z_mean_tot") %>%
    add_column(age_class="Total") %>%
    relocate(age_class, .after=variable)
  
  # Mean mortality per age-class
  Z_hat <-
    fit$summary("Z_hat") %>% mutate(age_class = factor(str_extract(variable, "(?<=\\[)[0-9]+"), 
                                                       levels = 1:A,        # Extract row index
                                                       labels = age_class)) %>%
    mutate(age_class=as.character(age_class)) %>% 
    relocate(age_class,.after=variable) # rearrange order of columns
  
  # 1-step_ahead forecast
  N_forecast <-
    fit$summary("N_forecast") %>% mutate(age_class = factor(str_extract(variable, "(?<=\\[)[0-9]+"), 
                                                            levels = 1:A,        # Extract row index
                                                            labels = age_class)) %>%
    mutate(age_class=as.character(age_class),
           year=as.integer(as.character(year[Y]))+1) %>% 
    relocate(year, age_class,.after=variable) # rearrange order of columns
  
  # put all parameters in a list
  STAN_output <- list(N=N, # Expected number of individuals observed per age-class
                      N_obs=N_obs, # Estimate of number of individuals observed per net-night
                      N_tot = N_tot, # Estimated stock size (deterministic prediction)
                      R=R, # Estimated recruitment for 0-year olds (Note!!! 0+ individuals in august)
                      Z=Z, # Mortality per age-class
                      Z_hat=Z_hat, # Mean mortality per age-class
                      Z_mean=Z_mean, # mean mortality per year
                      Z_mean_tot=Z_mean_tot, # mean mortality across whole time series
                      N_forecast=N_forecast) # 1-step-ahead forecast of N  
}
