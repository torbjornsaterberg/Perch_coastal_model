do_retrospective_analysis <- function(model, data_wide, n_peels){
  
  # inits  
  retro_data_list <- list()
  m_year <- max(data_wide$year)
  
  # Peel data sets
  for(k in 0:n_peels){
    STAN_input <- create_STAN_input_list(data_wide %>% filter(year<=(m_year-k))) # STAN_input list for one peel
    retro_data_list[[paste0("peel_", k)]] <- STAN_input
  }
  
  # Fit models to peeled data sets
  fit_list <- list()
  for (i in 0:n_peels) {
    data_i <- retro_data_list[[paste0("peel_", i)]]
    fit_list[[paste0("peel_", i)]] <- model$sample( # fit model to peeled data set
      data = data_i,
      chains = 4,
      parallel_chains = 4,
      iter_sampling = 1000,
      iter_warmup = 1000
    )
  }
  
  # re-arrange fitted data(add years etc.)
  retr_data <- tibble() # a tibble with retrospective fits etc.
  for(i in 0:n_peels){
    # append predictions
    output <- restruct_STAN_fit(fit_list[[paste0("peel_", i)]], retro_data_list[[paste0("peel_", i)]])
    tmp <- bind_rows(output$N_forecast, output$N, output$Z) %>% # temporary tibble with estimates of N, forecasts and Z
      add_column(peel=paste0("Peel",i)) # add a column defining the peel
    retr_data <- bind_rows(retr_data,tmp) # append tibble to a full output tibble
  }
  
  return(retr_data)
}