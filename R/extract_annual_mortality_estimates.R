extract_annual_mortality_estimates <- function(STAN_fit_summary, plottitle){
  data <- STAN_fit_summary$Z_mean %>% 
    select(year, median, q5, q95) %>%  # select parameters to include in output
    mutate(median=round(median,2), q5=round(q5,2), q95=round(q95,2)) %>% # round values
    add_column(Area=plottitle,.before="year")
  return(data)
}
