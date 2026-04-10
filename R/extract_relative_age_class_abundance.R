extract_relative_age_class_abundance <- function(STAN_fit_summary, plottitle){
  data <- STAN_fit_summary$rel_N %>% 
    select(year, age_class, year_class, mean, q5, q95) %>%  # select parameters to include in output
    mutate(mean=round(mean,2), q5=round(q5,2), q95=round(q95,2)) %>% # round values
    add_column(Area=plottitle,.before="year")
  return(data)
}
