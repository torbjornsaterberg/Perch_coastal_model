make_mean_age_class_mortality_table <- function(STAN_fit_summary, plottitle){
  data <- STAN_fit_summary$Z_hat %>% 
    bind_rows(STAN_fit_summary$Z_mean_tot) %>% # add mean total mortality
    select(age_class, median, q5, q95) %>%  # select parameters to include in output
    mutate(median=round(median,2), q5=round(q5,2), q95=round(q95,2)) %>% # round values
    add_column(Area=plottitle,.before="age_class")
  ft <- flextable(data)
  set_caption(ft,paste("Mean age-class mortality", plottitle))
}
