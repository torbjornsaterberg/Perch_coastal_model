extract_recruitment_estimates <- function(STAN_fit_summary, plottitle){
  data <- STAN_fit_summary$R %>% 
    filter(median != 0) %>% # remove non existing rectruitment indices
    select(year, min_age, median, q5, q95) %>%  # select parameters to include in output
    mutate(median=round(median,2), q5=round(q5,2), q95=round(q95,2)) %>% # round values
    rename(Recruitment_year = year) %>%
    add_column(Area=plottitle,.before="Recruitment_year")
  return(data)
}
