calculate_prediction_accuracy <- function(data_long, retr_data) {
  
  MASE <- 
    data_long %>% 
    group_by(year,age_class) %>% # first calculate summary statistic for observations
    summarize(mean_CPUE=mean(n)) %>% # calculate mean cpue
    ungroup() %>%
    group_by(age_class) %>%
    mutate(lag_diff=abs(mean_CPUE-lag(mean_CPUE))) %>% # calculate absolute value of difference between CPUE(t)-CPUE(t-1)
    ungroup() %>%
    left_join( # Join forecasts
      retr_data %>% 
        filter(str_starts(variable, "N_fore"),
               peel!="Peel0"), # we don´t want forecasts for the full model(there is nothing to compare to)
      join_by(year,age_class)) %>%
    mutate(err=ifelse(median==0, NA, abs(median-mean_CPUE))) %>% # absolute forecast errors
    filter(str_starts(variable, "N_fore")) %>% # summarize only for cases with predictions
    group_by(age_class) %>%
    summarize(MASE=mean(err)/mean(lag_diff)) %>%
    ungroup()
  
  MASE <- autofit(flextable(MASE))
  save_as_docx(MASE, path=paste0("Perch_figures/", plottitle,"_MASE.docx"))
  return(MASE)
}
