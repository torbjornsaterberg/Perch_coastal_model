
visualize_expected_CPUE_per_age_class <- function(STAN_fit_summary, data_long, plottitle){
  
  # wong colour palette  
  pal <- set_wong_colours()  
  
  # Time series plot for different age-classes
  expected_CPUE_per_age_class <-
    data_long %>% 
    group_by(year,age_class) %>% # first calculate summary statistic for observations
    summarize(mean_CPUE=mean(n)) %>% # calculate mean cpue
    ungroup() %>%
    ggplot(.,aes(x=year, y=mean_CPUE)) + # plot median and quantiles for observations
    geom_point(color="black") +
    labs(
      x = "Year",
      y = expression(bold(N["a,t"]))
    ) +
    ggtitle(plottitle) +  
    geom_line(data = STAN_fit_summary$N, aes(x = year, y = median), colour=pal[1]) +
    geom_point(data = STAN_fit_summary$N, aes(x = year, y = median), colour=pal[1]) +
    geom_ribbon(data = STAN_fit_summary$N,aes(x = year, y = median, ymin = q5, ymax = q95), alpha = 0.3, fill=pal[1]) +
    facet_grid(vars(age_class),scales="free_y")
  
  
  # print figure in QMD-file
  print(expected_CPUE_per_age_class)
  
  # save plot
  ggsave(paste0("Perch_figures/", plottitle, "_expected_CPUE_per_age_class.png"),
         plot=expected_CPUE_per_age_class,
         device="png",
         width=14,
         height=14*9/16,
         units="cm",
         dpi=300)
}  
