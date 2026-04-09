visualize_pred_and_obs_per_age_class <- function(STAN_fit_summary, data_long, plottitle){
  
  # wong colour palette  
  pal <- set_wong_colours()  
  
  # Time series plot for different age-classes
  pred_and_obs_per_age_class <-
    data_long %>% 
    group_by(year,age_class) %>% # first calculate summary statistic for observations
    summarize(q5=quantile(n,probs=0.05,type=3), # 5 percent quantile
              median=quantile(n,probs=0.5,type=3), # median
              q95=quantile(n,probs=0.95,type=3)) %>% # 95 percent quantile
    ungroup() %>%
    ggplot(.,aes(x=year-0.15, y=median, ymin=q5, ymax=q95)) + # plot median and quantiles for observations
    geom_pointrange(color=pal[1],linetype="dashed") +
    labs(
      x = "Year",
      y = expression(tilde(y) ~ "[a, t]")
    ) +
    ggtitle(plottitle) +  
    geom_pointrange(data=STAN_fit_summary$N_obs,color=pal[2],aes(x=year+0.15, y=median, ymin=q5, ymax=q95)) + # plot predictions 
    facet_wrap(vars(age_class),scales="free_y")
  
  # print figure in QMD-file
  print(pred_and_obs_per_age_class)
  
  # save plot
  ggsave(paste0("Perch_figures/", plottitle, "_pred_and_obs_per_age_class.png"),
         plot=pred_and_obs_per_age_class,
         device="png",
         width=14,
         height=14*9/16,
         units="cm",
         dpi=300)
  
}