# Time series plot for total stock size
visualize_pred_and_obs_total_stock_size <- function(STAN_fit_summary, data_long, plottitle){
  
  # figure palette in wong colours  
  pal <- set_wong_colours()
  
  # Time series plot for total stock size
  pred_and_obs_total_stock_size <-
    data_long %>% 
    group_by(year,OBS_ID) %>% 
    summarize(n=sum(n)) %>% 
    ungroup() %>% 
    group_by(year) %>% 
    summarize(obs=mean(n)) %>%
    ungroup() %>% 
    ggplot(., aes(x=year, y=obs)) +
    geom_point() +
    geom_line(data = STAN_fit_summary$N_tot, aes(x = year, y = median), colour=pal[1]) +
    geom_point(data = STAN_fit_summary$N_tot, aes(x = year, y = median), colour=pal[1]) +
    geom_ribbon(data = STAN_fit_summary$N_tot,aes(x = year, y = median, ymin = q5, ymax = q95), alpha = 0.3, fill=pal[1]) +
    labs(
      x = "Year",
      y = expression(paste(N[tot], " (Total stock size)")),
      title = plottitle
    )
  
  # print figure in QMD-file
  print(pred_and_obs_total_stock_size)
  
  # save plot
  ggsave(paste0("Perch_figures/", plottitle, "_pred_and_obs_total_stock_size.png"),
         plot=pred_and_obs_total_stock_size,
         device="png",
         width=14,
         height=14*9/16,
         units="cm",
         dpi=300)
  
}
