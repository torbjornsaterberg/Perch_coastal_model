plot_stock_indicators <- function(STAN_fit_summary, data_long, plottitle){
  
  # set wong-colours
  pal <- set_wong_colours()
  
  tmin <- min(STAN_fit_summary$N_tot$year)
  tmax <- max(STAN_fit_summary$N_tot$year)
  
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
      y = expression(bar(N)[t]),
      title = plottitle
    ) +
    theme_bw() +
    theme(
      axis.text.x = element_blank(),
      axis.title.x = element_blank(),
      plot.margin = margin(0, 5, -5, 5) # Adjust margin to minimize space
    ) +
    annotate("text", x = -Inf, y = Inf, label = "(a)", size = 6, fontface = "bold", hjust = -0.1, vjust = 1.2) +
    xlim(c(tmin,tmax))
  
  # Mortality
  annual_mortality <- 
    ggplot(STAN_fit_summary$Z_mean, aes(x = year, y = median)) +
    geom_line(colour=pal[2]) +
    geom_point(colour=pal[2]) +
    geom_ribbon(aes(ymin = q5, ymax = q95), alpha = 0.3, fill=pal[2]) +
    labs(
      x = "Year",
      y = expression(bar(Z)[t]),
    )+
    theme_bw() +
    theme(
      axis.text.x = element_blank(),
      axis.title.x = element_blank(),
      plot.margin = margin(0, 5, -5, 5) # Adjust margin to minimize space
    ) +
    annotate("text", x = -Inf, y = Inf, label = "(b)", size = 6, fontface = "bold", hjust = -0.1, vjust = 1.2) +
    xlim(c(tmin,tmax))
  
  # Recruitment index
  recruitment_index <- 
    STAN_fit_summary$R %>%
    filter(!is.na(rhat)) %>%  
    ggplot(., aes(x = year, y = median)) +
    geom_line(colour = pal[3]) +
    geom_point(colour = pal[3]) +
    geom_ribbon(aes(ymin = q5, ymax = q95), alpha = 0.3, fill= pal[3]) +
    labs(
      y = expression(hat(R)[t])
    ) +
    scale_y_log10() +
    theme_bw() +
    theme(
      plot.margin = margin(0, 5, -5, 5) # Adjust margin to minimize space
    )+
    annotate("text", x = -Inf, y = Inf, label = "(c)", size = 6, fontface = "bold", hjust = -0.1, vjust = 1.2) +
    xlim(c(tmin,tmax))
  
  
  
  
  stock_indicators <- pred_and_obs_total_stock_size/annual_mortality/recruitment_index
  
  # print figure in QMD-file
  print(stock_indicators)
  
  # save plot
  # ggsave(paste0("Perch_figures/", plottitle, "_stock_indicators.png"),
  #            plot=stock_indicators,
  #            device="png",
  #            width=14,
  #            height=2*14*9/16,
  #            units="cm",
  #            dpi=300)
  
}
