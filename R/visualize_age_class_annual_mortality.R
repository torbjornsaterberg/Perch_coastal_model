visualize_age_class_annual_mortality <- function(STAN_fit_summary, plottitle) {
  
  # set wong-colours
  pal <- set_wong_colours()
  
  age_class_annual_mortality <-  
    ggplot(STAN_fit_summary$Z, aes(x = year, y = median)) +
    geom_line(colour=pal[1]) +
    geom_point(colour=pal[1]) +
    geom_ribbon(aes(ymin = q5, ymax = q95), alpha = 0.3, fill=pal[1]) +
    labs(
      x = "Year",
      y = expression(paste("Z"['a, t']," (Age-class mortality)")),
      title = plottitle
    ) +
    facet_wrap(vars(age_class),scales="free_y")
  
  # print figure in QMD-file
  print(age_class_annual_mortality)
  
  # save plot
  ggsave(paste0("Perch_figures/", plottitle, "_age_class_annual_mortality.png"),
         plot=age_class_annual_mortality,
         device="png",
         width=14,
         height=14*9/16,
         units="cm",
         dpi=300)
}
