visualize_annual_mortality <- function(STAN_fit_summary, plottitle) {
  
  # set wong-colours
  pal <- set_wong_colours()
  
  annual_mortality <- 
    ggplot(STAN_fit_summary$Z_mean, aes(x = year, y = median)) +
    geom_line(colour=pal[1]) +
    geom_point(colour=pal[1]) +
    geom_ribbon(aes(ymin = q5, ymax = q95), alpha = 0.3, fill=pal[1]) +
    labs(
      x = "Year",
      y = expression(bar(Z)[t]),
      title = plottitle
    ) 
  
  # print figure in QMD-file
  print(annual_mortality)
  
  # save plot
  ggsave(paste0("Perch_figures/", plottitle, "_annual_mortality.png"),
         plot=annual_mortality,
         device="png",
         width=14,
         height=14*9/16,
         units="cm",
         dpi=300)
  
}
