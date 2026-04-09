visualize_recruitment_index <- function(STAN_fit_summary, plottitle) {
  
  # set wong-colour palette
  pal <- set_wong_colours()
  
  # Recruitment index
  recruitment_index <- 
    STAN_fit_summary$R %>%
    filter(!is.na(rhat)) %>%  
    ggplot(., aes(x = year, y = median)) +
    geom_line(colour = pal[1]) +
    geom_point(colour = pal[1]) +
    geom_ribbon(aes(ymin = q5, ymax = q95), alpha = 0.3, fill= pal[1]) +
    labs(
      x = "Year",
      y = expression(paste(R[year]," (Recruitment index)")),
      title = plottitle
    ) +
    scale_y_log10()
  
  # print figure in QMD-file
  print(recruitment_index)
  
  # save plot
  ggsave(paste0("Perch_figures/", plottitle, "_recruitment_index.png"),
         plot=recruitment_index,
         device="png",
         width=14,
         height=14*9/16,
         units="cm",
         dpi=300)
}
