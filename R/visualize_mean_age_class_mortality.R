visualize_mean_age_class_mortality <- function(STAN_fit_summary, plottitle) {
  
  # set wong-colour palette
  pal <- set_wong_colours()
  
  mean_age_class_mortality <-
    ggplot(STAN_fit_summary$Z_hat) +
    geom_bar( aes(x = age_class, y = median), stat="identity", fill=pal[1], alpha=0.7) +
    geom_pointrange( aes(x = age_class, y = median, ymin = q5, ymax = q95), colour = "black") +
    xlab("Age-class") +
    ylab(expression("Mean age-class mortality " * (bar(Z)[a]))) +
    ggtitle(plottitle)  
  
  # print figure in QMD-file
  print(mean_age_class_mortality)
  
  # save plot
  ggsave(paste0("Perch_figures/", plottitle, "_mean_age_class_mortality.png"),
         plot=mean_age_class_mortality,
         device="png",
         width=14,
         height=14*9/16,
         units="cm",
         dpi=300)
}