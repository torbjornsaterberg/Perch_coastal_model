
visualize_cohorts <- function(STAN_fit_summary, data_long, plottitle){
  
  # set wong-colour palette
  pal <- set_wong_colours()
  
  # first calculate median values per cohort of raw data  
  median_per_year_and_year_class <-     
    data_long %>% 
    group_by(year, year_class) %>% 
    summarize(median=median(n)) %>% # calculate median per year and year-class(this is what´s being predicted by the model) 
    ungroup() 
  
  nb_year_classes <- length(unique(data_long$year_class))
  
  # make a plot  
  cohort_plot <-
    ggplot(STAN_fit_summary$N, aes(x=year, y=median, group=as.factor(year_class), colour=as.factor(year_class))) +
    geom_point(data = median_per_year_and_year_class, aes(shape = as.factor(year_class))) +
    geom_line(aes(linetype = as.factor(year_class))) +
    scale_shape_manual(values = 1:nb_year_classes, name = "Observations(median)") +
    scale_linetype_manual(values = 1:nb_year_classes, name = "Year-class predictions") +
    #  scale_color_manual(pal) +
    #  scale_color_viridis_d(option = "magma", name = "Year Classes",guide="none") +
    labs(y = expression(tilde(y) ~ "[a, t] (Number of individuals caught per net-night)"),
         title = plottitle) +
    theme(
      legend.position = "right", # Position legend on the right
      legend.box = "vertical",  # Arrange legends vertically
      legend.spacing.y = unit(0.1, "cm"), # Increase spacing between legend elements
      legend.key.size = unit(0.75, "lines"),  # Reduce size of legend keys
      legend.text = element_text(size = 8) # Reduce font size of legend text
    ) +  
    guides(
      linetype = guide_legend(order = 1),
      shape = guide_legend(order = 2)
    )
  
  # print figure in QMD-file
  print(cohort_plot)
  
  # save plot
  ggsave(paste0("Perch_figures/", plottitle, "_cohort_plot.png"),
         plot=cohort_plot,
         device="png",
         width=14,
         height=14*9/16,
         units="cm",
         dpi=300)
}
