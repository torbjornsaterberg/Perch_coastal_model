plot_peels_Z <- function(retr_data, plottitle) {
  
  # set wong-colour palette
  pal <- set_wong_colours()
  
  retrospective_plot_Z <- 
    retr_data %>% 
    filter(str_starts(variable,"Z")) %>%
    ggplot(., aes(x=year,y=median,colour = peel)) +
    geom_line() +
    scale_colour_manual(values = pal) +
    ylab("Z")+
    ggtitle(plottitle) +
    facet_wrap(~age_class,scales="free_y")
  
  # print retrostpective analysis in QMD-file
  print(retrospective_plot_Z)
  
  # save plot
  ggsave(paste0("Perch_figures/", plottitle, "_retrospective_plot_Z.png"),
         plot=retrospective_plot_Z,
         device="png",
         width=14,
         height=14*9/16,
         units="cm",
         dpi=300)
  
}
