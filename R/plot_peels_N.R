plot_peels_N <- function(retr_data, plottitle) {
  
  # set wong-colour palette
  pal <- set_wong_colours()
  
  retrospective_plot_N <-
    retr_data %>% 
    filter(str_starts(variable,"N"),
           !str_starts(variable,"N_for")) %>%
    ggplot(., aes(x=year,y=median,colour = peel)) +
    geom_line()+
    ylab("N")+
    ggtitle(plottitle) +
    facet_wrap(~age_class,scales="free_y")
  
  # print retrostpective analysis in QMD-file
  print(retrospective_plot_N)
  
  # save plot
  ggsave(paste0("Perch_figures/", plottitle, "_retrospective_plot_N.png"),
         plot=retrospective_plot_N,
         device="png",
         width=14,
         height=14*9/16,
         units="cm",
         dpi=300)
}
