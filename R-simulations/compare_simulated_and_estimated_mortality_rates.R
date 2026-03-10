
compare_simulated_and_estimated_mortality_rates <- function(fit, sim_Z) {
  
  # Extract mortality estimates from STAN-fit object
  Z <- fit$summary("Z") %>% 
    mutate(age_class = str_extract(variable, "(?<=\\[)[0-9]+"), # Assign age_class to row index 
           time = str_extract(variable, "(?<=,)[0-9]+"))  # Assign years to column index
  
  sim_Z <- sim$Z
  sim_Z <- as_tibble(t(sim_Z)) %>% 
    rename_with(~ as.character(seq_along(.))) %>%
    add_column(time=rownames(.)) %>%
    pivot_longer(cols=!time, names_to="age_class", values_to="Z")
  
  Z <- Z %>% 
    left_join(sim_Z,join_by(age_class, time)) %>%
    mutate(time=as.numeric(time))
  
  # set wong-colours
  pal <- set_wong_colours()
  
  age_class_annual_mortality <-  
    ggplot(Z, aes(x = time, y = median)) +
    geom_line(colour=pal[1]) +
    geom_point(colour=pal[1]) +
    geom_ribbon(aes(ymin = q5, ymax = q95), alpha = 0.3, fill=pal[1]) +
    labs(
      x = "Year",
      y = expression(paste("Z"['a, t']," (Age-class mortality)"))
    ) +
    geom_point(aes(y=Z), colour="black") +
    facet_wrap(vars(age_class),scales="free_y")
  
  # print figure in QMD-file
  print(age_class_annual_mortality)
  
  # # save plot
  # ggsave(paste0("Perch_figures/", plottitle, "_age_class_annual_mortality.png"),
  #        plot=age_class_annual_mortality,
  #        device="png",
  #        width=14,
  #        height=14*9/16,
  #        units="cm",
  #        dpi=300)
}
