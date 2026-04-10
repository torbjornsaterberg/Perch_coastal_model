  visualize_relative_age_class_abundance <- function(STAN_fit_summary, plottitle) {
    

    # Restructure data somewhat (colours should be the same for the same age-class across areas)
    dat <- STAN_fit_summary$rel_N
    dat <- 
    dat %>%
      mutate(max_age = max(as.numeric(age_class[age_class != "plusgroup"]), na.rm = TRUE),
        age_class = case_when(age_class=="plusgroup" ~ as.numeric(max_age) + 1,
                                   .default=as.numeric(age_class))) %>%
      select(-max_age) %>%
      mutate(age_class=factor(age_class,levels=rev(unique(age_class))))
  
    # define colours
    pal <- set_wong_colours()[as.numeric(levels(dat$age_class))]    
    
   # make plot
      ggplot(dat, aes(x = year, y = mean, fill=age_class)) +
      geom_area() +
          labs(
        x = "Year",
        y = expression(frac(hat(N)[a,t], sum(hat(N)[i*","*t])) ~ "(Relative age-class abundance)"),
        title = plottitle
      ) +
      scale_fill_manual(values=pal) +
      theme_bw()
    
    
  }
