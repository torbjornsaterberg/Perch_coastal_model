#-----------------------------#
# Posterior predictive checks #
#-----------------------------#
# INPUTS
# input_data : A list with data according to the data structure in age_model_STAN.stan
# fit : A cmdstan model fit from the age_model_STAN.stan
# n_rep : number of simulated posterior samples to use in figures
#
# OUTPUT
# Posterior predictive checks checking:
# 1. Density plots (simulated and observed number of individuals caught per net) per age - group
# 2. Proportion of zeros, i.e. how many zeros are in simulated and observed data per age - group
create_PPC_plots <- function(input_data, fit, n_rep, plottitle){
  
  # inits  
  post <- head(posterior::as_draws_matrix(fit$draws()),n_rep) # reorder posterior for plotting
  A <- input_data$A # number of age-groups
  color_scheme_set(set_wong_colours()[1:6])
  bayesian_p_prop_zero <- data.frame(age_class=names(input_data$Na1_init),p_value=NA)
  
  
  for(i in 1:A){
    
    string <- paste0("^yrep.*,",i,"]$")
    col_ind <- grep(string, colnames(post), value=TRUE)
    
    # distribution of number of individuals caught and n_rep prediction of these
    p1 <- ppc_dens_overlay(y = input_data$y[,i],
                           yrep = post[,col_ind]) +
      labs(title = paste0("Density age-group ",i," ",plottitle))
    
    p2 <- ppc_stat(y=input_data$y[,i], 
                   yrep = post[,col_ind], 
                   stat = "prop_zero", 
                   binwidth = 0.005) +
      labs(title = paste0("Proportion zeros age-group ",i," ",plottitle))
    
    #calculate bayesian p-value for  proportion of  zeros
    p2_dat <- ppc_stat_data(y=input_data$y[,i], 
                            yrep = post[,col_ind], 
                            stat = "prop_zero")
    bayesian_p_prop_zero$p_value[i] <- mean((p2_dat %>% filter(variable=="y"))$value >= 
                                              (p2_dat %>% filter(variable!="y"))$value)
    
    # render figures in qmd    
    print(p1)
    print(p2)
    
    # save density plot
    ggsave(paste0("Perch_figures/", plottitle, "_PPC_density_age_class",i,".png"),
           plot=p1,
           width=14,
           height=14*9/16,
           units="cm",
           dpi=300)
    
    # save proportion of zeros PPC
    ggsave(paste0("Perch_figures/", plottitle, "_PPC_prop_zeros_age_class",i,".png"),
           plot=p2,
           device="png",
           width=14,
           height=14*9/16,
           units="cm",
           dpi=300)
    
  }
  
  ft_bayesian_p <- autofit(flextable(bayesian_p_prop_zero))
  save_as_docx(ft_bayesian_p, path=paste0("Perch_figures/", plottitle,"_bayesian_p_prop_zero.docx"))
  
}
