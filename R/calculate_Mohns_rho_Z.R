calculate_Mohns_rho_Z <- function(retr_data,n_peels){
  
  dat <- retr_data %>% 
    filter(str_starts(variable,"Z"))
  age_class <- unique(dat$age_class)
  max_year <- max(dat$year,na.rm=TRUE)
  rho <- rep(0,length(age_class))
  
  for (i in 1:n_peels){
    for (a in 1:length(age_class)){
      x_full <-
        dat %>%
        filter(age_class==age_class[a],
               year==max_year-i,
               peel=="Peel0") %>%
        select(median)
      
      x_retr <-
        dat %>%
        filter(age_class==age_class[a],
               year==max_year-i,
               peel==paste0("Peel",i)) %>%
        select(median)
      
      # check for cases where retrospective data does not exist
      if(is.numeric(x_retr$median) && length(x_retr$median) == 0){
        rho[a]<- NA
      } else{
        rho[a] <- rho[a] + (x_retr$median - x_full$median)/x_full$median
      }
    }  
  }
  rho <- round(rho/n_peels,5)
  rho <- data.frame(age_class, rho)
  rho <- autofit(flextable(rho))
  save_as_docx(rho, path=paste0("Perch_figures/", plottitle,"_Mohns_rho_Z.docx"))
  return(rho)
}
