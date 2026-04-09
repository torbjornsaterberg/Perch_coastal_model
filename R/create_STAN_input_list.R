#----------------------------------------------#
# Create list with data for fitting STAN-model #
#----------------------------------------------#
#
# Outputs a data list with variables used in the STAN model.
#
create_STAN_input_list <- function(data){
  
  # number of age groups
  A <- length(grep("^\\d+$", names(data), value = TRUE)) + 1 # all numeric columnnames(age-classes) and the plus group
  
  # years in time series
  years <- min(data$year):max(data$year) # this includes also years with no data
  
  if(length(min(years):max(years))!=length(unique(data$year))){
    print("Be aware that there are years with missing data in the data set!!")
  }
  
  # number of years in time series
  Y <- length(years)
  
  # indicator variable showing which time indicies have no data
  Y_missing <- as.integer(years %in% setdiff(years,unique(data$year)))
  
  # number of years with missing data
  n_missing <- sum(Y_missing)
  
  # Effort(i.e. number of net-nights) 
  E <- (data %>% group_by(year) %>% summarize(eff=n()))
  
  E <- E %>% complete(year=years) %>% 
    mutate(eff=ifelse(is.na(eff),0,eff))
  
  E <- E$eff
  
  # number of caught individuals per net-night
  age_groups <- grep("^\\d+$", names(data), value = TRUE) # all numeric columns
  y <- data %>% select(all_of(age_groups),"plusgroup")
  
  # total number of observations
  n <- nrow(data)
  
  # initial age-group in data
  N1_age <- as.numeric(min(age_groups))
  
  # mean on log recruitment(number of individuals caught for the youngest age-class)
  mean_log_N1 <- data %>% 
    group_by(year) %>% 
    summarize(across(min(age_groups), mean, .names="init")) %>%
    ungroup() %>%
    select(init) %>%
    filter(init!=0) %>%
    summarize(mean_log_N1=mean(log(init)))
  
  mean_log_N1 <- mean_log_N1$mean_log_N1
  
  # prior mean for first year of observations
  Na1_init <- data %>%
    filter(year==min(year)) %>%
    select(-Lokal,-year,-OBS_ID) %>%
    summarize_all(mean) %>% # mean at age for first year or data(if zero assign small value)
    unlist()
  
  for(i in 1:length(Na1_init)){ # if some values zero set these to the mean value across the full data set
    if(Na1_init[i]==0){
      Na1_init[i] <- mean(unlist(data[,names(Na1_init[i])]))
    }  
  }
  
  
  
  
  STAN_input <- list(A=A, # number of age groups(including plusgroup)
                     Y=Y, # total number of years
                     Y_missing=Y_missing, # Indicator of years with missing data
                     n_missing=n_missing, # number of years with missing data
                     years=years, # years with data
                     E=E, # total effort per year
                     n=n, # total number of observations(i.e. net-nights)
                     N1_age=N1_age, # the youngst age-class considered for modelling
                     mean_log_N1=mean_log_N1, # mean number of individuals of the first age-class per net-night and year
                     Na1_init=Na1_init, # mean number of individuals caught per net-night the first survey year
                     y=as.matrix(y)) # number of individuals caught per age-class and net-night (ages-class in column and observation per row) 
  
  return(STAN_input)  
}
