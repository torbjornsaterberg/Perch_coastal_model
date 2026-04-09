#------------------------------------------------#
# Summarize data into age-groups and a plusgroup #
#------------------------------------------------#
# INPUTS
# min_age - minimum age of fish recruiting to the gear
# plus_age - The age above which few individuals reach
# area - the area of interest
# data - A tibble with columns areas, years and age-classes, and rows 
#        representing the number of individuals caught per net-night.
#
# OUTPUT
# dat - A tibble with similar structure as data, but old age-classes have been 
#       summarized into one age-class representing number of individuals above
#       a certain age caught per net-night, and only one area has been filtered 
#       out.

summarize_plusgroup <- function(min_age, plus_age, area, Depth_strata = NA, data) {
  
  if (sum(area %in% unique(data$Lokal))!=1){
    print("Error: Lokal(area) not in your data")
    return(NULL)
  }  
  
  # find maximum age(columnname) in data  
  max_age <-  max(as.numeric(
    grep("^\\d+$", names(data), value = TRUE)
  ))
  
  # Ages of plus group
  pgroup <- as.character(c(plus_age:max_age))
  
  # Ages of age groups considered
  age_groups <- as.character(c(min_age:(plus_age-1)))  
  
  # sum across all ages above plusgroup age
  if(is.na(Depth_strata)) {
    dat <- data %>%
      filter(Lokal==area) %>% # filter one area
      mutate(plusgroup=rowSums(across(matches(pgroup)))) %>% # sum across plusgroup ages
      select(Lokal, År, OBS_ID, all_of(age_groups), plusgroup) %>% # select some of the variables
      arrange(År) %>% # sort based on year
      rename(year=År) # rename to year
  } else {
    dat <- data %>%
      filter(Lokal==area, 
             Djupstratum==Depth_strata) %>% # filter one area
      mutate(plusgroup=rowSums(across(matches(pgroup)))) %>% # sum across plusgroup ages
      select(Lokal, År, OBS_ID, all_of(age_groups), plusgroup) %>% # select some of the variables
      arrange(År) %>% # sort based on year
      rename(year=År) # rename to year
  }
  return(dat)
}
