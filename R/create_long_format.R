
#--------------------------#
# Long format for plotting #
#--------------------------#
#
# Restructure data into long format for plotting
#
create_long_format <- function(data){
  
  # age groups
  age_groups <- grep("^\\d+$", names(data), value = TRUE)
  
  # minimum age of plusgroup
  plusage <- max(as.numeric(age_groups)) + 1
  
  # age of first age-class
  N1_age <- min(as.numeric(age_groups))
  
  # columns to pivot
  cols=c(age_groups,"plusgroup") # we also want data for plusgroup
  
  # pivot data
  data_long <- data %>%
    pivot_longer(cols = all_of(cols), names_to = "age_class", values_to = "n") %>%
    mutate(year_class = case_when(age_class == "plusgroup" ~ year - plusage,
                                  age_class != "plusgroup" ~ year - as.numeric(age_class))) 
  
  return(data_long)
}
