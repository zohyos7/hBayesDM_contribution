 
rm(list=ls())  # remove all variables 

source("model_base.R")


kalman_filter <- model_base(
  task_name             = "bandit4arm2",
  model_name            = "kalman_filter",
  data_columns          = c("subjID", "trial", "choice", "outcome"),
  parameters            = list("lambda" = c(0, 0.9, 1),
                               "theta" = c(0, 50, 100),
                               "beta" = c(0,0.1,1),
                               "mu0" = c(0, 85, 100),
                               "sigma0_sq" = c(0, 40, 200),
                               "sigmaD_sq" = c(0, 9, 200)),
  regressors            = NULL,
  preprocess_function = function(raw_data, general_info) {
    DT      <- as.data.table(raw_data)
    subjs   <- general_info$subjs
    n_subj  <- general_info$n_subj
    t_subjs <- general_info$t_subjs
    t_max   <- general_info$t_max
    
    choice  <- array( -1, c(n_subj, t_max))
    outcome <- array(  0, c(n_subj, t_max))
    
    for (i in 1:n_subj) {
      subj <- subjs[i]
      t <- t_subjs[i]
      DT_subj <- DT[subjid == subj]
      
      choice[i, 1:t] <- DT_subj$choice
      outcome[i, 1:t] <-DT_subj$outcome
    }
    
    data_list <- list(
      N = n_subj,
      T = t_max,
      Tsubj = t_subjs,
      choice = choice,
      outcome = outcome
    )
    
    return(data_list)
  }
)

kalman_filter(data="example", niter=2000, nwarmup=1000, nchain=4, ncore=4)
