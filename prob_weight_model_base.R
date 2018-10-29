rm(list=ls())  # remove all variables 

source("model_base.R")
library(data.table)

prob_weight <- model_base(
  task_name             = "",
  model_name            = "prob_weight",
  data_columns          = c("subjID","trial","opt1_h_prob","opt2_h_prob","opt1_h_val","opt1_l_val","opt2_h_val","opt2_l_val","choice"),
  parameters            = list("tau" = c(0, 0.9, 1),
                               "gamma" = c(0, 50, 100),
                               "lambda" = c(0,0.1,1),
                               "beta" = c(0, 85, 100)),
  regressors            = NULL,
  preprocess_function = function(raw_data, general_info) {
    DT      <- as.data.table(raw_data)
    subjs   <- general_info$subjs
    n_subj  <- general_info$n_subj
    t_subjs <- general_info$t_subjs
    t_max   <- general_info$t_max
    
    opt1_h_prob  <- array( 0, c(n_subj, t_max))
    opt2_h_prob  <- array( 0, c(n_subj, t_max))
    opt1_h_val  <- array( 0, c(n_subj, t_max))
    opt1_l_val  <- array( 0, c(n_subj, t_max))
    opt2_h_val  <- array( 0, c(n_subj, t_max))
    opt2_l_val  <- array( 0, c(n_subj, t_max))
    choice <- array( -1, c(n_subj, t_max))
    
    for (i in 1:n_subj) {
      subj <- subjs[i]
      t <- t_subjs[i]
      DT_subj <- DT[subjid == subj]
      opt1_h_prob <- opt1_h_prob
      opt2_h_prob <- opt2_h_prob
      opt1_h_val <- opt1_h_val
      opt1_l_val <- opt1_l_val
      opt2_h_val <- opt2_h_val
      opt2_l_val <- opt2_l_val
      choice[i, 1:t] <- DT_subj$choice
    }
    
    data_list <- list(
      N = n_subj,
      T = t_max,
      Tsubj = t_subjs,
      opt1_h_prob,
      opt2_h_prob,
      opt1_h_val,
      opt1_l_val,
      opt2_h_val,
      opt2_l_val,
      choice = choice,
    )
    
    return(data_list)
  }
)

prob_weight(data="example", niter=2000, nwarmup=1000, nchain=4, ncore=4)