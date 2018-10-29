setwd("")
rm(list=ls())
install.packages("truncnorm")
library(truncnorm)

# Simulation parameters
seed <- 08826    # do not change the seed number!
num_subjs  <- 30 # number of subjects
num_trials <- 200 # number of trials per subject

# Set seed
set.seed(seed)   # always set a seed number for this homework!

# True parameters 
simul_pars <- data.frame(tau = rtruncnorm(num_subjs, a = 0, b = 1, mean = 0.85, sd = 0.08),
                         gamma = rtruncnorm(num_subjs, a = 0, b = 1, mean = 0.70, sd = 0.10),
                         lambda = rtruncnorm(num_subjs, a = 0, b = 5, mean = 2.5, sd = 0.30),
                         beta = rtruncnorm(num_subjs, a = 0, b = 1, mean = 0.20, sd = 0.08),
                         subjID  = 1:num_subjs)

# For storing simulated choice data for all subjects
all_data <- NULL

for (i in 1:num_subjs) {
  # Individual-level (i.e. per subject) parameter values
  tau <- simul_pars$tau[i]
  gamma <- simul_pars$gamma[i]
  lambda <- simul_pars$gamma[i]
  beta <- simul_pars$gamma[i]

  # For storing simulated data for current subject
  # subjID = subject ID
  # trial = trial number
  # choice = choice made on each trial (1 or 2)
  # outcome = outcome reveived on each trial (1 or -1)
  tmp_data = data.frame( subjID=NULL, 
                         trial=NULL, 
                         opt1_h_prob=NULL, 
                         opt2_h_prob=NULL,
                         opt1_h_val=NULL, 
                         opt1_l_val=NULL, 
                         opt2_h_val=NULL, 
                         opt2_l_val=NULL, 
                         choice=NULL)
  
  w_prob = c(0, 0, 0, 0)
  U_opt = c(0,0)
  
  for (t in 1:num_trials)  {
    opt1_h_prob = sample(1:9, 1)/10
    opt2_h_prob = sample(1:9, 1)/10
    opt1 = sample(-50:50, 2)
    opt1_h_val = max(opt1)
    opt1_l_val = min(opt1)
    opt2 = sample(-50:50, 2)
    opt2_h_val = max(opt2)
    opt2_l_val = min(opt2)
    
    w_prob = c(0, 0, 0, 0)
    U_opt = c(0,0)
    
    
    # Prob of choosing option 2
    w_prob[1] = exp(-(-log(opt1_h_prob))^tau)
    w_prob[2] = exp(-(-log(1-opt1_h_prob))^tau)
    w_prob[3] = exp(-(-log(opt2_h_prob))^tau)
    w_prob[4] = exp(-(-log(1-opt2_h_prob))^tau)
    
    if (opt1_h_val > 0) { 
      if (opt1_l_val >= 0) {
        U_opt[1]  = w_prob[1]*(opt1_h_val^gamma) + w_prob[2]*(opt1_l_val^gamma);
      } else {
        U_opt[1] = w_prob[1]*(opt1_h_val^gamma) - w_prob[2]*(abs(opt1_l_val)^gamma)*lambda;
      }
    } else {
      U_opt[1] = -w_prob[1]*(abs(opt1_h_val)^gamma)*lambda - w_prob[2]*(abs(opt1_l_val)^gamma)*lambda
    }
    
    if (opt2_h_val > 0) { 
      if (opt2_l_val >= 0) {
        U_opt[2]  = w_prob[3]*(opt2_h_val^gamma) + w_prob[4]*(opt2_l_val^gamma)
      } else {
        U_opt[2] = w_prob[3]*(opt2_h_val^gamma) - w_prob[4]*(abs(opt2_l_val)^gamma)*lambda
      }
    } else {
      U_opt[2] = -w_prob[3]*(abs(opt2_h_val)^gamma)*lambda - w_prob[4]*(abs(opt2_l_val)^gamma)*lambda
    }
    
    prob_choose_opt2 = 1 / (1 + exp(beta*(U_opt[1]-U_opt[2])))
    choice = rbinom(size=1, n = 1, prob = prob_choose_opt2)
    choice = choice + 1
    
    # append simulated task/response to subject data
    tmp_data[t, "subjID"] = i
    tmp_data[t, "trial"] = t
    tmp_data[t, "opt1_h_prob"] = opt1_h_prob
    tmp_data[t, "opt2_h_prob"] = opt2_h_prob
    tmp_data[t, "opt1_h_val"] = opt1_h_val
    tmp_data[t, "opt1_l_val"] = opt1_l_val
    tmp_data[t, "opt2_h_val"] = opt2_h_val
    tmp_data[t, "opt2_l_val"] = opt2_l_val
    tmp_data[t, "choice"] = choice
    
  } # end of t loop
  # Append current subject with all subjects' data
  all_data = rbind(all_data, tmp_data)
}

all_data

# Write out data
write.table(all_data, file = "probability_weight_simul_data.txt", row.names = F, col.names = T, sep = "\t")
save(simul_pars, file = "/home/zohyos7/hBayesDM_contribution/probability_weight_simul_param.Rdata")
