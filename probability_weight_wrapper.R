rm(list=ls())  # remove all variables 

library(rstan)
data = read.table("probability_weight_simul_data.txt", header=T, sep="\t")


allSubjs = unique(data$subjID)  # all subject IDs
N = length(allSubjs)      # number of subjects
T = table(data$subjID)  # number of trials per subject (=140)
Tsubj = as.vector(T)
maxTrials <- max(Tsubj)
opt1_h_prob <- array(0, c(N, maxTrials))
opt2_h_prob <- array(0, c(N, maxTrials))
opt1_h_val <- array(0, c(N, maxTrials))
opt1_l_val <- array(0, c(N, maxTrials))
opt2_h_val <- array(0, c(N, maxTrials))
opt2_l_val <- array(0, c(N, maxTrials))
choice <- array(-1, c(N, maxTrials))



for ( i in 1:N) {
  curSubj      <- allSubjs[i]
  useTrials    <- Tsubj[i]
  tmp          <- subset(data, data$subjID == curSubj)
  opt1_h_prob[i, 1:useTrials] <- tmp$opt1_h_prob
  opt2_h_prob[i, 1:useTrials] <- tmp$opt2_h_prob
  opt1_h_val[i, 1:useTrials] <- tmp$opt1_h_val
  opt1_l_val[i, 1:useTrials] <- tmp$opt1_l_val
  opt2_h_val[i, 1:useTrials] <- tmp$opt2_h_val
  opt2_l_val[i, 1:useTrials] <- tmp$opt2_l_val
  choice[i, 1:useTrials] <- tmp$choice
}

dataList <- list(
  N       = N,
  T       = maxTrials,
  Tsubj   = Tsubj,
  opt1_h_prob = opt1_h_prob,
  opt2_h_prob = opt2_h_prob,
  opt1_h_val = opt1_h_val,
  opt1_l_val = opt1_l_val,
  opt2_h_val = opt2_h_val,
  opt2_l_val = opt2_l_val,
  choice = choice
  )

dataList

# run!
output = stan("probability_weight.stan", data = dataList, pars = c('mu_tau','mu_gamma', 'mu_lambda', 'mu_beta',
                                                                   'sigma',
                                                                   'tau',  'gamma','lambda','beta', 
                                                                   'log_lik'),
              iter = 2000, warmup=1000, chains=4, cores=4)

stan("probability_weight.stan", data = dataList, pars = c('mu_lambda','mu_gamma', 'mu_tau', 'mu_beta',
                                                          'sigma',
                                                          'lambda',  'gamma','beta','tau', 
                                                          'log_lik'),
     iter = 2000, warmup=1000, chains=4, cores=4)

traceplot(output, "mu_lambda")
traceplot(output, "mu_gamma")
traceplot(output, "mu_tau")
traceplot(output, "mu_beta")
traceplot(output, "lambda")
traceplot(output, "gamma")
traceplot(output, "beta")
traceplot(output, "tau")


parameters <- rstan::extract(output)


stan_plot(output,"sigma", show_density=T)
stan_plot(output, "lambda", show_density=T)
stan_plot(output, "gamma", show_density=T)
stan_plot(output, "beta", show_density=T)
stan_plot(output, "tau", show_density=T)

