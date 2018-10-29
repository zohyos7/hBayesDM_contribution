data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=1, upper=T> Tsubj[N];
  int<lower=-1, upper=2> choice[N, T];
  real<lower=0, upper=1> opt1_h_prob[N, T];
  real<lower=0, upper=1> opt2_h_prob[N, T];
  real opt1_h_val[N, T];
  real opt1_l_val[N, T];
  real opt2_h_val[N, T];
  real opt2_l_val[N, T];
}
transformed data {
}
parameters{
  //group-level parameters
  vector[4] mu_p;
  vector<lower=0>[4] sigma;
  
  //subject-level raw parameters, follows norm(0,1), for later Matt Trick
  vector[N] tau_p; //probability weight function
  vector[N] gamma_p; //subject utility fucntion
  vector[N] lambda_p; //loss aversion parameter 
  vector[N] beta_p; //inverse softmax temperature
}

transformed parameters {
  //subject-level parameters
  vector<lower=0, upper=1>[N] tau;
  vector<lower=0, upper=2>[N] gamma;
  vector<lower=0, upper=5>[N] lambda;
  vector<lower=0, upper=1>[N] beta;
  
  //Matt Trick
  for (i in 1:N) {
    tau[i] = Phi_approx( mu_p[1] + sigma[1] * tau_p[i] );
    gamma[i]  = Phi_approx( mu_p[2] + sigma[2] * gamma_p[i] )*2;
    lambda[i]   = Phi_approx( mu_p[3] + sigma[3] * lambda_p[i] )*5; 
    beta[i]    = Phi_approx( mu_p[4] + sigma[4] * beta_p[i] );
  }
}

model {
  //prior : hyperparameters
  mu_p ~ normal(0,1);
  sigma ~ cauchy(0,5);
  
  //prior : individual parameters
  tau_p ~ normal(0,1);
  gamma_p ~ normal(0,1);
  lambda_p ~ normal(0,1);
  beta_p ~ normal(0,1);
  
  //subject loop and trial loop
  for (i in 1:N) {
    for (t in 1:Tsubj[i]) {
      vector[4] w_prob;
      vector[2] U_opt;
      
      //probability weight function
      w_prob[1] = exp(-(-log(opt1_h_prob[i,t]))^tau[i]);
      w_prob[2] = exp(-(-log(1-opt1_h_prob[i,t]))^tau[i]);
      w_prob[3] = exp(-(-log(opt2_h_prob[i,t]))^tau[i]);
      w_prob[4] = exp(-(-log(1-opt2_h_prob[i,t]))^tau[i]);
      
      if (opt1_h_val[i,t]>0) { 
        if (opt1_l_val[i,t]>= 0) {
          U_opt[1]  = w_prob[1]*(opt1_h_val[i,t]^gamma[i]) + w_prob[2]*(opt1_l_val[i,t]^gamma[i]);
        } else {
          U_opt[1] = w_prob[1]*(opt1_h_val[i,t]^gamma[i]) - w_prob[2]*(fabs(opt1_l_val[i,t])^gamma[i])*lambda[i];
        }
      } else {
        U_opt[1] = -w_prob[1]*(fabs(opt1_h_val[i,t])^gamma[i])*lambda[i] - w_prob[2]*(fabs(opt1_l_val[i,t])^gamma[i])*lambda[i];
        }
        
      if (opt2_h_val[i,t] > 0) { 
        if (opt2_l_val[i,t] >= 0) {
          U_opt[2]  = w_prob[3]*(opt2_h_val[i,t]^gamma[i]) + w_prob[4]*(opt2_l_val[i,t]^gamma[i]);
        } else {
          U_opt[2] = w_prob[3]*(opt2_h_val[i,t]^gamma[i]) - w_prob[4]*(fabs(opt2_l_val[i,t])^gamma[i])*lambda[i];
        }
      } else {
        U_opt[2] = -w_prob[3]*(fabs(opt2_h_val[i,t])^gamma[i])*lambda[i] -w_prob[4]*(fabs(opt2_l_val[i,t])^gamma[i])*lambda[i];
        }
      
      choice[i, t] ~ categorical_logit(U_opt*beta[i]);
    }
  }
}

generated quantities {
  real<lower = 0, upper = 1> mu_tau;
  real<lower = 0, upper = 2> mu_gamma;
  real<lower = 0, upper = 5> mu_lambda;
  real<lower = 0, upper = 1> mu_beta;
  real log_lik[N];
  
  real y_pred[N,T];
  
  for (i in 1:N) {
    for (t in 1:T) {
      y_pred[i, t] = -1;
    }
  }
  
  mu_tau  = Phi_approx(mu_p[1]);
  mu_gamma  = Phi_approx(mu_p[2])*2;
  mu_lambda  = Phi_approx(mu_p[3])*5;
  mu_beta = Phi_approx(mu_p[4]);

  { // local section, this saves time and space
    for (i in 1:N) {
      log_lik[i] = 0;
    for (t in 1:Tsubj[i]) {
      vector[4] w_prob;
      vector[2] U_opt;
      
      //probability weight function
      w_prob[1] = exp(-(-log(opt1_h_prob[i,t]))^tau[i]);
      w_prob[2] = exp(-(-log(1-opt1_h_prob[i,t]))^tau[i]);
      w_prob[3] = exp(-(-log(opt2_h_prob[i,t]))^tau[i]);
      w_prob[4] = exp(-(-log(1-opt2_h_prob[i,t]))^tau[i]);

      if (opt1_h_val[i,t]>0) { 
        if (opt1_l_val[i,t]>= 0) {
          U_opt[1]  = w_prob[1]*(opt1_h_val[i,t]^gamma[i]) + w_prob[2]*(opt1_l_val[i,t]^gamma[i]);
        } else {
          U_opt[1] = w_prob[1]*(opt1_h_val[i,t]^gamma[i]) - w_prob[2]*(fabs(opt1_l_val[i,t])^gamma[i])*lambda[i];
        }
      } else {
        U_opt[1] = -w_prob[1]*(fabs(opt1_h_val[i,t])^gamma[i])*lambda[i] - w_prob[2]*(fabs(opt1_l_val[i,t])^gamma[i])*lambda[i];
        }
        
      if (opt2_h_val[i,t] > 0) { 
        if (opt2_l_val[i,t] >= 0) {
          U_opt[2]  = w_prob[3]*(opt2_h_val[i,t]^gamma[i]) + w_prob[4]*(opt2_l_val[i,t]^gamma[i]);
        } else {
          U_opt[2] = w_prob[3]*(opt2_h_val[i,t]^gamma[i]) - w_prob[4]*(fabs(opt2_l_val[i,t])^gamma[i])*lambda[i];
        }
      } else {
        U_opt[2] = -w_prob[3]*(fabs(opt2_h_val[i,t])^gamma[i])*lambda[i] -w_prob[4]*(fabs(opt2_l_val[i,t])^gamma[i])*lambda[i];
        }
       
      
      log_lik[i] = log_lik[i] + categorical_logit_lpmf(choice[i,t] | U_opt*beta[i]);
      y_pred[i, t]  = categorical_rng(softmax(U_opt*beta[i]));
      
      }
    }
  }
}
