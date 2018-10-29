data {
  int<lower=1> N;
  int<lower=1> T;               
  //int<lower=1, upper=T> Tsubj[N];                 
  int<lower=1,upper=4> choice[N,T];
  real<lower=1,upper=100> outcome[N,T];    
}

transformed data {
  real sigmaO_sq; // sigma_O = 4
  sigmaO_sq = 16;
}

parameters {
  // group-level parameters
  vector[6] mu_p;   
  vector<lower=0>[6] sd_p;

  // subject-level raw parameters, follows norm(0,1), for later Matt Trick
  vector[N] lambda_pr;    // decay factor
  vector[N] theta_pr;     // decay center
  vector[N] beta_pr;      // inverse softmax temperature 
  vector[N] mu0_pr;       // anticipated initial mean of all 4 options
  vector[N] sigma0_sq_pr; // anticipated initial sd^2 (uncertainty factor) of all 4 options  
  vector[N] sigmaD_sq_pr; // sd^2 of diffusion noise
}

transformed parameters {
  // subject-level parameters
  vector<lower=0,upper=1>[N] lambda;  
  vector<lower=0,upper=100>[N] theta;
  vector<lower=0,upper=1>[N] beta;   
  vector<lower=0,upper=100>[N] mu0;    
  vector<lower=0,upper=200>[N] sigma0_sq; 
  vector<lower=0,upper=200>[N] sigmaD_sq;
  
  // Matt Trick  
  for (i in 1:N) {
    lambda[i] = Phi_approx( mu_p[1] + sd_p[1] * lambda_pr[i] );
    theta[i]  = Phi_approx( mu_p[2] + sd_p[2] * theta_pr[i] ) * 100;
    beta[i]   = Phi_approx( mu_p[3] + sd_p[3] * beta_pr[i] ); 
    mu0[i]    = Phi_approx( mu_p[4] + sd_p[4] * mu0_pr[i] ) * 100;
    sigma0_sq[i] = Phi_approx( mu_p[5] + sd_p[5] * sigma0_sq_pr[i] ) * 200;
    sigmaD_sq[i] = Phi_approx( mu_p[6] + sd_p[6] * sigmaD_sq_pr[i] ) * 200;
  }
}

model {
  // prior: hyperparameters
  mu_p ~ normal(0,1);
  sd_p ~ cauchy(0,5);

  // prior: individual parameters
  lambda_pr  ~ normal(0,1);;   
  theta_pr   ~ normal(0,1);;    
  beta_pr    ~ normal(0,1);;     
  mu0_pr     ~ normal(0,1);;    
  sigma0_sq_pr ~ normal(0,1);; 
  sigmaD_sq_pr ~ normal(0,1);;  

  // subject loop and trial loop
  for (i in 1:N) {
    vector[4] mu_ev;    // estimated mean for each option
    vector[4] sd_ev_sq; // estimated sd^2 for each option
    real pe;            // prediction error
    real k;             // learning rate
    
    mu_ev    = rep_vector(mu0[i] ,4);
    sd_ev_sq = rep_vector(sigma0_sq[i], 4);

    for (t in 1:T) {
    //for (t in 1:(Tsubj[i])) {        
      // compute action probabilities
      choice[i,t] ~ categorical_logit( beta[i] * mu_ev );

      // learning rate
      k = sd_ev_sq[choice[i,t]] / ( sd_ev_sq[choice[i,t]] + sigmaO_sq );

      // prediction error 
      pe = outcome[i,t] - mu_ev[choice[i,t]];

      // value updating (learning) 
      mu_ev[choice[i,t]]    = mu_ev[choice[i,t]] + k * pe;
      sd_ev_sq[choice[i,t]] = (1-k) * sd_ev_sq[choice[i,t]];

      // diffusion process
      mu_ev    = lambda[i] * mu_ev + (1 - lambda[i]) * theta[i];
      sd_ev_sq = lambda[i]^2 * sd_ev_sq + sigmaD_sq[i];
    }
  }
}

generated quantities {
  real<lower=0,upper=1> mu_lambda; 
  real<lower=0,upper=100> mu_theta;
  real<lower=0,upper=1> mu_beta;
  real<lower=0,upper=100> mu_mu0;
  real<lower=0,upper=15> mu_sigma0;
  real<lower=0,upper=15> mu_sigmaD;
  real log_lik[N]; 

  mu_lambda = Phi_approx(mu_p[1]);
  mu_theta  = Phi_approx(mu_p[2]) * 100;
  mu_beta   = Phi_approx(mu_p[3]);
  mu_mu0    = Phi_approx(mu_p[4]) * 100;
  mu_sigma0 = sqrt(Phi_approx(mu_p[5]) * 200);
  mu_sigmaD = sqrt(Phi_approx(mu_p[6]) * 200);

  { // local block
    for (i in 1:N) {
      vector[4] mu_ev;    // estimated mean for each option
      vector[4] sd_ev_sq; // estimated sd^2 for each option
      real pe;            // prediction error
      real k;             // learning rate
      
      log_lik[i] = 0;
      mu_ev    = rep_vector(mu0[i] ,4);
      sd_ev_sq = rep_vector(sigma0_sq[i], 4);

      for (t in 1:T) {
      //for (t in 1:(Tsubj[i])) {        
        // compute action probabilities
        log_lik[i] = log_lik[i] + categorical_logit_lpmf( choice[i,t] | beta[i] * mu_ev );

        // learning rate
        k = sd_ev_sq[choice[i,t]] / ( sd_ev_sq[choice[i,t]] + sigmaO_sq );

        // prediction error 
        pe = outcome[i,t] - mu_ev[choice[i,t]];

        // value updating (learning) 
        mu_ev[choice[i,t]]    = mu_ev[choice[i,t]] + k * pe;
        sd_ev_sq[choice[i,t]] = (1-k) * sd_ev_sq[choice[i,t]];

        // diffusion process
        mu_ev    = lambda[i] * mu_ev + (1 - lambda[i]) * theta[i];
        sd_ev_sq = lambda[i]^2 * sd_ev_sq + sigmaD_sq[i];
      }
    }
  } // local block END
}
