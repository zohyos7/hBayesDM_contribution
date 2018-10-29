
setwd("/Users/zohyoonseo/Downloads")
load('fab_df_s10.RData')
data <- data.frame(matrix(ncol = 4, nrow = 0))
col <- c("subjID", "trial", "choice", "outcome")
colnames(data) <- col
for (i in 1:10) {
  for (n in 1:300){
    data[300*(i-1)+n,1] = i
    data[300*(i-1)+n,2] = n
    data[300*(i-1)+n,3] = D[['choice']][i,n]
    data[300*(i-1)+n,4] = D[['outcome']][i,n]
  }
}

write.table(data, file='example_data.tsv', quote=FALSE, sep='\t', col.names = NA)

kalmanfilter <- function(data           = "choose",
                         niter          = 3000,
                         nwarmup        = 1000,
                         nchain         = 4,
                         ncore          = 1,
                         nthin          = 1,
                         inits          = "random",
                         indPars        = "mean",
                         saveDir        = NULL,
                         modelRegressor = FALSE,
                         vb             = FALSE,
                         inc_postpred   = FALSE,
                         adapt_delta    = 0.95,
                         stepsize       = 1,
                         max_treedepth  = 10) {
  
  # Path to .stan model file
  if (modelRegressor) { # model regressors (for model-based neuroimaging, etc.)
    stop("** Model-based regressors are not available for this model **\n")
  }
  
  # To see how long computations take
  startTime <- Sys.time()
  
  # For using example data
  if (data == "example") {
    data <- "example_data.tsv"
  } else if (data == "choose") {
    data <- file.choose()
  }
  
  # Load data
  if (file.exists(data)) {
    rawdata <- read.table(data, header = T, sep = "\t")
  } else {
    stop("** The data file does not exist. Please check it again. **\n  e.g., data = '/MyFolder/SubFolder/dataFile.txt', ... **\n")
  }
  # Remove rows containing NAs
  NA_rows_all = which(is.na(rawdata), arr.ind = T)  # rows with NAs
  NA_rows = unique(NA_rows_all[, "row"])
  if (length(NA_rows) > 0) {
    rawdata = rawdata[-NA_rows,]
    cat("The number of rows with NAs = ", length(NA_rows), ". They are removed prior to modeling the data. \n", sep = "")
  }
  
  # Individual Subjects
  subjList <- unique(rawdata[,"subjID"])  # list of subjects x blocks
  numSubjs <- length(subjList)  # number of subjects
  
  # Specify the number of parameters and parameters of interest
  numPars <- 3
  POI     <- c('mu_lambda','mu_theta', 'mu_beta','mu_mu0','mu_sigma0','mu_sigmaD',
               'sd_p',
               'lambda',  'theta','beta','mu0', 'sigma0_sq','sigmaD_sq',
               'log_lik')
  
  if (inc_postpred) {
    POI <- c(POI, "y_pred")
  }
  
  modelName <- "kalmanfilter"
  
  # Information for user
  cat("\nModel name = ", modelName, "\n")
  cat("Data file  = ", data, "\n")
  cat("\nDetails:\n")
  if (vb) {
    cat(" # Using variational inference # \n")
  } else {
    cat(" # of chains                   = ", nchain, "\n")
    cat(" # of cores used               = ", ncore, "\n")
    cat(" # of MCMC samples (per chain) = ", niter, "\n")
    cat(" # of burn-in samples          = ", nwarmup, "\n")
  }
  cat(" # of subjects                 = ", numSubjs, "\n")
  
  ################################################################################
  # THE DATA.  ###################################################################
  ################################################################################
  
  Tsubj <- as.vector(rep(0, numSubjs)) # number of trials for each subject
  
  for (i in 1:numSubjs)  {
    curSubj  <- subjList[i]
    Tsubj[i] <- sum(rawdata$subjID == curSubj)  # Tsubj[N]
  }
  
  # Setting maxTrials
  maxTrials <- max(Tsubj)
  
  # Information for user continued
  cat(" # of (max) trials per subject = ", maxTrials, "\n\n")
  
  choice  <- array(-1, c(numSubjs, maxTrials))
  outcome <- array(0, c(numSubjs, maxTrials))
  
  for (i in 1:numSubjs) {
    curSubj      <- subjList[i]
    useTrials    <- Tsubj[i]
    tmp          <- subset(rawdata, rawdata$subjID == curSubj)
    choice[i, 1:useTrials] <- tmp$choice
    outcome[i, 1:useTrials] <- tmp$outcome
  }
  
  dataList <- list(
    N        = numSubjs,
    T        = maxTrials,
    Tsubj    = Tsubj,
    choice   = choice,
    outcome  = outcome,
    numPars  = numPars
  )
  
  # inits
  if (inits[1] != "random") {
    if (inits[1] == "fixed") {
      inits_fixed <- c(0.5, 1.0)
    } else {
      if (length(inits) == numPars) {
        inits_fixed <- inits
      } else {
        stop("Check your inital values!")
      }
    }
    genInitList <- function() {
      list(
        mu_p   = c(qnorm(inits_fixed[1]), qnorm(inits_fixed[2] / 5)),
        sigma  = c(1.0, 1.0),
        A_pr   = rep(qnorm(inits_fixed[1]), numSubjs),
        tau_pr = rep(qnorm(inits_fixed[2]/5), numSubjs)
      )
    }
  } else {
    genInitList <- "random"
  }
  
  if (ncore > 1) {
    numCores <- parallel::detectCores()
    if (numCores < ncore) {
      options(mc.cores = numCores)
      warning('Number of cores specified for parallel computing greater than number of locally available cores. Using all locally available cores.')
    }
    else{
      options(mc.cores = ncore)
    }
  } else {
    options(mc.cores = 1)
  }
  
  cat("***********************************\n")
  cat("**  Loading a precompiled model  **\n")
  cat("***********************************\n")
  
  # Fit the Stan model
  m = "kalmanfilter.stan"
  if (vb) {   # if variational Bayesian
    fit = rstan::vb(m,
                    data   = dataList,
                    pars   = POI,
                    init   = genInitList)
  } else {
    fit = rstan::sampling(m,
                          data   = dataList,
                          pars   = POI,
                          warmup = nwarmup,
                          init   = genInitList,
                          iter   = niter,
                          chains = nchain,
                          thin   = nthin,
                          control = list(adapt_delta   = adapt_delta,
                                         max_treedepth = max_treedepth,
                                         stepsize      = stepsize))
  }
  ## Extract parameters
  parVals <- rstan::extract(fit, permuted = T)
  if (inc_postpred) {
    parVals$y_pred[parVals$y_pred == -1] <- NA
  }
  
  A   <- parVals$A
  tau  <- parVals$tau
  
  # Individual parameters (e.g., individual posterior means)
  allIndPars <- array(NA, c(numSubjs, numPars))
  allIndPars <- as.data.frame(allIndPars)
  
  for (i in 1:numSubjs) {
    if (indPars == "mean") {
      allIndPars[i,] <- c(mean(A[, i]),
                          mean(tau[, i]))
    } else if (indPars == "median") {
      allIndPars[i,] <- c(median(A[, i]),
                          median(tau[, i]))
    } else if (indPars == "mode") {
      allIndPars[i,] <- c(estimate_mode(A[, i]),
                          estimate_mode(tau[, i]))
    }
  }
  
  allIndPars           <- cbind(allIndPars, subjList)
  colnames(allIndPars) <- c("A",
                            "tau",
                            "subjID")
  
  # Wrap up data into a list
  modelData        <- list(modelName, allIndPars, parVals, fit, rawdata)
  names(modelData) <- c("model", "allIndPars", "parVals", "fit", "rawdata")
  class(modelData) <- "hBayesDM"
  
  # Total time of computations
  endTime  <- Sys.time()
  timeTook <- endTime - startTime
  
  # If saveDir is specified, save modelData as a file. If not, don't save
  # Save each file with its model name and time stamp (date & time (hr & min))
  if (!is.null(saveDir)) {
    currTime  <- Sys.time()
    currDate  <- Sys.Date()
    currHr    <- substr(currTime, 12, 13)
    currMin   <- substr(currTime, 15, 16)
    timeStamp <- paste0(currDate, "_", currHr, "_", currMin)
    dataFileName = sub(pattern = "(.*)\\..*$", replacement = "\\1", basename(data))
    save(modelData, file = file.path(saveDir, paste0(modelName, "_", dataFileName, "_", timeStamp, ".RData")))
  }
  
  # Inform user of completion
  cat("\n************************************\n")
  cat("**** Model fitting is complete! ****\n")
  cat("************************************\n")
  
  return(modelData)
}
kalmanfilter("example")
