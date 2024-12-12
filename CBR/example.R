

## Load CBR functions 
source('./functions/base_code_linear.R')
source('./functions/get_reg_init_linear.R')
source('./functions/bisection_mrl_linear.R')

## Load testing data
k <- 10   # Get true optimal treatment assignments under tau = 1
testing_list <- readRDS('./data/complex_testing_1.rds')
tau <- testing_list$O[[k]]$tau
A1_testing_opt <- testing_list$O[[k]]$A1_opt
A2_testing_opt <- testing_list$O[[k]]$A2_opt
H1 <- testing_list$H1
H2 <- testing_list$H2
H_testing_list <- list(H1 %>% scale(), H2 %>% scale())
A_testing_list <- list(testing_list$A1, testing_list$A2)
P_testing_list <- list(testing_list$P1, testing_list$P2)

# Set parallel setting
suppressPackageStartupMessages(library(doParallel))
numCores <- as.numeric(Sys.getenv("SLURM_NTASKS"))
if(Sys.getenv("SLURM_NTASKS") == ''){
  numCores = 1
}
registerDoParallel(cores=numCores)

## train Data setting 
epsilon <- 10^(-3)
eta <- 10^(-4)
mu2_list <- 10^(-8)
lambda_list <- list(2^(seq(-8,8,2)), 2^(seq(-8,8,2)))

# load train data
train_data <- readRDS('simulatedata.rds')
H_list=train_data$H_list
A_list=train_data$A_list
P_list=train_data$P_list
P=train_data$P
Reward=train_data$Reward
Risk=train_data$Risk

Output = foreach(i = 1:numCores, .combine = 'c') %dopar% {
  res_mrl <- try(br2_mrl_linear(H_list = H_list, A_list = A_list, P_list = P_list, P = P, 
                                  Reward = Reward, Risk = Risk, tau = tau, 
                                  mu2_list = mu2_list, lambda_list = lambda_list, 
                                  epsilon = epsilon, eta = eta))
  res <- res_mrl
  if(!is.character(res)){
  
    ## Get estimated treatment recommendations and testing reward/risk
    est <- est_opt(res$res, H_training = H_list, H_testing = H_testing_list, kernel = kernel, sigma_list = res$sigma_list)
    
    Reward_testing <- mean((testing_list$Y)*(est[[1]]==testing_list$A1&est[[2]]==testing_list$A2)/(0.25))
    Risk_testing   <- mean((testing_list$R)*(est[[1]]==testing_list$A1&est[[2]]==testing_list$A2)/(0.25))
    Efficacy_ratio <- (Reward_testing-1.200)/(Risk_testing-0.442)
  }else{
    Reward_testing <- NA
    Risk_testing <- NA
    Efficacy_ratio <- NA
  }
  
  out <- list(H_list = H_list, A_list = A_list, P_list = P_list,
            Reward = Reward, Risk = Risk, tau = tau,
            res = res, 
            Reward_testing = Reward_testing,
            Risk_testing = Risk_testing, 
            Efficacy_ratio = Efficacy_ratio)
  res_file_name <- sprintf('out.rds', i)
  saveRDS(out, res_file_name)
}

out <- readRDS('./out.rds')