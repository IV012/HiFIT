library(glmnet)
library(tidyverse)
library(minerva)
source("./settings.R")
source("./HFS.R")
source("./permfit.R")
source("./classes.R")
source("./control.R")

#### Load simulation settings ####
# Please pass three arguments when running the simulation script
# 1: seeds - int, 1-100
# 2: p.set - int, 1-3 (set dim to 500, 1000, 10000)
# 3: genmod - int, 1-4 (LR, NLR, GLM, NGLM)
args <- commandArgs(trailingOnly = TRUE)
seeds <- as.numeric(args[1])
p.set <- as.numeric(args[2])
genmod <- as.numeric(args[3])
p = c(500, 1000, 10000)[p.set]
result = matrix(NA, 6, 5)
colnames(result) = c("TV", "AV", "P", "SEED", "METHOD")
result <- as.data.frame(result)
result[,"P"] = p
result[,"SEED"] = seeds
result[, "METHOD"] <- c("Lasso","Pearson", "MIC","Spearman", "HSIC", "HFS")
model_settings <- list(
  list(generator=generator.lm, type="regression", file_path="./result/LR.screen.csv"),
  list(generator=generator.nonlinear, type="regression", file_path="./result/NLR.screen.csv"),
  list(generator=generator.glm, type="binary-classification", file_path="./result/GLM.screen.csv"),
  list(generator=generator.nglm, type="binary-classification", file_path="./result/NGLM.screen.csv")
)
settings <- model_settings[[genmod]]
generator <- settings$generator
type <- settings$type
file_path <- settings$file_path

#### Other settings of the hyperparameters ####
pvacut = 0.1
n_perm = 100
n_ensemble = 100 #100
n_epoch = 1000 #1000
n_tree = 1000
node_size = 3
esCtrl = list(
  n.hidden = c(50, 40, 30, 20), 
  activate = "relu", 
  l1.reg = 10**-4, 
  early.stop.det = 1000, 
  n.batch = 50, 
  n.epoch = n_epoch, 
  learning.rate.adaptive = "adam", 
  plot = FALSE)

mod_args = list(
  method = "ensemble_dnnet",
  n.ensemble = n_ensemble,
  esCtrl = esCtrl,
  verbose = 0
)

#### split training and fitting ####
seed = 6012023 * seeds
set.seed(seed)
validate = sample(1:500, 50)
df = generator(seed, p)
x = df$z
y = df$y
trainx = x[-validate,]
trainy = y[-validate]
testx = x[validate,]
testy = y[validate]

# HFS
val.idx <- sample(1:450, 50)
hfs.object <- hybrid.corr(trainx[-val.idx, ], trainy[-val.idx])
tau.object <- do.call(
  tune.tau,
  c(list(
    X = trainx,
    y = trainy,
    hfs.object = hfs.object,
    val.idx = val.idx,
    tune.type = "group-permfit"
  ), mod_args)
)
idx <- tau.object$idx
tau.best <- tau.object$tau.best
result[6, c(1, 2)] <- c(sum(idx %in% c(1:10)), length(idx))
q1 <- length(idx)

# Lasso
family = ifelse(type=="regression", "gaussian", "binomial")
lasso_coef = coef(glmnet(trainx, trainy, alpha = 1, nlambda = 1000, family=family))
for(i in seq(dim(lasso_coef)[2])){
  if(sum(lasso_coef[, i]!=0) >= q1 + 1){
    result[1, 1] = sum(which(lasso_coef[, i]!=0) %in% c(1:10))
    result[1, 2] = q1
    break
  }
}

# Pearson
pcc <- apply(trainx, 2, function(x){cor(x, trainy, method="pearson")})
result[2, c(1, 2)] = c(sum(which(rank(-pcc) <= q1) %in% c(1:10)), q1)

# MIC
mic <- apply(trainx, 2, function(x){mine(x, trainy)$MIC})
result[3, c(1, 2)] = c(sum(which(rank(-mic) <= q1) %in% c(1:10)), q1)

# Spearman
spc <- apply(trainx, 2, function(x){cor(x, trainy, method = "spearman")})
result[4, c(1, 2)] = c(sum(which(rank(-spc) <= q1) %in% c(1:10)), q1)

# HSIC
hsic <- apply(trainx, 2, function(x){dhsic(X=x, Y=trainy)$dHSIC})
result[5, c(1, 2)] = c(sum(which(rank(-hsic) <= q1) %in% c(1:10)), q1)
print(result)

# Check if the file exists
if (file.exists(file_path)) {
  existing_data <- read.csv(file_path)
  combined_data <- rbind(existing_data, as.data.frame(result))
  write.csv(combined_data, file_path, row.names = FALSE)
} else {
  write.csv(as.data.frame(result), file_path, row.names = FALSE)
}