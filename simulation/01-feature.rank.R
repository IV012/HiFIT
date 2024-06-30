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
result = matrix(NA, 6, 13)
colnames(result) = c(paste("V", seq(10), sep=""), "P", "SEED", "METHOD")
result <- as.data.frame(result)
result[,"P"] = p
result[,"SEED"] = seeds
result[, "METHOD"] <- c("Lasso","Pearson", "MIC","Spearman", "HSIC", "HFS")
model_settings <- list(
  list(generator=generator.lm, type="regression", file_path="./result/LR.rank.csv"),
  list(generator=generator.nonlinear, type="regression", file_path="./result/NLR.rank.csv"),
  list(generator=generator.glm, type="binary-classification", file_path="./result/GLM.rank.csv"),
  list(generator=generator.nglm, type="binary-classification", file_path="./result/NGLM.rank.csv")
)
settings <- model_settings[[genmod]]
generator <- settings$generator
type <- settings$type
file_path <- settings$file_path

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

# Lasso
family = ifelse(type=="regression", "gaussian", "binomial")
lasso_coef = coef(glmnet(trainx, trainy, alpha = 1, family=family, nlambda = 1000))
j = 0
for(i in seq(dim(lasso_coef)[2])){
  if(sum(lasso_coef[2:11, i] != 0) > j){
    k <- sum(lasso_coef[2:11, i] != 0)
    result[1, seq(j+1, k)] <- sum(lasso_coef[, i] != 0) - 1
    j <- k
    if(j == 10){
      break
    }
  }
}

# HFS
hfs.object <- hybrid.corr(trainx, trainy)
cutoff <- apply(hfs.object$corrs, 2, function(x) quantile(x, probs=.95, na.rm=TRUE))
cond <- apply(hfs.object$corrs, 1, function(x) TRUE %in% c(x>=cutoff))
if(all(cond[1:10])){
  result[6, 1:10] = sort(rank(-hfs.object$score[cond])[1:10])
}else{
  result[6, 1:10] = sort(rank(-hfs.object$score)[1:10])
}


# Pearson
pcc <- apply(trainx, 2, function(x){cor(x, trainy, method="pearson")})
result[2, 1:10] = sort(rank(-pcc)[1:10])

# MIC
mic <- apply(trainx, 2, function(x){mine(x, trainy)$MIC})
result[3, 1:10] = sort(rank(-mic)[1:10])

# Spearman
spc <- apply(trainx, 2, function(x){cor(x, trainy, method = "spearman")})
result[4, 1:10] = sort(rank(-spc)[1:10])

# HSIC
hsic <- apply(trainx, 2, function(x){dhsic(X=x, Y=trainy)$dHSIC})
result[5, 1:10] = sort(rank(-hsic)[1:10])
print(result)

# Check if the file exists
if (file.exists(file_path)) {
  existing_data <- read.csv(file_path)
  combined_data <- rbind(existing_data, as.data.frame(result))
  write.csv(combined_data, file_path, row.names = FALSE)
} else {
  write.csv(as.data.frame(result), file_path, row.names = FALSE)
}