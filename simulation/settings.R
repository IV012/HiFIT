#### Linear Regression ####
generator.lm <- function(seed=1, p=c(500, 1000, 10000)[1]){
  set.seed(seed)
  z <- matrix(rnorm(500*p), 500, p)
  beta <- runif(10, 1, 1.5)
  
  return(list(
    z = z,
    y = z[, 1:10] %*% beta + rnorm(500),
    coef = beta
  ))
}

#### NonLinear Regression ####
generator.nonlinear <- function(seed=1, p=c(500, 1000, 10000)[1]){
  set.seed(seed)
  z <- matrix(rnorm(500*p), 500, p)
  
  return(list(
    z = z,
    y = 2*rowSums(sin(2*z[, 1:4])) - 2*rowSums(log(2*z[, 5:8]^2+1)) + z[, 9]*exp(z[, 10]) + rnorm(500)
  ))
}

#### Linear Logistic ####
generator.glm <- function(seed=1, p=c(500, 1000, 10000)[1]){
  set.seed(seed)
  z <- matrix(rnorm(500*p), 500, p)
  beta <- runif(10, 2, 3)
  pi <- exp(z[, 1:10] %*% beta) / (1 + exp(z[, 1:10] %*% beta))
  return(list(
    z = z,
    y = rbinom(500, 1, pi),
    coef = beta
  ))
}


#### Nonlinear logistic ####
generator.nglm <- function(seed=1, p=c(500, 1000, 10000)[1]){
  set.seed(seed)
  z <- matrix(rnorm(500*p), 500, p)
  mu <- 2*rowSums(sin(2*z[, 1:4])) - 2*rowSums(log(2*z[, 5:8]^2+1)) + z[, 9]*exp(z[, 10]) + 6.5
  mu <- 4*mu
  pi <- exp(mu) / (1 + exp(mu))
  return(list(
    z = z,
    y = rbinom(500, 1, pi),
    coef = beta
  ))
}