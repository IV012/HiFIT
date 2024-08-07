#' Append an argument on a list
#'
#' Append an argument on a list of arguments
#'
#' @param args The original argument list.
#' @param argName The name of the argument to add.
#' @param argValue The value of the argument to add.
#' @param forced a \code{logical} value indicating if the argument with the same
#'  name already existed, whether it should be added forcedly.
#'
#' @return An argument list.
#'
#' @export
appendArg <- function(args, argName, argValue, forced) {
  
  if((!argName %in% names(args)) && (!forced)) {
    
    # cat("Default behavior: setting", argName, "to", argValue, "...\n")
    args[[argName]] <- argValue
  } else if(forced) {
    
    # cat("Forced behavior: setting", argName, "to", argValue, "(cannot override) ...\n")
    args[[argName]] <- argValue
  }
  
  return(args)
}

#' Remove an argument on a list
#'
#' Remove an argument on a list of arguments
#'
#' @param args The original argument list.
#' @param argName The name of the argument to be remmoved. If the name does not exist,
#'  no argument will be removed.
#'
#' @return An argument list.
#'
#' @export
removeArg <- function(args, argName) {
  
  if(argName %in% names(args))
    args[[which(names(args) == argName)]] <- NULL
  return(args)
}