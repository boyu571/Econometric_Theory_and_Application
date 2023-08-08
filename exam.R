install.packages('glmnet')
install.packages("stats")
install.packages('rstudioapi')



### 1?? ??   ### -> ??????  



#install.packages("glmnet")
library(haven)
library(lmtest)
library(plm)
library(tidyverse)
library(car)     
library(gplots)   
library(tseries)  
library(gmm)
library(sandwich)
library(readxl)
library(glmnet)
library(stats)



### 2?? ??   ###
rstudioapi::getActiveDocumentContext()$path
CURRENT_WORKING_DIR <- dirname(rstudioapi::getActiveDocumentContext()$path)
CURRENT_WORKING_DIR
file_path = paste0(CURRENT_WORKING_DIR, '/highdim.xlsx')

### DATA LOAD ###
data <- read_excel("C:/Users/boyu571/R/highdim.xlsx")
head(data)
y <- data$gdpgrowth
x <- subset(data, select=-gdpgrowth)

names(data)
### Ridge, Lasso Function ###
test <- function(select, mode, fold, post){
  country_data = dplyr::filter(data, country3letters == select)
  y_AUS <- data.matrix(country_data$gdpgrowth)
  x_AUS = data.matrix(subset(country_data, select= -c(gdpgrowth, country3letters, time0, country)))
  
  country_model <- glmnet::cv.glmnet(x_AUS, y_AUS, alpha = mode, nfolds=fold)
  summary(country_model)
  plot(log(country_model$lambda), country_model$cvm,
       xlab='log(lambda)',
       ylab='MSE',
       pch=21, col="blue", bg="lightblue", grid.col=NA)
  abline(v=log(country_model$lambda.min), col="red", )
  
  best_lambda <- country_model$lambda.min
  selected_coefficients <- coef(country_model, s = best_lambda)
  selected_coefficients
  final_result <- data.frame(which(selected_coefficients!=0, arr.ind=TRUE))
  final_result
  final_result$value <- selected_coefficients[selected_coefficients != 0]
  final_result <- subset(final_result, select=-row)
  final_result <- subset(final_result, select=-col)
  print(final_result)
  print(paste0("Total selected Value -> ", length(final_result[,1])-1))
  
  if (post ==TRUE && mode == 1){
    selected_coefficients
    selected_indices <- which(selected_coefficients != 0)
    selected_indices = selected_indices -1
    if (length(selected_indices) > 1){
      selected_predictors <- x_AUS[, selected_indices]
      selected_ols_model <- lm(y_AUS ~ selected_predictors)
      summary(selected_ols_model)
    }
    else{
      print("Selected variable dose not exist")
    }
  }  
  
}

set.seed(1234)
#### Function Parameter ####
### ?????? ?????? ?޸??? ###
select = 'CAN' ### AUS, CAN, FRA, USA, JPN
mode = 1 ## Ridge = 0. Lasso = 1
fold = 10 ## N-fold-validation
post = FALSE

#### Function start ####
test(select, mode, fold, post)






### 3?? ??   ###
# rstudioapi::getActiveDocumentContext()$path
# CURRENT_WORKING_DIR <- dirname(rstudioapi::getActiveDocumentContext()$path)
# CURRENT_WORKING_DIR
# file_path_2 = paste0(CURRENT_WORKING_DIR, '/gdpgrowth.csv')

### DATA LOAD ###
data_2 <- read.csv("C:/Users/boyu571/R/gdpgrowth.csv")
head(data_2)
data_2 = data.matrix(subset(data_2, select=-TIME))
#data_2$TIME = seq.int(nrow(data_2))

# PCA
pca_result <- prcomp(data_2, scale = TRUE)

# Extract the first five principal components
first_five_pcs <- pca_result$x[, 1:5]
first_five_pcs

# Draw the scree plot
scree_variances <- pca_result$sdev^2
percentage_var_explained <- scree_variances / sum(scree_variances) * 100

# Create a scree plot
plot(1:length(scree_variances), percentage_var_explained, type = "b", 
     xlab = "Principal Component", ylab = "Percentage of Variance Explained",
     main = "Scree Plot for PCA", col = "black", pch = 19)

# Add a horizontal line at 5 principal components
abline(h = 5, col = "red", lty = 2)

# Add labels to the points
text(1:length(scree_variances), percentage_var_explained, labels = sprintf("%.2f%%", percentage_var_explained), 
     pos = 3, cex = 0.7, col = "black")

