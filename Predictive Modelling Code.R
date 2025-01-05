rm(list = ls())
# Import packages
library(ISLR) 
library(ROCR) # for constructing ROC curves
library(kknn) # allows us to do KNN for regression and classification
install.packages("tree")
install.packages("rpart")
library(tree)
library(MASS)
library(rpart)
library(ROCR)
library(pls)
library(caret)
install.packages("corrplot")
library(corrplot)
library(ggplot2)

# Load dataset and set seed
house = HDB_data_2021_sample
house$resale_price <- house$resale_price / 1000

### Data visualisations - Plotting charts and box plots
par(mfrow= c(1,1))
## Summary of data
summary(house)

## Checking correlation values with the output variable
output_variable <- house$resale_price
# Extract predictor variables (excluding the output variable)
predictor_variables <- house[, !(names(house) %in% "resale_price")]
# Calculate correlation coefficients
correlations <- sapply(predictor_variables, function(var) cor(output_variable, var))
# Create a data frame with variable names and correlation coefficients
correlation_data <- data.frame(Variable = names(correlations), Correlation = correlations)
# Order the data frame by absolute correlation values
correlation_data <- correlation_data[order(abs(correlation_data$Correlation), decreasing = TRUE),]
# Print or inspect the sorted correlation data
print(correlation_data)

# Positive correlation: floor_area_sqm (0.63), max_floor_lvl (0.51), Remaining_lease (0.35), flat_type_EXECUTIVE, X5room_sold, flat_type_5.ROOM, exec_sold, matyre, 
# Negative correlation: flat_type_3.ROOM, X3room_sold, flat_model_new.generation, Dist_CBD, Dist_nearest_University

## Scatter Plot 
plot(house$floor_area_sqm, house$resale_price,
     xlab = "floor area (sqm)", ylab = "Resale Price ($1000)",
     pch = 19, col = "Red")
p1 = lm(resale_price ~ floor_area_sqm , data = house)
abline(p1, col = "blue", lwd = 3)
adj_r_squared <- summary(p1)$adj.r.squared
title(paste("Adjusted R-squared =", round(adj_r_squared, 4)),cex.main = 0.9,line=0.5)
summary(p1) # adj R2: 0.398

plot(house$max_floor_lvl, house$resale_price,
     xlab = "max_floor_lvl", ylab = "Resale Price ($1000)",
     pch = 19, col = "Red")
p2 = lm(resale_price ~ max_floor_lvl, data = house)
abline(p2, col = "blue", lwd = 3)
adj_r_squared <- summary(p2)$adj.r.squared
title(paste("Adjusted R-squared =", round(adj_r_squared, 4)),cex.main = 0.9,line=0.5)
summary(p2) # adj R2: 0.2571

plot(house$Remaining_lease, house$resale_price,
     xlab = "Remaining_lease", ylab = "Resale Price ($1000)",
     pch = 19, col = "Red")
p3 = lm(resale_price ~ Remaining_lease , data = house)
abline(p3, col = "blue", lwd = 3)
adj_r_squared <- summary(p3)$adj.r.squared
title(paste("Adjusted R-squared =", round(adj_r_squared, 4)),cex.main = 0.9,line=0.5)
summary(p3) # adj R2: 0.123

plot(house$Dist_nearest_station, house$resale_price,
     xlab = "nearest stn (km)", ylab = "Resale Price ($1000)",
     pch = 19, col = "Red")
p4 = lm(resale_price ~ Dist_nearest_station , data = house)
abline(p4, col = "blue", lwd = 3)
adj_r_squared <- summary(p4)$adj.r.squared
title(paste("Adjusted R-squared =", round(adj_r_squared, 4)),cex.main = 0.9,line=0.5)
summary(p4) # adj R2: 0.011

plot(house$Dist_CBD, house$resale_price,
     xlab = "dist from cbd (km)", ylab = "Resale Price ($1000)",
     pch = 19, col = "Red")
p5 = lm(resale_price ~ Dist_CBD, data = house)
abline(p5, col = "blue", lwd = 3)
adj_r_squared <- summary(p5)$adj.r.squared
title(paste("Adjusted R-squared =", round(adj_r_squared, 4)),cex.main = 0.9,line=0.5)
summary(p5) # adj R2: 0.07235

plot(house$Dist_nearest_hospital, house$resale_price,
     xlab = "nearest hosp (km)", ylab = "Resale Price ($1000)",
     pch = 19, col = "Red")
p6 = lm(resale_price ~ Dist_nearest_hospital, data = house)
abline(p6, col = "blue", lwd = 3)
adj_r_squared <- summary(p6)$adj.r.squared
title(paste("Adjusted R-squared =", round(adj_r_squared, 4)),cex.main = 0.9,line=0.5)
summary(p6) #adj R2: 0.01392

plot(house$Dist_nearest_primary_school, house$resale_price,
     xlab = "nearest pri school (km)", ylab = "Resale Price ($1000)",
     pch = 19, col = "Red")
p7 = lm(resale_price ~ Dist_nearest_primary_school, data = house)
adj_r_squared <- summary(p7)$adj.r.squared
title(paste("Adjusted R-squared =", round(adj_r_squared, 4)),cex.main = 0.9,line=0.5)
abline(p7, col = "blue", lwd = 3)
summary(p7) #adj R2: -0.000137


## Box Plots

# Maturity of estate
boxplot(house$resale_price ~ house$mature,
        xlab = 'Mature estate',
        ylab = 'Resale price ($1000)')

library(ggplot2)

# Flat type box plots 
house$room_type <- factor(
  apply(house[, c('flat_type_1.ROOM','flat_type_2.ROOM', 'flat_type_3.ROOM', 'flat_type_4.ROOM', 'flat_type_5.ROOM','flat_type_EXECUTIVE','flat_type_MULTI.GENERATION')], 1, which.max),
  labels = c('1 room','2 room', '3 room','4 room', '5 room', 'exec','multigen')
)

ggplot(house, aes(x = room_type, y = resale_price, fill = room_type)) +
  geom_boxplot() +
  labs(title = "Distribution of Prices by Flat type", x = "Flat Type", y = "Resale Price ($1000)")

# MRT box plots
house$MRT <- factor(
    apply(house[, c('NSL','EWL', 'NEL', 'CCL', 'DTL','TEL','LRT')], 1, which.max), 
    labels = c('nsl','ewl', 'nel','ccl', 'dtl', 'tel','lrt')
)
ggplot(house, aes(x = MRT, y = resale_price, fill = MRT)) +
  geom_boxplot() +
  labs(title = "Distribution of Prices by MRT", x = "MRT", y = "Resale Price") 

## Multiple linear regression
set.seed(1101) 
ntrain=4800
tr = sample(1:nrow(house),ntrain)  
train = house[tr,]   
test = house[-tr,] 


# All predictors
housemlr = lm(formula = resale_price ~  ., data = train)
predmlr = predict.lm(housemlr, newdata=test)
mlrrmse = sqrt(mean((test$resale_price-predmlr)^2))
summary(housemlr)
# RMSE: 46.29, adj R2: 0.9295

# Specified predictors
housemlr1 = lm(formula = resale_price ~ Remaining_lease + floor_area_sqm + max_floor_lvl + Dist_nearest_station + mature + Dist_CBD + Dist_nearest_GHawker + Dist_nearest_mall + Dist_nearest_beach + Dist_nearest_CC + Dist_nearest_primary_school + Dist_nearest_secondary_school + Dist_nearest_jc + Dist_nearest_polytechnic + Dist_nearest_university + Dist_nearest_hospital, data = train)
predmlr1 = predict.lm(housemlr1, newdata=test)
mlrrmse1 = sqrt(mean((test$resale_price-predmlr1)^2))
summary(housemlr1)
# RMSE: 67.16, R2 adjusted = 0.8479
housemlr2 = lm(formula = resale_price ~  Remaining_lease + floor_area_sqm + max_floor_lvl + mature + Dist_CBD + Dist_nearest_station + room_type + MRT, data = train)
predmlr2 = predict.lm(housemlr2, newdata=test)
mlrrmse2 = sqrt(mean((test$resale_price-predmlr2)^2))
summary(housemlr2)
# RMSE: 65.63, R2 adjusted = 0.8501


## KNN predictor by finding best k
set.seed(1101)

# all predictors

houseknn=train.kknn(resale_price~.,data=train,kmax=100, kernel = "rectangular")
kbest=houseknn$best.parameters$k # K best is 2
knnreg = kknn(resale_price~.,train,test,k=kbest,kernel = "rectangular")
knnrmse = sqrt(mean((test$resale_price-knnreg$fitted.values)^2)) #test set MSE
# rmse = 70.69

houseknn=train.kknn(resale_price~.,data=train,kmax=100, kernel = "gaussian")
kbest=houseknn$best.parameters$k # K best is 2
knnreg = kknn(resale_price~.,train,test,k=kbest,kernel = "rectangular")
knnrmse = sqrt(mean((test$resale_price-knnreg$fitted.values)^2)) #test set MSE
# rmse = 70.41

# specified predictors (using kernel as gaussian)
house2knn=train.kknn(resale_price~ Remaining_lease + floor_area_sqm + max_floor_lvl + mature + Dist_nearest_station + Dist_CBD + room_type + MRT,data=train,kmax=100, kernel = "gaussian")
kbest2=house2knn$best.parameters$k # K best is 2
knnreg2 = kknn(resale_price~Remaining_lease + floor_area_sqm + max_floor_lvl + mature + Dist_nearest_station + Dist_CBD + room_type + MRT,train,test,k=kbest2,kernel = "gaussian")
knnrmse2 = sqrt(mean((test$resale_price-knnreg2$fitted.values)^2)) #test set RMSE
# rmse = 45.04 

# specified predictors (using kernel as rectangular)
house3knn=train.kknn(resale_price~ Remaining_lease + floor_area_sqm + max_floor_lvl + mature + Dist_nearest_station + Dist_CBD + room_type + MRT,data=train,kmax=100, kernel = "rectangular")
kbest3=house3knn$best.parameters$k # K best is 2
knnreg3 = kknn(resale_price~Remaining_lease + floor_area_sqm + max_floor_lvl + mature + Dist_nearest_station + Dist_CBD + room_type + MRT,train,test,k=kbest2,kernel = "rectangular")
knnrmse3 = sqrt(mean((test$resale_price-knnreg3$fitted.values)^2)) #test set RMSE
# rmse = 46.22

## Decision Tree

# All predictors

hse.tree = rpart(resale_price~.,method="anova",data=train, minsplit=5,cp=.0005)
bestcp= hse.tree$cptable[which.min(hse.tree$cptable[,"xerror"]),"CP"] 
best.tree = prune(hse.tree,cp=bestcp) #get tree for best cp on CV
plot(best.tree,uniform=TRUE)
text(best.tree,digits=4,use.n=TRUE,fancy=FALSE,bg='lightblue') 
treefit=predict(best.tree,newdata=test,type="vector") #prediction on test data
treermse = sqrt(mean((test$resale_price-treefit)^2))
# rmse = 53.13

big.tree2 = tree(resale_price~.,data=train,mindev=0.0001)
cv.bigtree2 = cv.tree(big.tree2, , prune.tree) #10-fold cross-validation
bestcp = cv.bigtree2$size[max(which(cv.bigtree2$dev == min(cv.bigtree2$dev)))]
final.tree=prune.tree(big.tree2,best=bestcp)
plot(final.tree,type="uniform")
text(final.tree,col="blue",label=c("yval"),cex=.8)
treefit2=predict(final.tree, newdata=test)
treermse2 = sqrt(mean((test$resale_price-treefit2)^2))
# rmse = 81.12

# Specified predictors

hse.tree = rpart(resale_price~ Remaining_lease + floor_area_sqm + max_floor_lvl + mature + Dist_nearest_station + Dist_CBD + room_type + MRT,method="anova",data=train, minsplit=5,cp=.0005)
bestcp2= hse.tree$cptable[which.min(hse.tree$cptable[,"xerror"]),"CP"] 
best.hsetree = prune(hse.tree,cp=bestcp2) #get tree for best cp on CV
plot(best.hsetree,uniform=TRUE)
text(best.hsetree,digits=4,use.n=TRUE,fancy=FALSE,bg='lightblue') 
housefit=predict(best.hsetree,newdata=test,type="vector") #prediction on test data
hsetreermse = sqrt(mean((test$resale_price-housefit)^2))
# rmse = 55.33

big.tree2 = tree(resale_price~Remaining_lease + floor_area_sqm + max_floor_lvl + mature + Dist_nearest_station + Dist_CBD + room_type + MRT,data=train,mindev=0.0001)
cv.bigtree2 = cv.tree(big.tree2, , prune.tree) #10-fold cross-validation
bestcp = cv.bigtree2$size[max(which(cv.bigtree2$dev == min(cv.bigtree2$dev)))]
final.tree=prune.tree(big.tree2,best=bestcp)
plot(final.tree,type="uniform")
text(final.tree,col="blue",label=c("yval"),cex=.8)
treefit2=predict(final.tree, newdata=test)
treermse2 = sqrt(mean((test$resale_price-treefit2)^2))
# rmse = 81.12


# Selectively plot a portion of the tree (e.g., nodes 2 and 3)
prp(best.hsetree, extra = 101, fallen.leaves = FALSE, varlen = 0, faclen = 0, cex=0.5)

## PCA regression

library(dplyr)

# Select only numeric columns
house_numeric <- select_if(house, is.numeric)
constant_var_cols <- nearZeroVar(house_numeric, saveMetrics = TRUE)$zeroVar
house_fin <- house_numeric[, !constant_var_cols]

# Perform principal component analysis
prall <- prcomp(house_fin, scale = TRUE)
biplot(prall)
prall.s = summary(prall)
prall.s$importance

scree = prall.s$importance[2,]
plot(scree, main = "Scree Plot", xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", ylim = c(0,1), type = 'b', cex = .8)
# 69 PCs required to explain 80% variability of data

# PC regression
pcr.fit=pcr(resale_price~.,data=test, scale=TRUE, validation="CV")
validationplot(pcr.fit, val.type="MSEP", main="LOOCV",legendpos = "topright")
# error unable to resolve
# Error in La.svd(X) : infinite or missing values in 'x'



