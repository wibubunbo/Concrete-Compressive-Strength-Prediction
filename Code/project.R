library(ggplot2)
library(tidyr)
library(corrplot)
library(moments)
library(boot)
library(car)
library(caret)
library(glmnet)
library(randomForest)
library(tree)
library(leaps)
library(gbm)
library(BART)
library(MLmetrics)

# Loading the dataset
df <- read.csv("D:\\wibubunbo\\R\\Concrete_Data.csv")
colnames(df) <- c("cement", "blast_furnace_slag", "fly_ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age", "strength")
head(df)

# Checking the duplications in our dataset
sum(duplicated(df))
df <- df[!duplicated(df), ]

# Find the observations with same predictors value but different target value
rownames(df) <- 1:nrow(df) # reset the index of our dataset
check <- df[, -9]
special_rows <- as.numeric(rownames(check[duplicated(check) | duplicated(check, fromLast = TRUE), ]))
head(df[special_rows, ])

# Calculate the mean of these observations and merge it into our dataset
mean_df_rows <- aggregate(strength~., data = df, subset = special_rows, mean)
df <- df[-special_rows, ]
df <- rbind(df, mean_df_rows)
rownames(df) <- 1:nrow(df) #  reset the index of our dataset

# Spliting our dataset to training and test dataset
set.seed(1)
train <- sample(dim(df)[1], dim(df)[1] * 0.8)
df.train <- df[train, ]
df.test <- df[-train, ]

# Create X and y
X_train <- df.train[, -9]
y_train <- df.train$strength
X_test <- df.test[, -9]
y_test <- df.test$strength

# Distribution plots of all columns
dl <- df %>%
  pivot_longer(colnames(df)) %>% 
  as.data.frame()

ggplot(dl, aes(x = value)) +    
  geom_histogram(aes(y = ..density..), col = "gray") + 
  geom_density(col = "lightblue", size = 1) + 
  facet_wrap(~ name, scales = "free")

# Correlation matrix of all features and target strength
corrplot(cor(df), method = "number", number.cex = 0.8)

# Multiple Linear Regression
lm.fit <- lm(strength~., data = df.train)
print(summary(lm.fit))
# Check VIF values of lm.fit
v0 <- data.frame(vif(lm.fit))
v0

# Remove coarse_agg and fine_agg features and do OLS again
lm.fit1 <- lm(strength~cement+blast_furnace_slag+fly_ash+water+superplasticizer+age, data = df.train)
summary(lm.fit1)
v1 <- data.frame(vif(lm.fit1))
v1

# Calculate train and test RMSE + R^2 score of OLS
y_train_pred <- predict(lm.fit1, df.train)
y_test_pred <- predict(lm.fit1, df.test)
print(RMSE(y_train, y_train_pred)) # Train RMSE
print(R2_Score(y_train_pred, y_train)) # Train R^2 score
print(RMSE(y_test, y_test_pred)) # Test RMSE
print(R2_Score(y_test_pred, y_test)) # Test R^2 score

predict.regsubsets = function(object, newdata, id, ...) {
    form = as.formula(object$call[[2]])
    mat = model.matrix(form, newdata)
    coefi = coef(object, id = id)
    mat[, names(coefi)] %*% coefi
}

# Best subset selection with 10-fold CV
set.seed(1)
k <- 10
folds <- sample(rep(1:k, length = nrow(df)))
cv.errors <- matrix(NA, k, 8, dimnames = list(NULL, paste(1:8)))
for(j in 1:k) {
  best.fit <- regsubsets(strength~., data = df[folds != j, ], nvmax = 8)
  for(i in 1:8) {
    pred <- predict.regsubsets(best.fit, df[folds == j, ], id = i)
    cv.errors[j, i] <- mean((df$strength[folds == j] - pred)^2)
  }
}
mean.cv.errors <- apply(cv.errors, 2, mean)
which.min(mean.cv.errors) # Best subset selection result

plot(mean.cv.errors , type = "b", xlab = "Number of Features", ylab = "Mean Cross-validation Errors")
text(mean.cv.errors, labels=round(mean.cv.errors, 2), pos = 3)

# Coefficients of best 6-variable model
reg.best <- regsubsets(strength~., data = df.train, nvmax = 8)
coef(reg.best, 6)

# The LASSO
set.seed(1)
train.matrix <- model.matrix(strength~., data = df.train)
test.matrix <- model.matrix(strength~., data = df.test)
grid <- 10^seq(3, -2, by = -0.1)
lasso.mod <- cv.glmnet(train.matrix, df.train[, "strength"], alpha = 1, lambda = grid, standardize = TRUE)
bestlam.lasso <- lasso.mod$lambda.min
bestlam.lasso
# Calculate train and test RMSE + R^2 score
y_train_pred <- predict(lasso.mod, s = bestlam.lasso, newx = train.matrix)
y_test_pred <- predict(lasso.mod, s = bestlam.lasso, newx = test.matrix)
print(RMSE(y_train, y_train_pred)) # Train RMSE
print(R2_Score(y_train_pred, y_train)) # Train R^2 score
print(RMSE(y_test, y_test_pred)) # Test RMSE
print(R2_Score(y_test_pred, y_test)) # Test R^2 score
# Find coefficients estimate of best model
predict(lasso.mod, type = "coefficients", s = bestlam.lasso)

# Check the coefficient estimates of the LASSO
predict(lasso.mod, newx = test_mat, type = "coefficients", s = bestlam.lasso)

# WITHOUT FEATURES SELECTION

# Regression Decision Trees
set.seed(1)
tree.strength <- tree(strength~., df.train)
# Calculate train and test RMSE + R^2 score
y_train_pred <- predict(tree.strength, newdata = df.train)
y_test_pred <- predict(tree.strength, newdata = df.test)
print(RMSE(y_train, y_train_pred)) # Train RMSE
print(R2_Score(y_train_pred, y_train)) # Train R^2 score
print(RMSE(y_test, y_test_pred)) # Test RMSE
print(R2_Score(y_test_pred, y_test)) # Test R^2 score

# Bagging
set.seed(1)
bag.strength <- randomForest(strength~., data = df.train, mtry = 8, importance = TRUE)
# Calculate train and test RMSE + R^2 score
y_train_pred <- predict(bag.strength, newdata = df.train)
y_test_pred <- predict(bag.strength, newdata = df.test)
print(RMSE(y_train, y_train_pred)) # Train RMSE
print(R2_Score(y_train_pred, y_train)) # Train R^2 score
print(RMSE(y_test, y_test_pred)) # Test RMSE
print(R2_Score(y_test_pred, y_test)) # Test R^2 score

# Random Forest
set.seed(1)
rf.strength <- randomForest(strength~., data = df.train, mtry = 5, importance = TRUE)
# Calculate train and test RMSE + R^2 score
y_train_pred <- predict(rf.strength, newdata = df.train)
y_test_pred <- predict(rf.strength, newdata = df.test)
print(RMSE(y_train, y_train_pred)) # Train RMSE
print(R2_Score(y_train_pred, y_train)) # Train R^2 score
print(RMSE(y_test, y_test_pred)) # Test RMSE
print(R2_Score(y_test_pred, y_test)) # Test R^2 score

# Features Importance of Random Forest
importance(rf.strength)
varImpPlot(rf.strength)

# Boosting
set.seed(1)
boost.strength <- gbm(strength~., data = df.train, distribution = "gaussian", n.trees = 500, interaction.depth = 2)
# Calculate train and test RMSE + R^2 score
y_train_pred <- predict(boost.strength, newdata = df.train, n.trees = 500)
y_test_pred <- predict(boost.strength, newdata = df.test, n.trees = 500)
print(RMSE(y_train, y_train_pred)) # Train RMSE
print(R2_Score(y_train_pred, y_train)) # Train R^2 score
print(RMSE(y_test, y_test_pred)) # Test RMSE
print(R2_Score(y_test_pred, y_test)) # Test R^2 score

# Features Importance of Boosting
names(df.train)[names(df.train) == "blast_furnace_slag"] <- "bfslag"
names(df.train)[names(df.train) == "superplasticizer"] <- "SPs"
boost.strength <- gbm(strength~., data = df.train, distribution = "gaussian", n.trees = 500, interaction.depth = 2)
summary(boost.strength)
names(df.train)[names(df.train) == "bfslag"] <- "blast_furnace_slag"
names(df.train)[names(df.train) == "SPs"] <- "superplasticizer"

# BART  
set.seed(1)
bart.fit <- gbart(df.train[, 1:8], df.train$strength, x.test = df.test[, 1:8])
# Calculate train and test RMSE + R^2 score
y_train_pred <- bart.fit$yhat.train.mean
y_test_pred <- bart.fit$yhat.test.mean
print(RMSE(y_train, y_train_pred)) # Train RMSE
print(R2_Score(y_train_pred, y_train)) # Train R^2 score
print(RMSE(y_test, y_test_pred)) # Test RMSE
print(R2_Score(y_test_pred, y_test)) # Test R^2 score

# BART check how many times each variable appeared in the collection of trees
ord <- order(bart.fit$varcount.mean , decreasing = T)
bart.fit$varcount.mean[ord]

# WITH FEATURES SELECTION

# Regression Decision Trees
set.seed(1)
tree.strength <- tree(strength~cement+blast_furnace_slag+fly_ash+water+superplasticizer+age, df.train)
# Calculate train and test RMSE + R^2 score
y_train_pred <- predict(tree.strength, newdata = df.train)
y_test_pred <- predict(tree.strength, newdata = df.test)
print(RMSE(y_train, y_train_pred)) # Train RMSE
print(R2_Score(y_train_pred, y_train)) # Train R^2 score
print(RMSE(y_test, y_test_pred)) # Test RMSE
print(R2_Score(y_test_pred, y_test)) # Test R^2 score

# Bagging
set.seed(1)
bag.strength <- randomForest(strength~cement+blast_furnace_slag+fly_ash+water+superplasticizer+age, data = df.train, mtry = 6, importance = TRUE)
# Calculate train and test RMSE + R^2 score
y_train_pred <- predict(bag.strength, newdata = df.train)
y_test_pred <- predict(bag.strength, newdata = df.test)
print(RMSE(y_train, y_train_pred)) # Train RMSE
print(R2_Score(y_train_pred, y_train)) # Train R^2 score
print(RMSE(y_test, y_test_pred)) # Test RMSE
print(R2_Score(y_test_pred, y_test)) # Test R^2 score

# Random Forest
set.seed(1)
rf.strength <- randomForest(strength~cement+blast_furnace_slag+fly_ash+water+superplasticizer+age, data = df.train, mtry = 4, importance = TRUE)
# Calculate train and test RMSE + R^2 score
y_train_pred <- predict(rf.strength, newdata = df.train)
y_test_pred <- predict(rf.strength, newdata = df.test)
print(RMSE(y_train, y_train_pred)) # Train RMSE
print(R2_Score(y_train_pred, y_train)) # Train R^2 score
print(RMSE(y_test, y_test_pred)) # Test RMSE
print(R2_Score(y_test_pred, y_test)) # Test R^2 score

# Features Importance of Random Forest
importance(rf.strength)
varImpPlot(rf.strength)

# Boosting
set.seed(1)
boost.strength <- gbm(strength~cement+blast_furnace_slag+fly_ash+water+superplasticizer+age, data = df.train, distribution = "gaussian", n.trees = 500, interaction.depth = 2)
# Calculate train and test RMSE + R^2 score
y_train_pred <- predict(boost.strength, newdata = df.train, n.trees = 500)
y_test_pred <- predict(boost.strength, newdata = df.test, n.trees = 500)
print(RMSE(y_train, y_train_pred)) # Train RMSE
print(R2_Score(y_train_pred, y_train)) # Train R^2 score
print(RMSE(y_test, y_test_pred)) # Test RMSE
print(R2_Score(y_test_pred, y_test)) # Test R^2 score

# Features Importance of Boosting
names(df.train)[names(df.train) == "blast_furnace_slag"] <- "bfslag"
names(df.train)[names(df.train) == "superplasticizer"] <- "SPs"
boost.strength <- gbm(strength~cement+bfslag+fly_ash+water+SPs+age, data = df.train, distribution = "gaussian", n.trees = 500, interaction.depth = 2)
summary(boost.strength)
names(df.train)[names(df.train) == "bfslag"] <- "blast_furnace_slag"
names(df.train)[names(df.train) == "SPs"] <- "superplasticizer"

# BART  
set.seed(1)
bart.fit <- gbart(df.train[, c("cement", "blast_furnace_slag", "fly_ash", "water", "superplasticizer", "age")], df.train$strength, x.test = df.test[, c("cement", "blast_furnace_slag", "fly_ash", "water", "superplasticizer", "age")])
# Calculate train and test RMSE + R^2 score
y_train_pred <- bart.fit$yhat.train.mean
y_test_pred <- bart.fit$yhat.test.mean
print(RMSE(y_train, y_train_pred)) # Train RMSE
print(R2_Score(y_train_pred, y_train)) # Train R^2 score
print(RMSE(y_test, y_test_pred)) # Test RMSE
print(R2_Score(y_test_pred, y_test)) # Test R^2 score

# BART check how many times each variable appeared in the collection of trees
ord <- order(bart.fit$varcount.mean , decreasing = T)
bart.fit$varcount.mean[ord]
