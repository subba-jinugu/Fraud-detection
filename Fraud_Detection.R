setwd("~/Desktop")

# Read file

ccf <- read.csv("creditcard.csv")

# View data
View(ccf)

# Explore the data

str(ccf)

table(as.factor(ccf$Class))

# Find out fraud incident rate
round(prop.table(table(ccf$Class))*100,2)

# Split the data 
#install.packages("caret")
library(lattice)
library(ggplot2)
library(caret)

set.seed(1234)

training <- createDataPartition(ccf$Class, times=1, p=0.75, list = F)
train <- ccf[training,]
test <- ccf[-training,]

# Find out fraud incident rate between Train and Test
round(prop.table(table(train$Class))*100,2)

round(prop.table(table(test$Class))*100,2)

# Note: Proportion of fraud incident remains almost the same between tarin and test

# Let's try different clasiffiers 

#####    Logistic Regression    #####

# Logistic Regression - Fit model on train data

#install.packages("pROC")
#install.packages("PRROC")

library(pROC)
library(PRROC)

logistic <- glm(Class ~ ., data = train, family = "binomial")

# Logistic Regression - Predict on test data

logistic.predict <- predict(logistic, test)


# Add predicted value in the original table
test$predicted_class = ifelse(logistic.predict >0.5,1,0)

# Compute Confusion Matrix and get "Precision" "Recall" Chart 

table(test$Class, logistic.predict >0.5)

plot(pr.curve(test$Class, logistic.predict, curve = TRUE))

test$predicted_class<-factor(test$predicted_class)
test$Class<-factor(test$Class)

str(test)
levels(test$Class) <- c(0, 1)
levels(test$predicted_class) <- c(0, 1)

confusionMatrix(test$predicted_class, test$Class)



#####   Decision Tree   #####

# Decision Tree (CART) - Fit model on train data

library(rpart)
library(rpart.plot)

dt <- rpart(Class ~ ., data = train, method = "class")

# Plot the model
prp(dt, type=2, extra=2, under=TRUE)


# Predict on test data

predict_dt <- predict(dt, test, type = "class")

# Compute Confusion Matrix and get "Precision" "Recall" Chart 

confusionMatrix(predict_dt, test$Class)


# Conclusion: In the above couple of classifiers we see that the decision tree is doing a better job than logistic regression. 
# Similarly you can try several other classifiers such as Naive Bayes, SVM and pick the best. 
# You may also want to do oversampling and under-sampling to find the best precision and recall.

#####    Random Forest   #####

# Random Forest Classifier - Fit Model on Train Data 

library(randomForest)

RM <- randomForest(Class ~ ., data = train, ntree = 50 )



##################### ___________________________________ #####################

# let???s focus on how to detect outliers or anomalies using exploratory data analysis (EDA) and other such techniques

# Method 1- Find any values below 1% or higher than 99%

quantile(ccf$V1, c(0.01, 0.99))
quantile(ccf$V1, seq(0,1,0.05))



# Method 2- Use BOX and WHISKER plot

boxplot(ccf$V1)

outliers <- boxplot.stats(ccf$V1)$out

# Find index of Outliers

outlier_V1 <- which(ccf$V1 %in% boxplot.stats(ccf$V1)$out)
outlier_V3 <- which(ccf$V3 %in% boxplot.stats(ccf$V3)$out)

# Find values which are outliers for both vars V1 and V3

outlier_list = intersect(outlier_V1, outlier_V3)
View(outlier_list)

newdata =  ccf[outlier_list,]


# Method 3- Distence from Cluster

clusters <- kmeans(ccf, centers =5)

# Compute distence between centers points

centers <- clusters$centers[clusters$cluster, ]
distances <- sqrt(rowSums((ccf - centers)^2))

# Print top 100 outliers index

outliers <- order(distances, decreasing = T)[1:100]
View(outliers)


# There are other methods such as Local Outlier Factor Method (LOF) from package called ???DMwR???
# and other similar density or distance based methods that one can try. 

# However, please ensure to standardize/normalize your data before running any such algorithms.

