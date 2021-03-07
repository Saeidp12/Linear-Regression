# Linear Regression With R

#---------- Loading The Dataset
# install.packages("mlbench")
library(mlbench)

data("BostonHousing")

#---------- To View The first 6 observations of the dataset:
head(BostonHousing)

#---------- To Evaluate Missing Values of the Data (A table of true-false
#---------- for each variable)
apply(BostonHousing,2 , function(x) table(complete.cases(x)))

#---------- To view the dimension of the data:
dim(BostonHousing)

#---------- To shuffle the data 
shuffle_index = sample(1:nrow(BostonHousing), nrow(BostonHousing), replace = FALSE)

#---------- To divide the set of observations we put 80% of the data in train set
#---------- and the rest in the test set
train_index = shuffle_index[1:floor(0.8*length(shuffle_index))]

#---------- We put each observation with an index from the train_index set in the training data
#---------- and then we put the remainder in the test data
train_data = BostonHousing[train_index,]
test_data = BostonHousing[-train_index,]

#---------- Now we can train our model using the lm() function
model = lm(medv~., data=train_data)

#---------- To view the coefficients only
model

#---------- To dive in deeper and view some interesting stuff
summary(model)

#---------- With the model in our hands, we can predict our test data
ytest_pred = predict(model, test_data[,-14])

#---------- Now we can evaluate our prediction with metrics such as mse (Mean Squared Error)
mse = mean((test_data[,14] - ytest_pred)^2)
mse 

#---------- Lastly, in order to improve the model we can do many things such as feature selection,
#---------- variable standardization and etc.