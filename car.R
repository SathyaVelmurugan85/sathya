
library(tseries)
library(forecast)
library(car)

setwd("C:\\R\\Linear\\car-consume")

car.data <- read.csv("measurements2.csv")

colnames(car.data)
dim(car.data)

##car.data <- subset(car.data, select = -c(6,11,12))
View(car.data)

car.data$specials <-  ifelse(car.data$specials == "",0, as.numeric(car.data$specials))
car.data$gas_type<-as.numeric(car.data$gas_type)
car.data$refill.gas <- ifelse(car.data$refill.gas == "",0,as.numeric(car.data$refill.gas))
car.data$refill.liters <- ifelse(is.na(car.data$refill.liters),0,car.data$refill.liters)
car.data$temp_inside <- ifelse(is.na(car.data$temp_inside),0,car.data$temp_inside)

View(car.data)

set.seed(100)


temp.car <- sample(c("train","test"), nrow(car.data),replace = T , prob = c(.8,.2))
temp.car

train.car <- car.data[temp.car == "train",]
test.car <- car.data[temp.car == "test",]

cor(car.data)

carmodel1 <- lm(consume ~.,train.car)
summary(carmodel1)

vif(carmodel1)

carmodel2 <- lm(consume ~ +ï..distance+speed+temp_outside+temp_inside+gas_type+AC,train.car)
summary(carmodel2)

vif(carmodel2)

step.train <- step(carmodel2,direction = "both")
summary(step.train)

final.carmodel <- lm(consume ~ + speed + temp_inside +temp_outside + AC, train.car)
summary(final.carmodel)


test.predictcar <- predict.lm(final.carmodel, newdata = test.car)
test.predictcar

testing.data <- test.car[,2]
View(testing.data)

bind.testcar <- cbind(test.predictcar,testing.data)
View(bind.testcar)



qqnorm(final.carmodel$residuals)
qqline(final.carmodel$residuals)
hist(final.carmodel$residuals)
shapiro.test(final.carmodel$residuals)


durbinWatsonTest(final.carmodel)
crPlots(final.carmodel)


##min_max_accuracy <- mean(apply(bind.testcar, 1, min) / apply(bind.testcar, 1, max))
##min_max_accuracy
