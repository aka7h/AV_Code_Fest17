library(xgboost)
library(data.table)
library(ggplot2)
library(gridExtra)
library(e1071)
library(caret)
library(lubridate)
library(anytime)

user  <- read.csv('train/user_data.csv', na.strings = "",stringsAsFactors = F)
problem <- read.csv('train/problem_data.csv',na.strings = "",stringsAsFactors = F)
train <-  read.csv('train/train_submissions.csv')
samplesub <- read.csv('sample_submissions.csv')
test <- read.csv('test_submissions.csv')

summary(test)
summary(train)
summary(user)
summary(problem)

#we have already converted the problem$tags as DocumentTermMatrix lets use that code

library(tm)

doc <- VCorpus(VectorSource(problem$tags))
doc <- tm_map(doc,tolower)
doc <- tm_map(doc, removeWords, stopwords(kind="en"))
doc <- tm_map(doc, removePunctuation)
doc <- tm_map(doc, stripWhitespace)
doc <- tm_map(doc, stemDocument)
doc <- tm_map(doc, PlainTextDocument)

dtm <- DocumentTermMatrix(doc)
dim(dtm)

dense_dtm <- removeSparseTerms(dtm,0.993)
dim(dense_dtm)

newdt <- as.data.frame(as.matrix(dense_dtm))
colnames(newdt) <- make.names(colnames(newdt))
row.names(newdt) <- 1:6544

problem <- cbind(problem, newdt)

mean(problem$points,na.rm=T)
median(problem$points,na.rm = T)

problem$points[is.na(problem$points)] = median(problem$points,na.rm = T)
#problem$points[is.na(problem$points)] = -1
summary(problem$points)
problem$tags <- NULL

#checking the NA in r
'checkNA' <- function(df){
  cNA <- lapply(df, function(x) sum(is.na(x)))
  return(cNA)
}

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

#lets create ID
train$ID <- paste(train$user_id,train$problem_id,sep="_")

#noe lets merge the dataset and create ntraina dn ntest
ntrain <- merge(train,user,by="user_id")
ntrain <- merge(ntrain,problem,by="problem_id")

ntest <- merge(test,user,by="user_id")
ntest <- merge(ntest,problem,by="problem_id")

ID <- ntest$ID

checkNA(ntrain)
checkNA(ntest)
setDT(ntrain)
setDT(ntest)

#imputing mode values in for both train and test
ntrain$level_type[is.na(ntrain$level_type)] <- getmode(ntrain$level_type)
ntest$level_type[is.na(ntest$level_type)] <- getmode(ntest$level_type)

#from the summary we can see that the country has the most numer of missing NA.
#we cannot impute mode to country since the count of missing value is higher than count of other's
#hence lets create a new category 'Others'

ntrain$country[is.na(ntrain$country)] <- 'Others'
ntest$country[is.na(ntest$country)] <- 'Others'


ntrain[,set:="train"]
ntest[,set:='test']

ntest[,attempts_range:=NA]

full <- rbind(ntrain,ntest)
str(full)


full[,nUser := sum(attempts_range,na.rm = T),.(user_id)]
full[,dUser := sum(set=="train",na.rm = T),.(user_id)]
full[,tgtUser := ifelse(set=='train',(nUser - attempts_range)/(dUser-1),nUser/dUser)]

full[,nProblem := sum(attempts_range,na.rm = T),.(problem_id)]
full[,dProblem := sum(set=="train",na.rm = T),.(problem_id)]
full[,tgtProblem := ifelse(set=='train',(nProblem - attempts_range)/(dProblem-1),nProblem/dProblem)]

full[,nUserProb := sum(attempts_range,na.rm = T),.(ID)]
full[,dUserProb := sum(set=="train",na.rm = T),.(ID)]
full[,tgtUserProb := ifelse(set=='train',(nUserProb - attempts_range)/(dUserProb-1),nUserProb/dUserProb)]

full[,nCountry := sum(attempts_range,na.rm = T),.(country)]
full[,dCountry := sum(set=="train",na.rm = T),.(country)]
full[,tgtCountry := ifelse(set=="train",(nCountry - attempts_range)/(dCountry-1),nCountry/dCountry)]

full[,nRank := sum(attempts_range,na.rm=T),.(rank)]
full[,dRank := sum(set=='train',na.rm=T),.(rank)]
full[,tgtRank := ifelse(set=="train",(nRank-attempts_range)/(dRank-1),nRank/dRank)]

full[,nLevel := sum(attempts_range,na.rm=T),.(level_type)]
full[,dLevel := sum(set=="train",na.rm=T),.(level_type)]
full[,tgtLevel := ifelse(set=="train",(nLevel-attempts_range)/(dLevel-1),nLevel/dLevel)]

#convert the unix timestamp to date time
full[,registration := parse_date_time(anytime(registration_time_seconds),'ymd HMS')]
full[,lastonline := parse_date_time(anytime(last_online_time_seconds),'ymd HMS')]
full[,datediff := as.numeric(lastonline-registration),.(user_id)]

#full[,reg_date:= day(registration)][,reg_month:=month(registration)][,reg_year:=year(registration)][,reg_hrs:=hour(registration)][,reg_min:=minute(registration)][,reg_sec:=second(registration)][,lo_date:= day(lastonline)][,lo_month:=month(lastonline)][,lo_year:=year(lastonline)][,lo_hrs:=hour(lastonline)][,lo_min:=minute(lastonline)][,lo_sec:=second(lastonline)]


full[,nUserCountry := sum(attempts_range,na.rm=T),.(user_id,country)]
full[,dUserCountry := sum(set=="train",na.rm=T),.(user_id,country)]
full[,tgtUserCountry := ifelse(set=='train',(nUserCountry - attempts_range)/(dUserCountry-1),nUserCountry/dUserCountry)]

full[,nUserRank := sum(attempts_range,na.rm=T),.(user_id,rank)]
full[,dUserRank := sum(set=="train",na.rm=T),.(user_id,rank)]
full[,tgtUserRank := ifelse(set=='train',(nUserRank - attempts_range)/(dUserRank-1),nUserRank/dUserRank)]

full[,nUserCountry := sum(attempts_range,na.rm=T),.(user_id,country)]
full[,dUserCountry := sum(set=="train",na.rm=T),.(user_id,country)]
full[,tgtUserCountry := ifelse(set=='train',(nUserCountry - attempts_range)/(dUserCountry-1),nUserCountry/dUserCountry)]

full[,nUserLevel := sum(attempts_range,na.rm=T),.(user_id,level_type)]
full[,dUserLevel := sum(set=="train",na.rm=T),.(user_id,level_type)]
full[,tgtUserLevel := ifelse(set=='train',(nUserLevel - attempts_range)/(dUserLevel-1),nUserLevel/dUserLevel)]

full[,nProbCountry := sum(attempts_range,na.rm=T),.(problem_id,country)]
full[,dProbCountry := sum(set=="train",na.rm=T),.(problem_id,country)]
full[,tgtProbCountry := ifelse(set=='train',(nProbCountry - attempts_range)/(dProbCountry-1),nProbCountry/dProbCountry)]

full[,nProbRank := sum(attempts_range,na.rm=T),.(problem_id,rank)]
full[,dProbRank := sum(set=="train",na.rm=T),.(problem_id,rank)]
full[,tgtProbRank := ifelse(set=='train',(nProbRank - attempts_range)/(dProbRank-1),nProbRank/dProbRank)]

full[,nProbLevel := sum(attempts_range,na.rm=T),.(problem_id,level_type)]
full[,dProbLevel := sum(set=="train",na.rm=T),.(problem_id,level_type)]
full[,tgtProbLevel := ifelse(set=='train',(nProbLevel - attempts_range)/(dProbLevel-1),nProbLevel/dProbLevel)]


full$country <- as.factor(full$country)
full$level_type <- as.factor(full$level_type)
full$rank <- as.factor(full$rank)

#Now lets remove the following
full[,c('registration_time_seconds','last_online_time_seconds')] <- NULL

checkNA(full)
full$tgtUserProb <- NULL #since all the values are missing
full[is.na(full)] <- 0 #we are imputing NaN with 0.


ntrain <- full[set == "train"]
ntest <- full[set == "test"]
ntest$attempts_range <- NULL

# write.csv(ntrain,'train_full.csv',row.names = F)
# write.csv(ntest,'test_full.csv',row.names = F)

# #here we are checking the near zero columns
# nearZero <- nearZeroVar(ntrain[,-c(1:4)],saveMetrics = T)
# nearZero
# nearZero <- nearZeroVar(ntrain[,-c(1:4)])
# nearZero <- nearZero+4
# #these list of 
# colName <- colnames(ntrain)
# removeCol <- colName[nearZero]
# ntrain[,(removeCol) := NULL]
# ntest[,(removeCol):=NULL]


#Visualization
plotDen <- function(data_in,i){
  d <- data.frame(x=data_in[[i]])
  p <- ggplot(d)+geom_line(aes(x=x),stat = "density", size=1, color="red")+theme_light()+
    xlab(paste0(colnames(data_in)[i],"\n Skewness: ",round(skewness(data_in[[i]], na.rm = TRUE),2)))
  return(p)
}

plotHist <- function(data_in,i){
  d <- data.frame(x=data_in[[i]])
  p <- ggplot(d)+stat_count(aes(x=x))+theme_light()+xlab(colnames(data_in)[i])+theme(axis.text.x = element_text(angle = 90, hjust =1))
  return(p)
}

plotfunct <- function(dt,fun,features,ncol){
  pp <- list()
  for(ii in features){
    p <- fun(data_in=dt,i=ii)
    pp <- c(pp, list(p))
  }
  do.call("grid.arrange",c(pp,ncol=ncol))
}

#here we are consider this as a multi class problem
plotfunct(ntrain, fun=plotDen, features = 3,ncol = 1)
#from this we can understand that most of the user will attempted the problem atleast once.




#lets analyze rank and level_type
plotfunct(ntrain, fun=plotHist, features = c(12,14),ncol = 2)
#country
plotfunct(ntrain, fun=plotHist,features = 7,ncol=1)

plotfunct(ntrain, fun=plotDen, features = c(4:6,8),ncol=2)
plotfunct(ntrain, fun=plotDen, features = c(10,11,15), ncol=2)

checkNA(ntrain)


set.seed(112)
samp <- sample(nrow(ntrain),size = 0.8*nrow(ntrain))

ntr <- ntrain[samp]
ntv <- ntrain[-samp]

library(xgboost)

train_x <- ntrain
train_x$set <- NULL
train_x$user_id <- NULL
train_x$problem_id <- NULL
train_x$ID <- NULL
train_x[] <- lapply(train_x,as.numeric)
train_y <- train_x$attempts_range - 1
train_x$attempts_range <- NULL


# train_v <- ntv
# train_v$set <- NULL
# train_v$user_id <- NULL
# train_v$problem_id <- NULL
# train_v$ID <- NULL
# train_v[] <- lapply(train_v,as.numeric)
# train_vy <- train_v$attempts_range -1
# train_v$attempts_range <- NULL


train_t <- ntest
train_t$set <- NULL
train_t$user_id <- NULL
train_t$problem_id <- NULL
train_t$ID <- NULL
train_t[] <- lapply(train_t,as.numeric)



train_1 <- xgb.DMatrix(as.matrix(train_x),label=train_y)
# train_2 <- xgb.DMatrix(as.matrix(train_v))
train_3 <- xgb.DMatrix(as.matrix(train_t))

xgb.par <- list(colsample_bytree = 0.7, #how many variables to consider for each tree
                subsample = 0.7, #how much of the data to use for each tree
                booster = "gbtree",
                max_depth = 5, #how many levels in the tree
                eta = 0.05, #shrinkage rate to control overfitting through conservative approach
                objective = "multi:softmax",
                eval_metric="mlogloss",
                gamma = 3,
                num_class=6
)

# xgbcv <- xgb.cv(params = xgb.par, nfold = 4,nrounds = 500, 
#                 print_every_n = 2,early_stopping_rounds = 100,data = train_1)

xgb_model <- xgb.train(xgb.par,train_1,
                       nrounds = 1000, print_every_n = 1)
xgb_model

xgb.importance(feature_names = colnames(train_x), model=xgb_model)


#train
xgb.train.predict <- predict(xgb_model,train_1)
confusionMatrix(table(xgb.train.predict,train_y))
#validation
# xgb.val.predict <- predict(xgb_model,train_2)
# confusionMatrix(table(xgb.val.predict,train_vy))



#test
xgb.test.predict <- predict(xgb_model,train_3)
xgb.test.predict <- xgb.test.predict + 1



#############################################################################
#xgb_model v2

#noe lets merge the dataset and create ntraina dn ntest
# ntrain2 <- merge(train,user,by="user_id")
# ntrain2 <- merge(ntrain2,problem,by="problem_id")
# 
# ntest2 <- merge(test,user,by="user_id")
# ntest2 <- merge(ntest2,problem,by="problem_id")
# 
# 
# checkNA(ntrain2)
# summary(ntrain2)
# 
# ntrain2$contribution[ntrain2$contribution < 0] <- 0
# ntest2$contribution[ntest2$contribution < 0] <- 0
# 
# ntest2$points[ntest2$points < 0] <- 0
# ntrain2$points[ntrain2$points < 0] <- 0
# 
# library(gbm)




submission <- data.frame('ID'=test$ID,'attempts_range'=xgb.test.predict)
write.csv(submission,'ak_xgboost_24_0256.csv',row.names = F)


