library(glmnet)
library(MASS)
library(ROCR)
library(ggplot2)
library(caret)
#Main Logistic Regression
trdat=read.csv("C:/Users/Zhang/Desktop/2014Fall/Data Mining/Project/training.csv")
tedat=read.csv("C:/Users/Zhang/Desktop/2014Fall/Data Mining/Project/testing.csv")
train_input=as.matrix(trdat[,3:19])
train_tar=as.matrix(trdat[,2])
test_input=as.matrix(tedat[,3:19])
test_tar=as.matrix(tedat[,2])
minor_train=trdat[trdat[,2]==1,3:19]
major_train=trdat[trdat[,2]==0,3:19]
minor_mean=colMeans(minor_train)

km <- kmeans(major_train, 6)
cl_mean=(km$centers)
cl=km$cluster
dist=matrix(0,nrow=1,ncol=0)
for (i in 1:nrow(cl_mean)){
  dist=cbind(dist,sum(abs(cl_mean[i,]-minor_mean)))
}
dist_sort=order(dist)
new_idx=dist_sort[2:6]
new_major_train=matrix(0,nrow=0,ncol=18)
for (i in (new_idx)){
  new0=cbind(major_train[cl==i,],matrix(0,nrow=nrow(major_train[cl==i,]),ncol=1))
  new_major_train=rbind(new_major_train,new0)
}
#new_major_train=cbind(major_train[cl==new_idx,],matrix(0,nrow=nrow(major_train[cl==new_idx,]),ncol=1))
new_minor_train=cbind(minor_train,matrix(1,nrow=nrow(minor_train),ncol=1))
new_train=rbind(as.matrix(new_major_train),as.matrix(new_minor_train))
x<-scale(new_train[,1:17])
testx<-scale(test_input)
logfit<-cv.glmnet(x,new_train[,18],family="binomial",alpha=0,nfolds=20)
plot(logfit)
logfit$lambda.min

trainpred<-predict(logfit,newx=x,s="lambda.min")
train.error.rate<-mean((trainpred>0.5&new_train[,18]==0)|(trainpred<0.5&new_train[,18]==1))
testpred<-predict(logfit,newx=testx,s="lambda.min",type="response")
test.error.rate<-mean((testpred>0.5&test_tar==0)|(testpred<0.5&test_tar==1))
pred<-prediction(testpred,test_tar)
perf<-performance(pred,"tpr","fpr")
auc<-attr(performance(pred,"auc"),"y.values")[[1]]
tpr<-unlist(perf@x.values)
fpr<-unlist(perf@y.values)
roc<-cbind(tpr,fpr)
ggplot(data=data.frame(roc),aes(x=tpr,y=fpr))+theme_bw()+
  geom_line(aes(x=tpr,y=fpr),color="red")+
  geom_abline(slope=1,intercept=0,color='blue')+
  labs(title="ROC Curve",
       x="False Positive Rate",
       y="True Positive Rate")

coef(logfit,s="lambda.min")
result=cbind(testpred,test_tar)

write.csv(result, file = "C:/Users/Zhang/Desktop/2014Fall/Data Mining/Project/clustering_classify.csv",row.names=FALSE)
read.csv("clustering_classify.csv")
