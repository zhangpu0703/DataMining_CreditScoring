from sklearn import svm
import csv
import numpy as np
from sklearn import preprocessing
from time import clock as now
from sklearn.metrics import roc_curve, auc
import pylab as pl
train=np.genfromtxt("training.csv",dtype=None,delimiter=',')
test=np.genfromtxt("testing.csv",dtype=None,delimiter=',')
origin_train=train
origin_test=test
# train[:,2:18]=(train[:,2:18] - train[:,2:18].mean(axis=0)) / train[:,2:18].std(axis=0)
# test[:,2:18]=(test[:,2:18] - test[:,2:18].mean(axis=0)) / test[:,2:18].std(axis=0)
# train_input=train[:,2:18]
# train_target=train[:,1]
# train_indx=train[:,0]
# test_input=test[:,2:18]
# test_target=test[:,1]
# test_indx=test[:,0]
# print np.shape(train_input),np.shape(train_target)
mini_train=train[0:4999,:]
mini_test=test[0:999,:]
count=0
for i in range(np.shape(origin_train)[0]):
	if origin_train[i,1]==1:
		count+=1

train_oneclass=np.zeros([count,np.shape(origin_train)[1]])
temp_3=0
for i in range(np.shape(origin_train)[0]):
	if origin_train[i,1]==1:
		train_oneclass[temp_3,]=origin_train[i,]
		temp_3+=1

		
#train_oneclass[:,0:57]=(train_oneclass[:,0:57] - train_oneclass[:,0:57].min(axis=0)) / train_oneclass[:,0:57].std(axis=0)


def find_best_lin(train,test):
	best_r=0
	T=[]
	for pen in [0.05,0.2,1,2,10]:
		start=now()
		clf=svm.LinearSVC(C=pen,random_state=12345)
		clf.fit(train[:,2:18],train[:,1])
		finish=now()
		T.append(finish-start)
		scores=clf.decision_function(test[:,2:18])
		print pen
		scaled_score=scores
		# for i in range(len(scores)):
		# 	scaled_score[i]=(scores[i]-min(scores))/(max(scores)-min(scores))

		fpr, tpr, thresholds = roc_curve(test[:,1], scaled_score)
		roc_auc = auc(fpr, tpr)
		print roc_auc
		r_score=clf.score(test[:,2:18],test[:,1])
		if best_r<roc_auc:
			best_clf=clf
			best_r=roc_auc
			best_pen=pen
	return best_pen,best_r,best_clf,T

# best_pen,best_r,best_clf,T=find_best_lin(train,test)
# scores=best_clf.decision_function(test[:,2:18])
# scaled_score=scores
# for i in range(len(scores)):
# 	scaled_score[i]=(scores[i]-min(scores))/(max(scores)-min(scores))
# # Compute ROC curve and area the curve
# fpr, tpr, thresholds = roc_curve(mini_test[:,1], scaled_score)
# roc_auc = auc(fpr, tpr)
# print "Area under the ROC curve : %f" % roc_auc

# # Plot ROC curve
# print "The best pen for Linear SVM is:",best_pen
# print "The best score for Linear SVM is:",best_r
# print "Time for Computing is:", T
# pl.clf()
# pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# pl.plot([0, 1], [0, 1], 'k--')
# pl.xlim([0.0, 1.0])
# pl.ylim([0.0, 1.0])
# pl.xlabel('False Positive Rate')
# pl.ylabel('True Positive Rate')
# pl.title('Receiver operating characteristic example')
# pl.legend(loc="lower right")
# pl.show()

def oneclass_best_rbf(train_oneclass,test):
	best_r=0
	best_gamma=0
	T=[]
	S=[]
	#for para in [2e-6,2e-3,0.02,0.05,0.1,0.2,1]:
	#for para in [2e-12,2e-8,2e-6,2e-4,0.02]:
	for para in [2e-6]:
		clf_oneclass=svm.OneClassSVM(kernel='rbf',gamma=para)
		start=now()
		clf_oneclass.fit(train_oneclass[:,2:18])
		finish=now()
		T.append(finish-start)
		print para
		scores=clf_oneclass.decision_function(test[:,2:18])
		fpr, tpr, thresholds = roc_curve(test[:,1], scores)
		roc_auc = auc(fpr, tpr)
		print roc_auc
		#score=0.0
		#for i in range(np.shape(test)[0]):
			#if ((predict[i]==-1.0)&(test[i,1]==0)) or ((predict[i]==1.0)&(test[i,1]==1)):
				#score=score+1.0

		#score=score/np.shape(test)[0]
		#S.append(score)

		if best_r<roc_auc:
			best_r=roc_auc
			best_gamma=para
			best_clf=clf_oneclass

	return best_gamma,best_r,T,best_clf,S

best_gamma,best_r,T,best_clf,S=oneclass_best_rbf(train_oneclass,test)
predict=best_clf.predict(test[:,2:18])
# Compute ROC curve and area the curve
test_oneclass=test
total_pos=0
total_neg=0
false_pos=0
true_pos=0


for i in range(np.shape(test)[0]):
	print test[i,1]
	if test[i,1]==0:
		total_neg+=1
		if predict[i]==1.0:
			false_pos+=1
	elif test[i,1]==1:
		total_pos+=1
		if predict[i]==1.0:
			true_pos+=1

print "Number of total pos:", total_pos
print "Number of true pos:", true_pos
print "Number of total neg:", total_neg
print "Number of false pos:", false_pos
scores=best_clf.decision_function(test[:,2:18])
fpr, tpr, thresholds = roc_curve(test_oneclass[:,1], scores)
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc

print "The best gamma for RBF oneclassSVM is:",best_gamma
print "The best score for RBF oneclassSVM is:",best_r
print "Time consuming:", T
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.6f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right")
pl.show()


def find_best_rbf(train,test):
	best_r=0
	T=[]
	for para in [[0.1,1e-6],[0.1,1e-4],[1,1e-4],[1,0.01],[10,0.01],[10,1]]:
		pen=para[0]
		gam=para[1]
		start=now()
		clf=svm.SVC(C=pen,gamma=gam)
		print para
		clf.fit(train[:,2:18],train[:,1])
		score=clf.score(test[:,2:18],test[:,1])
		finish=now()
		T.append(finish-start)
		print score
		if best_r<score:
			best_clf=clf
			best_r=score
			best_pen=pen
			best_gamma=para
	return best_gamma,best_pen,best_r,best_clf,T

# best_gamma,best_pen,best_r,best_clf,T=find_best_rbf(train,test)
# print "The best penalty term for RBF SVM is:",best_pen
# print "The best score for RBF SVM is:",best_r
# print "The best Gamma for RBF SVM is:",best_gamma
# print "Time consuming:", T



