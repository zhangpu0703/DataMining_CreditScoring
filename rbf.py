from sklearn import svm
import csv
import numpy as np
from sklearn import preprocessing
from time import clock as now
from sklearn.metrics import roc_curve, auc
import pylab as pl
train=np.genfromtxt("training.csv",dtype=None,delimiter=',')
test=np.genfromtxt("testing.csv",dtype=None,delimiter=',')

train[:,2:18]=(train[:,2:18] - train[:,2:18].mean(axis=0)) / train[:,2:18].std(axis=0)
test[:,2:18]=(test[:,2:18] - test[:,2:18].mean(axis=0)) / test[:,2:18].std(axis=0)
train_input=train[:,2:18]
train_target=train[:,1]
train_indx=train[:,0]
test_input=test[:,2:18]
test_target=test[:,1]
test_indx=test[:,0]
print np.shape(train_input),np.shape(train_target)
mini_train=train[0:9999,:]
mini_test=test[0:4999,:]

def find_best_rbf(train,test):
	best_r=0
	T=[]
	pen=1.0
	for para in [10.0]:
		start=now()
		clf=svm.SVC(kernel="rbf",probability=True,class_weight="auto")
		clf.fit(train[:,2:18],train[:,1])
		finish=now()
		T.append(finish-start)
		print para
		scores=clf.predict_proba(test[:,2:18])
		print scores[17,:]
		print scores[16,:]
		fpr, tpr, thresholds = roc_curve(test[:,1], scores[:,1])
		roc_auc = auc(fpr, tpr)
		print roc_auc
		r_score=clf.score(test[:,2:18],test[:,1])
		if best_r<roc_auc:
			best_clf=clf
			best_r=roc_auc
			best_para=para
			best_scores=scores
	return best_clf,T,best_scores


best_clf,T,best_scores=find_best_rbf(train,test)

print "Time Consuming:", T
# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(test[:,1], best_scores[:,1])
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc

# Plot ROC curve
#print "The best gamma for Linear SVM is:",best_para
#print "The best score for Linear SVM is:",best_r
print "Time for Computing is:", T
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right")
pl.show()