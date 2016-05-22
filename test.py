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

def find_best_lin(train,test):
	best_r=0
	T=[]
	for pen in [0.1,0.5,1,10,50]:
		start=now()
		clf=svm.LinearSVC(C=pen,class_weight="auto")
		clf.fit(train[:,2:18],train[:,1])
		finish=now()
		T.append(finish-start)
		scores=clf.predict(test[:,2:18])
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
			best_scores=scaled_score
	return best_pen,best_r,best_clf,best_scores,T

best_pen,best_r,best_clf,best_scores,T=find_best_lin(train,test)
fpr, tpr, thresholds = roc_curve(test[:,1], best_scores)
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc

# Plot ROC curve
print "The best pen for Linear SVM is:",best_pen
print "The best score for Linear SVM is:",best_r
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
result=[best_scores,test[:,1]]
result=np.transpose(result)
c = csv.writer(open("linearsvm_result.csv", "wb"))
for line in result:
	c.writerow(line)
