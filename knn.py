import numpy as np
import pylab as pl
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from time import clock as now
import csv


train=np.genfromtxt("training.csv",dtype=None,delimiter=',')
test=np.genfromtxt("testing.csv",dtype=None,delimiter=',')

train[:,2:18]=(train[:,2:18] - train[:,2:18].mean(axis=0)) / train[:,2:18].std(axis=0)
test[:,2:18]=(test[:,2:18] - test[:,2:18].mean(axis=0)) / test[:,2:18].std(axis=0)
X=train[:,2:18]
y=train[:,1]
train_indx=train[:,0]
T=test[:,2:18]
test_target=test[:,1]
test_indx=test[:,0]
mini_train=train[0:19999,:]
mini_test=test[0:9999,:]
best_r=0.0
Time=[]
ROC=[]
for n in [150]:
	neigh = KNeighborsClassifier(n_neighbors=n)
	start=now()
	neigh.fit(train[:,2:12], train[:,1])
	finish=now()
	Time.append(finish-start)
	predict=neigh.predict_proba(test[:,2:12])
	fpr, tpr, thresholds = roc_curve(test[:,1], predict[:,1])
	roc_auc = auc(fpr, tpr)
	ROC.append(roc_auc)
	print roc_auc
	if best_r<roc_auc:
		best_r=roc_auc
		best_neigh=neigh

best_neigh.fit(X, y)
predict=best_neigh.predict_proba(T)
fpr, tpr, thresholds = roc_curve(test[:,1], predict[:,1])
roc_auc = auc(fpr, tpr)
print "Time spend:", Time
print "ROC's:", ROC
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
result=[predict[:,1],test[:,1]]
result=np.transpose(result)
c = csv.writer(open("knn_result.csv", "wb"))
for line in result:
	c.writerow(line)

