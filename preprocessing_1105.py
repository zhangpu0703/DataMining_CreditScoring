#import pandas as pd 
import csv
import random
import numpy as np 
import numpy.ma as ma
import matplotlib.pyplot as plt
#data=pd.DataFrame(np.genfromtxt("cs-training.csv",dtype=None,delimiter=','))
data=np.genfromtxt("rawdata.csv",delimiter=',')
print np.shape(data)
#data.fillna(data.mean())
#data.dropna(axis=0,how="any")
#print np.shape(data)
#print data
data_class0=data[data[:,1]==0]
data_class1=data[data[:,1]==1]
mean_0 = ma.masked_values(data_class0,-9999)[:,6].mean()
mean_1 =ma.masked_values(data_class1,-9999)[:,6].mean()
for i in range(np.shape(data)[0]):
	if (data[i,1]==0) & (data[i,6]==-9999):
		data[i,6]=mean_0
	elif (data[i,1]==1) & (data[i,6]==-9999):
		data[i,6]=mean_1


m_data=ma.masked_outside(data[:,5],-0.001,1.001)
data[:,5]=m_data.filled(m_data.mean())
m_data=ma.masked_outside(data[:,5],-0.001,20)
data[:,4]=m_data.filled(m_data.mean())


def data_expansion(data,number_new):
	expand_data=np.append(data,np.zeros((np.shape(data)[0],number_new)),axis=1)
	print np.shape(expand_data)
	for row in range(np.shape(data)[0]):
		if data[row,3]>60:
			expand_data[row,np.shape(data)[1]]=1.0
		# generate mortgage
		expand_data[row,np.shape(data)[1]+1]=data[row,5]*data[row,6]*0.33
		# generage mortgage and other debts
		expand_data[row,np.shape(data)[1]+2]=data[row,5]*data[row,6]*0.43
		# generate late rate
		if data[row,7]!=0:
			expand_data[row,np.shape(data)[1]+3]=data[row,8]/data[row,7]
		# generate monthly savings
		expand_data[row,np.shape(data)[1]+4]=data[row,6]*(1-data[row,5])
		# generate average income
		expand_data[row,np.shape(data)[1]+5]=data[row,6]/(data[row,11]+1)
		# generate prior credit score given different age groups from other source
		if data[row,3]<29:
			expand_data[row,np.shape(data)[1]+6]=637
		elif data[row,3]<39: expand_data[row,np.shape(data)[1]+6]=654
		elif data[row,3]<49: expand_data[row,np.shape(data)[1]+6]=675
		elif data[row,3]<59: expand_data[row,np.shape(data)[1]+6]=697
		elif data[row,3]<69: expand_data[row,np.shape(data)[1]+6]=722
		else: expand_data[row,np.shape(data)[1]+6]=747
	return expand_data

expand_input=data_expansion(data,7)


def remove_out(data):
	n_drop=[]
	remaining=np.shape(data)[0]
	for col in [3,6]:
		feature=data[:,col]
		drop=[]
		#print col
		iqr = np.percentile(feature, 75) - np.percentile(feature, 25)
		lcl=np.percentile(feature, 25) - 1.5 * iqr
		ucl=np.percentile(feature, 75) + 1.5 * iqr
		data=data[(feature<ucl) & (feature>lcl)]
		n_drop.append(-np.shape(data)[0]+remaining)
		remaining=np.shape(data)[0]
		#data=np.delete(data(abs(data - np.mean(data)) > m * np.std(data)),drop)
		'''
		for row in range(np.shape(data)[0]): 
			if abs(feature[row]-np.mean(feature))>m*np.std(feature):
				drop.append(row)
		data=np.delete(data,drop)
		'''
	return data,n_drop

pre_processed_data,n_drop=remove_out(expand_input)
print np.shape(pre_processed_data)
print "Number of Observations Dropped for Each Feature is: ", n_drop

#print pre_processed_data[5,]

'''
VAR=['Age','Monthly Income/$','Mortage/$','All Debts/$','Monthly Savings/$','Average Income/$']
col=[3,6]
for i in range(len(col)):
	plt.figure(i+1)
	plt.boxplot(expand_input[:,col[i]])
	plt.title(VAR[i])
	plt.show()
'''
def split_train(prob,data):
	train_index=[]
	test_index=[]
	for i in range(np.shape(data)[0]):
		accept=random.random()
		if accept<prob:
			train_index.append(i)
		else: test_index.append(i)
	train=data[train_index,]
	test=data[test_index,]
	return train, test

train,test=split_train(0.7,pre_processed_data)


total_data=pre_processed_data.tolist()
train_data=train.tolist()
test_data=test.tolist()

c = csv.writer(open("data.csv", "wb"))
for line in total_data:
	c.writerow(line)

c = csv.writer(open("training.csv", "wb"))
for line in train_data:
	c.writerow(line)

c = csv.writer(open("testing.csv", "wb"))
for line in test_data:
	c.writerow(line)