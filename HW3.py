from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np
import random
import csv

def readcsv(filename):	
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=",")
    rownum = 0	
    a = []
    for row in reader:
        a.append (row)
        rownum += 1
    ifile.close()
    return a

def getIris():
	iris=readcsv('iris.csv')
	random.shuffle(iris)
	d=[]
	t=[]
	for x in iris:
		tmp=[]
		for i in range(0,4):
			tmp.append(x[i])
		d.append(tmp)
		t.append(x[4])
	return np.array(d).astype(float),np.array(t)

def dateToNum(date):
	dates=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
	for i in range(0,12):
		if dates[i] == date:
			return i+1

def dayToNum(day):
	days=['mon','tue','wed','thu','fri','sat','sun']
	for i in range(0,7):
		if days[i] == day:
			return i+1

def classifyFF(x):
	if x == 0:
		return '0'
	elif x >= 100:
		return '>=100'
	elif x >= 10:
		return'>=10'
	else:
		return '>=1'

def getForestFires():
	FF=readcsv('forestfires.csv')
	random.shuffle(FF)
	d=[]
	t=[]
	for x in FF:
		tmp=[]
		for i in range(0,12):
			if i==2:
				tmp.append(dateToNum(x[i]))
			elif i==3:
				tmp.append(dayToNum(x[i]))
			else:
				tmp.append(x[i])
		d.append(tmp)
		t.append(classifyFF(float(x[12])))
	return np.array(d).astype(float),np.array(t)

def scoring(clf,data,target):
	X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
	clf.fit(X_train,y_train)
	pre=clf.predict(X_test)
	print(classification_report(y_test, pre, target_names=sorted(set(target))))
	return accuracy_score(y_test,pre)

def DecisionTree((data,target)):
	clf = tree.DecisionTreeClassifier()
	return scoring(clf,data,target)

def Knn((data,target)):
	clf=KNeighborsClassifier(n_neighbors=10)
	return scoring(clf,data,target)

def NaiveBayes((data,target)):
	clf = GaussianNB()
	return scoring(clf,data,target)




print("Decision Tree for Iris: %.2f %%" %(DecisionTree(getIris())*100))
print "\n================================================================\n"
print("KNN for Iris: %.2f %%" %(Knn(getIris())*100))
print "\n================================================================\n"
print("NaiveBayes for Iris: %.2f %%" %(NaiveBayes(getIris())*100))
print "\n================================================================\n"
print("Decision Tree for Forest Fires: %.2f %%" %(DecisionTree(getForestFires())*100))
print "\n================================================================\n"
print("KNN for Forest Fires: %.2f %%" %(Knn(getForestFires())*100))
print "\n================================================================\n"
print("NaiveBayes for Forest Fires: %.2f %%" %(NaiveBayes(getForestFires())*100))
print "\n================================================================\n"
