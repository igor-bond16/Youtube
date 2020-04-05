#from sklearn.datasets import load_iris
#import numpy as np
from sklearn import tree

#iris = load_iris()
#test_idx = [0,50,100]
features = [[237,7.8],[241,7.8],[241,7.9],[200,7.8],[207,7.8],[197,7.3]]
labels = [1,1,1,0,0,0]
#train_target = np.delete(iris.target,test_idx)
#train_data = np.delete(iris.data,test_idx,axis=0)
test_data = [[220,8]]
#test_target = iris.target[test_idx]
#test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)

#print(test_target)
print(clf.predict(test_data))
