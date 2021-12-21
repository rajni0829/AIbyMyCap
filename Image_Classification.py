import sys
import scipy
import matplotlib
import pandas as pd
import sklearn
print(sys.version)
print(scipy.__version__)
print(matplotlib.__version__)

from pandas.plotting import scatter_matrix
from matplotlib import pyplot as pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

url = 'https://gist.github.com/curran/a08a1080b88344b0c8a7'
# data = pd.read_csv('https://github.com/jbrownlee/Datasets/blob/master/iris.csv')

names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = pd.read_csv('/content/Iris.csv')

print(dataset)

si = dataset.groupby('Iris-setosa').size()
print(si)

dataset.plot(kind='box',subplots='True',layout=(5,5),sharex=False,sharey=False)
pyplot.show()

dataset.hist()
pyplot.show()

#multivariate plots - iteraction bw var
scatter_matrix(dataset)
pyplot.show()

#creating a validation set
#splitting dtset
array = dataset.values
X = array[:, 0:4]
y = array[:,4]
X_train,X_validation,Y_train,Y_validation = train_test_split(X,y,test_size=0.2,random_state=1)

#building models
models = []
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

from scipy.sparse.construct import random
# evaluate created models
results = []
names = []
for name,model in models:
  kfold = StratifiedKFold(n_splits=10,random_state=None)
  cv_results = cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  print(name,cv_results.mean(),cv_results.std())

  # compare models
pyplot.boxplot(results,labels=names)
pyplot.title('Algo Comparison')
pyplot.show()

model = SVC(gamma='auto')
model.fit(X_train,Y_train)
predictions = model.predict(X_validation)

print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))

