import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

#Tap du lieu co ket qua la 0
dt0=pd.read_csv("dulieu0.csv", delimiter=",", header=0, na_values='?')
#Tap du lieu co ket qua la 1
dt1=pd.read_csv("dulieu1.csv", delimiter=",", header=0, na_values='?')

dt0=dt0.dropna()
dt1=dt1.dropna()
#print(dt0)
#print(dt1)
#Phan chia tap co ket qua la 0 thanh 5 phan va lay 1/5 de ghep voi tap co ket qua la 1
X1, X2 = train_test_split(dt0 , test_size = 0.6, random_state=10)

#print(X1)
#print(X2)
#Noi 2 tap du lieu lai
frames = [X2,dt1]
dt = pd.concat(frames)
#Trộn 2 tập dữ liệu đã phân chia
dt=dt.sample(frac=1,random_state=10).reset_index(drop=True)

print(dt)

X = dt.iloc[:,0:5]
y = dt.iloc[:,5:6]

#print(X)
#print(y)

sumdt = 0;
sumbn = 0;

for i in range(1,11):
    X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 1/3.0, random_state=10)
    #GINI
    # clf_gini = DecisionTreeClassifier(criterion = "gini",random_state =10, min_samples_leaf=5, max_depth=10)
    # clf_gini.fit(X_train,y_train)
    # y_pred = clf_gini.predict(X_test)
    #ENTROPY
    clf_entropy = DecisionTreeClassifier(criterion= "entropy",min_samples_leaf=2+i, max_depth=i)
    clf_entropy.fit(X_train,y_train)
    y_pred = clf_entropy.predict(X_test)

    from sklearn.metrics import accuracy_score
    print("DT Accuracy is ", round(accuracy_score(y_test, y_pred)*100,4))
    sumdt=sumdt+round(accuracy_score(y_test, y_pred)*100,4)

    my_maxtrix=confusion_matrix(y_test,y_pred)
    normalized_confusion_maxtrix=my_maxtrix/my_maxtrix.sum(axis=1,keepdims=True)*100
    # print(my_maxtrix)
    # print(normalized_confusion_maxtrix)

    model = GaussianNB()
    model.fit(X_train,np.ravel(y_train)) # hàm ravel chuyển cột thành mảng

    thucte = y_test
    dubao = model.predict(X_test)
    print("BN Accuracy is ", round(accuracy_score(thucte, dubao)*100,4))
    sumbn=sumbn+round(accuracy_score(thucte, dubao)*100,4)
sumdt=sumdt/10
sumbn=sumbn/10
print(sumdt)
print(sumbn)
