import time
import cifar10
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras import datasets
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

start = time.time()
data = datasets.cifar10.load_data()
train, test = data

x_train_image , y_train = train
x_test_image, y_test = test
x_train_image.shape, y_train.shape, x_test_image.shape, y_test.shape

#reshape data
x_train = x_train_image.reshape(50000, 32*32*3)
x_test = x_test_image.reshape(10000, 32*32*3)
y_train = y_train.reshape(50000)
y_test = y_test.reshape(10000)

fig = plt.figure(figsize = (8, 8))
for i in range(64):
    ax = fig.add_subplot(8, 8, i+1)
    ax.imshow(x_train_image[i])
plt.show()

pca = PCA()
pca.fit(x_train)

k = 0
curr = 0
total = sum(pca.explained_variance_)
while curr/total < .99:
    curr += pca.explained_variance_[k]
    k+=1
k

pca = PCA(n_components = k)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
x_test_pca.shape, x_train_pca.shape

model1 = svm.SVC()
model1.fit(x_train_pca, y_train)
y_pred1 = model1.predict(x_test_pca)
score1 = model1.score(x_test_pca, y_test)
score1

print(classification_report(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))

model2 = RandomForestClassifier()
model2.fit(x_train_pca, y_train)
y_pred2 = model2.predict(x_test_pca)
score2 = model2.score(x_test_pca, y_test)
score2

model3 = LinearRegression()
model3.fit(x_train_pca, y_train)
y_pred3 = model3.predict(x_test_pca)
score3 = model3.score(x_test_pca, y_test)
score3

def change(i):
    classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    return classes[i]
score = max(score1, score2, score3)
if score == score1:
    print(1)
    y_pred = y_pred1
elif score == score3:
    print(3)
    y_pred = y_pred3
else:
    print(2)
    y_pred = y_pred2
    
df = pd.DataFrame(y_pred, columns = ["pred"])
df["pred"] = df["pred"].apply(change)

df.head(5)
end = time.time()

print(end - start)
