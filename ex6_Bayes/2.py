from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ex1 import clustering_performance
from ex6_Bayes.ex import EmailFeatureGeneration as Email

X, Y = Email.Text2Vector()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=22)
# print("X_train.shape =", X_train.shape)
# print("X_test.shape =", X_test.shape)



# 朴素贝叶斯
clf = GaussianNB()
clf.fit(X_train, y_train)
y_sample_bayes = clf.predict(X_test)
Bayes_ACC = clustering_performance.clusteringMetrics(y_test, y_sample_bayes)
print("Bayes_ACC =", Bayes_ACC)

fig = plt.figure()
plt.subplot(121)
plt.title('Bayes')
confusion = confusion_matrix(y_sample_bayes, y_test)
confusion = confusion/X_test.shape[0]
# print(confusion)
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='.3g')
plt.xlabel('Predicted label')
plt.ylabel('True label')

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_sample_knn = knn.predict(X_test)
KNN_ACC = clustering_performance.clusteringMetrics(y_test, y_sample_knn)
print("KNN_ACC =", KNN_ACC)

plt.subplot(122)
plt.title('KNN')
confusion = confusion_matrix(y_sample_knn, y_test)
confusion = confusion/X_test.shape[0]
sns.heatmap(confusion, annot=True, cmap='YlGn', fmt='.3g')
plt.xlabel('Predicted label')

plt.show()