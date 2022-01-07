@@ -0,0 +1,25 @@
import pickle
from sklearn import svm ,metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#loading our clf for OCR\
filename1 = "svm.pkl"
filename2 = 'knn.pkl'
filename3 = 'rf.pkl'

# testing file
x_test = []
y_test = []

# model loading
svm_model = pickle.load(open(filename1, 'rb'))
knn_model = pickle.load(open(filename2, 'rb'))
rf_model = pickle.load(open(filename3, 'rb'))

print(svm_model)
# # results
# svm_result = svm_model.score(X_test, Y_test)
# knn_result = knn_model.score(X_test, Y_test)
# rf_model = rf_model.score(X_test, Y_test)

# print('svm: ',svm_result,'knn: ',knn_result,'rf: ',rf_model)