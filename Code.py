import sklearn
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.ensemble import VotingClassifier
import pickle as pkl

train = pd.read_csv('Lab3_train.csv')
train_set = train.values
labels = train_set[:, 0]
train_data = train_set[:, 1: 25]
scaler = MinMaxScaler((0, 1))
selectors = VarianceThreshold()
train_data = selectors.fit_transform(train_data)
train_data = scaler.fit_transform(train_data)
Q1 = []
score = []

para_svm = [
    {'kernel': ['rbf', 'poly', 'sigmoid'],
     'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
]
t_svm = time()
svm = SVC(probability=True)
svm_grid_search = GridSearchCV(svm, para_svm)
svm_grid_search.fit(train_data, labels)
svm_clf = svm_grid_search.best_estimator_
svm_pred = svm_clf.predict(train_data)
svm_acc = format(accuracy_score(labels, svm_pred), '.2f')
Q1.append(('svm', svm_grid_search.best_params_, svm_acc))
score.append(format(svm_grid_search.best_score_, '.2f'))
print("SVM time cost is", format(time()-t_svm, '.2f'), "second.\n")

para_dt = [
    {'max_depth': [i for i in range(1, 100)]}
]
t_dt = time()
dt = tree.DecisionTreeClassifier()
dt_grid_search = GridSearchCV(dt, para_dt)
dt_grid_search.fit(train_data, labels)
dt_clf = dt_grid_search.best_estimator_
dt_pred = dt_clf.predict(train_data)
dt_acc = format(accuracy_score(labels, dt_pred), '.2f')
Q1.append(('decision tree', dt_grid_search.best_params_, dt_acc))
score.append(format(dt_grid_search.best_score_, '.2f'))
print("Decision Tree time cost is", format(time()-t_dt, '.2f'), "second.\n")

para_knn = [
    {'n_neighbors': [i for i in range(1, 11)]}
]
t_knn = time()
knn = KNeighborsClassifier()
knn_grid_search = GridSearchCV(knn, para_knn)
knn_grid_search.fit(train_data, labels)
knn_clf = knn_grid_search.best_estimator_
knn_pred = knn_clf.predict(train_data)
knn_acc = format(accuracy_score(labels, knn_pred), '.2f')
Q1.append(('k-nn', knn_grid_search.best_params_, knn_acc))
score.append(format(knn_grid_search.best_score_, '.2f'))
print("K-nn time cost is", format(time()-t_knn, '.2f'), "second.\n")

para_mlp = [
    {'hidden_layer_sizes': [(3, 5)],
     'solver': ['lbfgs', 'sgd', 'adam'],
     'max_iter': [10000]}
]
t_mlp = time()
mlp = MLPClassifier()
mlp_grid_search = GridSearchCV(mlp, para_mlp)
mlp_grid_search.fit(train_data, labels)
mlp_clf = mlp_grid_search.best_estimator_
mlp_pred = mlp_clf.predict(train_data)
mlp_acc = format(accuracy_score(labels, mlp_pred), '.2f')
Q1.append(('mlp', mlp_grid_search.best_params_, mlp_acc))
score.append(format(mlp_grid_search.best_score_, '.2f'))
print("MLP time cost is", format(time()-t_mlp, '.2f'), "second.\n")

t_ensemble = time()
ensemble_cls = VotingClassifier(
    estimators=[('svm', svm_clf), ('dt', dt_clf), ('knn', knn_clf), ('mlp', mlp_clf)],
    voting='soft')
ensemble_cls.fit(train_data, labels)
print("Ensemble_cls time cost is", format(time()-t_ensemble, '.2f'), "second.\n")
score1 = ensemble_cls.score(train_data, labels)
score.append(format(score1, '.2f'))
svm_prob = svm_clf.fit(train_data, labels).predict_proba(train_data)
dt_prob = dt_clf.fit(train_data, labels).predict_proba(train_data)
knn_prob = knn_clf.fit(train_data, labels).predict_proba(train_data)
mlp_prob = mlp_clf.fit(train_data, labels).predict_proba(train_data)

Q2 = []
for i in range (0, len(labels)):
    mean = format(
        (max((svm_prob[i][0] + dt_prob[i][0] + knn_prob[i][0] + mlp_prob[i][0])/4,
             (svm_prob[i][1] + dt_prob[i][1] + knn_prob[i][1] + mlp_prob[i][1])/4)), '.2f')
    maximum = format(
        (max(max(svm_prob[i][0], dt_prob[i][0], knn_prob[i][0], mlp_prob[i][0]),
             max(svm_prob[i][1], dt_prob[i][1], knn_prob[i][1], mlp_prob[i][1]))), '.2f')
    Q2.append([('mean', mean), ('max', maximum)])

test = pd.read_csv('Lab3_test.csv')
test_set = test.values
test_data = test_set[:, 1: 25]
test_data = selectors.fit_transform(test_data)
test_data = scaler.fit_transform(test_data)

choice = score.index(max(score))
if choice == 0:
    print('Best classifier is svm!')
    Q3 = svm_clf.predict(test_data)
elif choice == 1:
    print('Best classifier is decision tree!')
    Q3 = dt_clf.predict(test_data)
elif choice == 2:
    print('Best classifier is K-nn!')
    Q3 = knn_clf.predict(test_data)
elif choice == 3:
    print('Best classifier is MLP!')
    Q3 = mlp_clf.predict(test_data)
elif choice == 4:
    print('Best classifier is ensemble classifier!')
    Q3 = ensemble_cls.predict(test_data)

f = open('Lab03_201930630330_wangzihan.pkl', 'wb')
pkl.dump({'Q1': Q1}, f)
pkl.dump({'Q2': Q2}, f)
pkl.dump({'Q3': Q3}, f)
f.close()
