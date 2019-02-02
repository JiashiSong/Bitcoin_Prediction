import numpy as np
from sklearn import svm
from sklearn import preprocessing
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import pyprind
import sys
from sklearn.externals import joblib


import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(12345)
clf_svm = svm.SVC(cache_size=2000, C=4000, gamma=1e-08)

clf_rf = RandomForestClassifier(min_samples_leaf=1, min_samples_split=15,
                                n_estimators=1000, bootstrap=False,
                                max_depth=100, max_features='auto')

# label_data = pd.read_csv("output/label.csv", header=None).values
# relative_dimport sysist_mat = pd.read_csv("output/relative_distance.csv", header=None).values
# state_data = pd.read_csv("output/state_data.csv", header=None).values
# officer_focus = pd.read_csv("output/officer_focus.csv", header=None).values

label_data = pd.read_csv("output/label.csv", header=None).values
relative_dist_mat = pd.read_csv("output/relative_distance.csv", header=None).values
state_data = pd.read_csv("output/state_data.csv", header=None).values
officer_focus = pd.read_csv("output/officer_focus.csv", header=None).values


Y = label_data.ravel()
min_max_scaler = preprocessing.MaxAbsScaler()
#relative_dist_mat = min_max_scaler.fit_transform(relative_dist_mat.astype(float))
#state_data = min_max_scaler.fit_transform(state_data.astype(float))
#officer_focus = min_max_scaler.fit_transform(officer_focus.astype(float))
X = np.concatenate((relative_dist_mat, state_data, officer_focus), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


clf_svm.fit(X_train, y_train)
clf_rf.fit(X_train, y_train)

y_pred_svm = clf_svm.predict(X_test)
y_pred_rf = clf_rf.predict(X_test)

print("The SVM accuracy is: {}".format(accuracy_score(y_test, y_pred_svm)))
print("The Random Forrest accuracy is: {}".format(accuracy_score(y_test, y_pred_rf)))

#joblib.dump(clf_svm, 'output/svm_model.pkl')
#joblib.dump(clf_rf, 'output/rf_model.pkl')


# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [9e-7, 1e-8, 5e-8, 2e-8],
#                      'C': [4000, 4100]}]
#
scores = ['accuracy']
#
#
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()
#
#     clf = GridSearchCV(svm.SVC(cache_size=4000), tuned_parameters, cv=5, n_jobs=10, scoring=score)
#     clf.fit(X_train, y_train)
#
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print()

n_estimators = [int(x) for x in np.linspace(start=1000, stop=2000, num=2)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(60, 100, num=5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [5, 10, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1]
# Method of selecting samples for training each tree
bootstrap = [False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

param_grid = {'bootstrap': [True], 'max_depth': [80, 100, 150], 'min_samples_leaf': [4, 5],
              'min_samples_split': [8, 10],
              'n_estimators': [2000, 4000]
            }
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               n_iter=10,
                               cv=3,
                               verbose=2,
                               n_jobs=11)

rf_random.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(rf_random.best_params_)
print("Detailed classification report:\n")
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.\n")
y_true, y_pred = y_test, rf_random.predict(X_test)
print(classification_report(y_true, y_pred))
print()


# for score in scores:
#     print("# Tuning hyper-parameters for %s in RF" % score)
#     print()
#
#     clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=10, scoring=score)
#     clf.fit(X_train, y_train)
#
#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print()

model = rf_random.best_estimator_


def plot_results(model, param='n_estimators', name='Num Trees'):
    param_name = 'param_%s' % param

    # Extract information from the cross validation model
    test_scores = model.cv_results_['mean_test_score']
    train_time = model.cv_results_['mean_fit_time']
    param_values = list(model.cv_results_[param_name])

    # Plot the scores over the parameter
    plt.subplots(1, 2, figsize=(10, 6))
    plt.subplot(121)

    plt.plot(param_values, test_scores, 'go-', label='test')
    plt.ylim(ymin=0, ymax=1)
    plt.legend()
    plt.xlabel(name)
    plt.ylabel('Accuracy')
    plt.title('Score vs %s' % name)

    plt.subplot(122)
    plt.plot(param_values, train_time, 'ro-')
    #plt.ylim(ymin=0.0, ymax=2.0)
    plt.xlabel(name)
    plt.ylabel('Train Time (sec)')
    plt.title('Training Time vs %s' % name)
    plt.tight_layout(pad=4)
    plt.show()

plt.style.use('fivethirtyeight')

feature_grid = {'max_depth': max_depth}
feature_grid_search = GridSearchCV(model, param_grid=feature_grid, cv=3, n_jobs=11, verbose=2,
                                   scoring='accuracy')
feature_grid_search.fit(X_train, y_train)

plot_results(feature_grid_search, param='max_depth', name='max_depth')

