import numpy as np
from reduction.adult import preprocess_adult_data
from reduction.exponential_gradient import tranform_onehot, reduction_exponential_gradient_equalized_odds, predict
from reduction.exponential_gradient import reduction_exponential_gradient, reduction_exponential_gradient_v2
from sklearn.ensemble import RandomForestClassifier
from reduction.metrics import EO_metrics, DP_metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import json


seed = 0
dataset_orig_train, dataset_orig_test = preprocess_adult_data(seed = seed, 
                                                              path = "/Users/smaity/projects/missing_sensitive_attr/dataset/adult")

x_train, a_train = dataset_orig_train.features[:, :39], dataset_orig_train.features[:, 39:]
x_test, a_test = dataset_orig_test.features[:, :39], dataset_orig_test.features[:, 39:]
y_train, y_test = dataset_orig_train.labels.reshape((-1,)), dataset_orig_test.labels.reshape((-1,))


X, y, a = x_train, y_train, a_train

a_onehot_train = tranform_onehot(a_train)
a_onehot_test = tranform_onehot(a_test)

results = {}


## Equalized odds stable

base_estimator = DecisionTreeClassifier(max_depth=10)
estimators_, estimator_index = reduction_exponential_gradient_v2(X, y, a_onehot_train, base_estimator, eps = 0.02, 
                                                                  eta = 0.5, B = 10, verbose = 2, onehot_protected=True, 
                                                                  constraint="EO", n_estimators=25)

print("All sensitive attributes..")
y_pred = predict(x_test, estimators_, estimator_index)
print("\nFor gender")
bal_acc, eod_sex = EO_metrics(y_test, y_pred, a_test[:, 0])
print("\nFor race")
bal_acc, eod_race =EO_metrics(y_test, y_pred, a_test[:, 1])
results["EO-stable"] = dict(bal_acc = bal_acc, eod_sex = eod_sex, eod_race = eod_race)




base_estimator = DecisionTreeClassifier(max_depth=10)
estimators_, estimator_index = reduction_exponential_gradient(X, y, a_onehot_train, base_estimator, eps = 0.02, 
                                                                  eta = 0.5, B = 10, verbose = 2, onehot_protected=True, 
                                                                  constraint="EO")


print("All sensitive attributes..")
y_pred = predict(x_test, estimators_, estimator_index)
print("\nFor gender")
bal_acc, eod_sex = EO_metrics(y_test, y_pred, a_test[:, 0])
print("\nFor race")
bal_acc, eod_race =EO_metrics(y_test, y_pred, a_test[:, 1])
results["EO"] = dict(bal_acc = bal_acc, eod_sex = eod_sex, eod_race = eod_race)

RF = RandomForestClassifier(class_weight="balanced", max_depth=10, n_estimators=100).fit(X, y)
y_pred = RF.predict(x_test)
print("\n\nwithout fairness...")
print("\nFor gender")
bal_acc, eod_sex = EO_metrics(y_test, y_pred, a_test[:, 0])
print("\nFor race")
bal_acc, eod_race =EO_metrics(y_test, y_pred, a_test[:, 1])

results["without-fairness"] = dict(bal_acc = bal_acc, eod_sex = eod_sex, eod_race = eod_race)


print(json.dumps(results, indent=3))
