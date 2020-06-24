import pandas as pd
import numpy as np

data = pd.read_csv("turnover.csv")
# print(data.head())

np.array(['low','medium','high'], dtype = object)
# print(data.salary.unique())

data.salary = data.salary.astype('category')
data.salary = data.salary.cat.reorder_categories(['low', 'medium', 'high'])
data.salary = data.salary.cat.codes
# print(data.salary)

departments = pd.get_dummies(data.department)
departments = departments.drop("technical",axis=1)
data = data.drop("department",axis=1)
data = data.join(departments)
print(departments.head())
# e_employees = len(data)
# print(data.churn.value_counts())
# print(data.churn.value_counts()/e_employees*100)



# gini_A = 0.65
# gini_B = 0.15
# if gini_A < gini_B:
#     print("split by A!")
# else:
#     print("split by B!")


from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn.tree import export_graphviz

target = data.churn

features = data.drop("churn",axis=1)

target_train,target_test,features_train,features_test = train_test_split(
    target,features,test_size=0.25)

# model = DecisionTreeClassifier(random_state=42)
# model.fit(features_train,target_train)
# model.score(features_test,target_test)*100
# model.score(features_train,target_train)*100

# model_depth_5 = DecisionTreeClassifier(max_depth=5,random_state=42)
# model_depth_5.fit(features_train,target_train)
#
# print(model_depth_5.score(features_train,target_train)*100)
# print(model_depth_5.score(features_test,target_test)*100)


model_sample_100 = DecisionTreeClassifier(min_samples_leaf=100,random_state=42)
model_sample_100.fit(features_train,target_train)
print(model_sample_100.score(features_train,target_train)*100)
print(model_sample_100.score(features_test,target_test)*100)


# export_graphviz(model_sample_100,"tree1.dot")

from sklearn.metrics import precision_score
prediction = model_sample_100.predict(features_test)
print(precision_score(target_test,prediction)*100)

from sklearn.metrics import recall_score
prediction = model_sample_100.predict(features_test)
print(recall_score(target_test,prediction)*100)

from sklearn.metrics import roc_auc_score
prediction = model_sample_100.predict(features_test)
print(roc_auc_score(target_test,prediction)*100)

model_depth_5_b =DecisionTreeClassifier(max_depth=5,class_weight="balanced",
                                        random_state=42)
model_depth_5_b.fit(features_train,target_train)
print(model_depth_5_b.score(features_test,target_test)*100)


# from sklearn.model_selection import GridSearchCV
#
# depth = [i for i in range(5,21,1)]
# samples = [i for i in range(50,500,50)]
# parameters = dict(max_depth = depth,min_samples_eaf=samples)
#
# param_search = GridSearchCV(model_depth_5_b,parameters)
# param_search.fit(features_train, target_train)
# print(param_search.best_params_)

features_importances = model_sample_100.feature_importances_
feature_list = list(features)
relative_importances = pd.DataFrame(index=feature_list,data=features_importances,columns=["importance"])
print(relative_importances.sort_values(by="importance",ascending=False))


selected_features = relative_importances[relative_importances.importance>0.01]
selected_list = selected_features.index
features_train_selected = features_train[selected_list]
features_test_selected = features_test[selected_list]
# print(features_train_selected)
# print(features_test_selected)

model_sample_100 = DecisionTreeClassifier(max_depth=9,min_samples_leaf=150,class_weight="balanced",random_state=42)
model_sample_100.fit(features_train_selected,target_train)
prediction_sample_100 = model_sample_100.predict(features_test_selected)
print(model_sample_100.score(features_test_selected,target_test)*100)
print(recall_score(target_test,prediction_sample_100)*100)
print(roc_auc_score(target_test,prediction_sample_100)*100)













