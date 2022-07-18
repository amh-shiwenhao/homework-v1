import sklearn
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

newdata = pd.read_csv(r'D:\homework\python data\Ad_click_prediction_train.csv')
newdata.isnull().any()
train_path = newdata
# ------- 数据处理 -------
# 缺失值填补 (众数填补)
train_path['gender'] = train_path['gender'].fillna('Male')
train_path['user_group_id'] = train_path['user_group_id'].fillna(3)
train_path['age_level'] = train_path['age_level'].fillna(3)
train_path['user_depth'] = train_path['user_depth'].fillna(3)
train_path['city_development_index'] = train_path['city_development_index'].fillna(-100)

# 哑变量编码
from sklearn.preprocessing import LabelEncoder
train_path.gender = LabelEncoder().fit_transform(np.array(train_path.gender))
train_path.user_group_id = LabelEncoder().fit_transform(np.array(train_path.user_group_id))
train_path.product_category_1 = LabelEncoder().fit_transform(np.array(train_path.product_category_1))


# 选取变量
features_columns = ['product_category_1','user_group_id','gender','age_level','user_depth','city_development_index','var_1']
train = train_path[features_columns]
target = train_path['is_click']
print(train.info())

# 
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.3,random_state=21)

# model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
auc = roc_auc_score(y_test,clf.predict(X_test))

#run_dir = context['run_dir']
#model_path = context['run_dir'] + '/clf.pmml'
metrics_result = {'auc': auc, 'acc': acc}

# 输出
#context.set_output('model_path', model_path)
# context.set_output('metrics', metrics_result)
print(metrics_result)

gbdt = GradientBoostingClassifier()
gbdt = gbdt.fit(X_train, y_train)

acc = gbdt.score(X_test, y_test)
print(acc)