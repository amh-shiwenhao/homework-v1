def train_pre(train_data):
    from sklearn import tree
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score


    
    newdata = train_data
    newdata.isnull().any()
    train_path = newdata
    # ------- 数据处理 -------
    # 缺失值填补 (众数填补)
    train_path['gender'] = train_path['gender'].fillna('Male')
    train_path['user_group_id'] = train_path['user_group_id'].fillna(3)
    train_path['age_level'] = train_path['age_level'].fillna(3)
    train_path['user_depth'] = train_path['user_depth'].fillna(3)
    train_path['city_development_index'] = train_path['city_development_index'].fillna(2)

    # 哑变量编码
    from sklearn.preprocessing import LabelEncoder
    train_path.gender = LabelEncoder().fit_transform(np.array(train_path.gender))
    return train_path
import sklearn
import numpy as np
import pandas as pd
train_data = pd.read_csv(r'D:\homework\python data\Ad_click_prediction_train.csv')
pre_data = train_pre(train_data)
print(pre_data)


