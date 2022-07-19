import os
import kflearn as kfl
import train_lgb

# os.environ['kfl_activate_component'] = 'train_node'
if __name__ == '__main__':
    # 获取上下文
    context = kfl.get_context_v2()
    # 获取本节点输入及配置
    train_path = context.get_input('train_data')
    test_path = context.get_input('valid_data')
    run_dir = context['run_dir']
    sep = context.get_config('sep')
    print(sep)

    # 预处理
    pre_df_train = pre.pret(path=train_path, sep=sep)
    pre_df_test = pre.pret(path=test_path, sep=sep)

    def train_pre(train_data):
        import sklearn
        import numpy as np
        import pandas as pd
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
        train_path.user_group_id = LabelEncoder().fit_transform(np.array(train_path.user_group_id))
        
        return train_path

    # train_data = pd.read_csv(r'D:\homework\python data\Ad_click_prediction_train.csv')
    train_path = context.get_input('train_data')
    test_path = context.get_input('test_data')
    train_pre_data = train_pre(train_path)
    test_pre_data = train_pre(test_path)

    # 设置输出
    context.set_output('train_pre_data', train_pre_data)
    context.set_output('test_pre_data', test_pre_data)


    # 序列化（必填）
    context.dump()
