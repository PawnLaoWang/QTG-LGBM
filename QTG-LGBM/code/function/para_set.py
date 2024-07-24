import numpy as np
import optuna
import pandas as pd
import random
import re
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

def clean_feature_name(name):
    name = re.sub(r'[^\w]', '_', name)
    name = name.strip('_')
    return name

def cross_validation(df, train_set, clf, scale):
    k_numb_iter=50
    ROC_all_auc=[]
    neg_numb_iter=50
    for _ in range(0, k_numb_iter):
        skf=KFold(n_splits=5,shuffle=True, random_state=20)
        ROC_auc_mean=[]
        for _, (train_cv, test_cv) in enumerate(skf.split(train_set)): 
            for i in range(0, neg_numb_iter):
                train_data = train_set.iloc[train_cv] 
                test_data = train_set.iloc[test_cv] 
                training_negative = random.sample(list(df[df['class']==0].index), int(len(train_data)*scale))
                testing_negatives = random.sample(list(df[df['class']==0].index), len(test_cv)*scale)
                train_data=train_data.append(df.iloc[training_negative]) 
                test_data=test_data.append(df.iloc[testing_negatives])
                train_feature=train_data.drop(['class'], axis=1)
                test_feature=test_data.drop(['class'], axis=1)
                probas_=clf.fit(train_feature, train_data['class']).predict_proba(test_feature)
                fpr, tpr1, _ = roc_curve(test_data['class'], probas_[:, 1])
                ROC_auc_mean.append(auc(fpr, tpr1)) # AUC-ROC
        ROC_all_auc.append(np.mean(ROC_auc_mean))
    return np.mean(ROC_all_auc)

# 定义优化目标函数
def objective(trial, df, train_set):
    # 定义超参数搜索空间
    params = {
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
        'num_leaves': trial.suggest_int('num_leaves', 7, 31),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 30, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
        'is_unbalance': trial.suggest_categorical('is_unbalance', [True]),
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 20
    }
    clf = lgb.LGBMClassifier(**params)
    scale = trial.suggest_int('scale', 10, 100, step=10)
    ROCs = cross_validation(df, train_set, clf, scale)
    return ROCs

def para_set(dt):
    df = pd.read_csv(dt)
    df.columns = [clean_feature_name(col) for col in df.columns]
    df = df.drop(['ID'], axis=1)
    df = df.dropna(axis=1,how='all')
    train_set = df[df['class']==1]
    
    study = optuna.create_study(direction='maximize')
    objective_with_data = lambda trial: objective(trial, df, train_set)
    study.optimize(objective_with_data, n_trials=50)
    best_params = study.best_params
    best_clf = lgb.LGBMClassifier(**best_params)
    best_scale = best_params['scale']

    return best_clf, best_scale
