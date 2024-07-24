import numpy as np
import pandas as pd
import random
import re
import time
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score
from scipy.interpolate import interp1d
from para_set import para_set
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dt', default='', type=str)   # dataset
parser.add_argument('--trait', default='', type=str)  # phenotype:PH, FT and TBN
parser.add_argument('--writer_path', default='', type=str)  # path to save
args = parser.parse_args()

dt = args.dt
trait = args.trait
writer_path = args.writer_path

start_time = time.time()
def cross_validation(df, train_set):
    k_numb_iter=50
    if trait == 'PH':
        clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=15, reg_alpha=0.0, reg_lambda=1,
                max_depth=9, n_estimators=50, objective='binary',
                subsample=1, colsample_bytree=1, subsample_freq=1,is_unbalance=True,
                learning_rate=0.09, min_child_weight=1, random_state=20)
        scale = 30
    elif trait == 'FT':
        clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=15, reg_alpha=0.0, reg_lambda=1,
                max_depth=9, n_estimators=50, objective='binary',
                subsample=1, colsample_bytree=1, subsample_freq=1,is_unbalance=True,
                learning_rate=0.07, min_child_weight=1, random_state=20)
        scale = 130
    elif trait == 'TBN':
        clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=15, reg_alpha=0.0, reg_lambda=1,
                max_depth=9, n_estimators=50, objective='binary',
                subsample=1, colsample_bytree=1, subsample_freq=1,is_unbalance=True,
                learning_rate=0.07, min_child_weight=1, random_state=20)
        scale = 120
    else:
        clf, scale = para_set(dt)

    all_mean_tpr1 = [float()]*k_numb_iter
    all_mean_prec = [float()]*k_numb_iter
    P_R_all_auc = []
    ROC_all_auc=[]
    ACC_all_mean=[]
    fprs = []
    tprs = []
    precisions = []
    recalls = []
    F1_all=[]
    
    if  k_numb_iter > 1:
        shuffle_switch=True
    else: 
        shuffle_switch=False
        
    neg_numb_iter=50 # iteration for randomly select negatives and re-train model
    for n in range(0, k_numb_iter): #  re-spliting causal gene list for cross-validation
        skf=KFold(n_splits=5,shuffle=shuffle_switch, random_state=20) # five fold cross-validatoin
        P_R_auc_mean=[]
        ROC_auc_mean=[]
        F1_sum=0
        fold_mean_prec = [0.0]*101 
        fold_mean_tpr1 = [0.0]*101 
        temp_actual_label=[]
        temp_predicted_class=[]
        ACC_mean=[]
        fprs_mean = []
        tprs_mean = []
        precisions_mean = []
        recalls_mean = []
        fold_mean_tpr1 = [0.0]*101 
        for m, (train_cv, test_cv) in enumerate(skf.split(train_set)): 
            fold_mean_fpr = np.linspace(0, 1, 101) # set fpr grid for ROC
            for i in range(0, neg_numb_iter): # iterate for randomly selecting negatives. 
                train_data = train_set.iloc[train_cv] 
                test_data = train_set.iloc[test_cv] 
                training_negative = random.sample(list(df[df['class']==0].index), int(len(train_data)*scale))
                testing_negatives = random.sample(list(df[df['class']==0].index), len(test_cv)*scale)
                train_data=train_data.append(df.iloc[training_negative]) 
                test_data=test_data.append(df.iloc[testing_negatives])
                train_feature=train_data.drop(['class'], axis=1)
                test_feature=test_data.drop(['class'], axis=1)
                probas_=clf.fit(train_feature, train_data['class']).predict_proba(test_feature) #  Probability of falling into a class
                predicted_class=clf.fit(train_feature, train_data['class']).predict(test_feature)  
                acc = accuracy_score(test_data['class'], predicted_class)
                ACC_mean.append(acc)
                prec, tpr, _ = precision_recall_curve(test_data['class'], probas_[:, 1]) # precision, recall,  associated threshhold 
                fpr, tpr1, _ = roc_curve(test_data['class'], probas_[:, 1]) # false positve rate, true positiverate, and treshhold
                temp_actual_label.extend(test_data['class'])
                temp_predicted_class.extend(predicted_class)
                ROC_auc_mean.append(auc(fpr, tpr1)) # AUC-ROC
                fprs_mean.append(np.mean(fpr))
                tprs_mean.append(np.mean(tpr1))
                precisions_mean.append(np.mean(prec))
                recalls_mean.append(np.mean(tpr))
                P_R_auc_mean.append(auc(tpr, prec)) # AUC-precision_recall
                fun2=interp1d(fpr, tpr1)
                fold_mean_tpr1 += fun2(fold_mean_fpr) # fit fpr with tpr1 grid
                fun3=interp1d (tpr, prec)
                fold_mean_prec+=fun3(fold_mean_fpr) 

        F1_score=f1_score(temp_actual_label, temp_predicted_class)
        F1_all.append(F1_score)
        fold_mean_tpr1 /= 5*neg_numb_iter #  the average of cross validation iterations  
        all_mean_tpr1[n]= fold_mean_tpr1 
        P_R_all_auc.append(np.mean(P_R_auc_mean))
        ROC_all_auc.append(np.mean(ROC_auc_mean))
        ACC_all_mean.append(np.mean(ACC_mean))
        fold_mean_prec/=5*neg_numb_iter
        all_mean_prec[n]= fold_mean_prec
        fprs.append(np.mean(fprs_mean))
        tprs.append(np.mean(tprs_mean))
        precisions.append(np.mean(precisions_mean))
        recalls.append(np.mean(recalls_mean))

    all_mean_tpr1=np.array(all_mean_tpr1)
    average_tpr1 = [float()]*101 # 101 is grid number,see fold_mean_tpr
    sd_tpr1 = [float()]*101
    for i in range(101):
        average_tpr1[i]=np.mean(all_mean_tpr1[:,i]) # average of true positive rate at each grid point
        sd_tpr1[i]=np.std(all_mean_tpr1[:,i])
    
    print("trait:", trait)
    print("Accuracy: ", np.mean(ACC_all_mean))
    print ('ROC AUC averge %f ; SD is %f' % (np.mean(ROC_all_auc), np.std(ROC_all_auc))) # mean and standard deviation of AUC-ROC  
    t_total = sum(tprs)
    t_count = len(tprs)
    t_result = t_total/t_count
    print("TPR: ",t_result)
    f_total = sum(fprs)
    f_count = len(fprs)
    f_result = f_total/f_count
    print("FPR: ",f_result)
    
    p_total = sum(precisions)
    p_count = len(precisions)
    p_result = p_total/p_count
    print("Precision: ",p_result)
    r_total = sum(recalls)
    r_count = len(recalls)
    r_result = r_total/r_count
    print("Recall: ",r_result)
    
    print ('precision recall AUC averge %f ; SD is %f' % (np.mean(P_R_all_auc), np.std(P_R_all_auc)))
    
    all_mean_prec=np.array(all_mean_prec)
    average_prec = [float()]*101
    sd_prec = [float()]*101
    for i in range(101):
        average_prec[i]=np.mean(all_mean_prec[:,i])
        sd_prec[i]=np.std(all_mean_prec[:,i])      
    
   
    print('\nF1 score average:\n',np.mean(F1_all),'F1 score sd:\n',np.std(F1_all))
    with open(writer_path, 'a') as file:
        file.write("scale:" + str(scale) + '\n')
        file.write("Accuracy: " + str(np.mean(ACC_all_mean)) + '\n')
        file.write('ROC AUC averge '+ str(np.mean(ROC_all_auc)) +' ; SD is ' + str(np.std(ROC_all_auc)) + '\n')
        file.write('precision recall AUC averge %f ; SD is %f' % (np.mean(P_R_all_auc), np.std(P_R_all_auc)) + '\n')
        file.write('\nF1 score average:'+str(np.mean(F1_all))+'  F1 score sd:'+str(np.std(F1_all))+'\n')
        file.write("TPR: " + str(t_result) + '\n')
        file.write("FPR: " + str(f_result) + '\n')
        file.write("Precision: " + str(p_result) + '\n')
        file.write("Recall: " + str(r_result) + '\n')
        file.write("--- "+ str(round((time.time() - start_time),2)) +" seconds ---" + '\n\n')
        for i in range(0, len(fold_mean_fpr)):
            file.write('{0}\t{1}\t{2}\n'.format(fold_mean_fpr[i], average_tpr1[i], sd_tpr1[i]))
        # file.write(str(clf) + '\n')
        file.write('\n')
            

def clean_feature_name(name):
    # 只保留字母、数字和下划线
    name = re.sub(r'[^\w]', '_', name)
    # 删除前后的下划线
    name = name.strip('_')
    return name
# input feature list
df = pd.read_csv(dt)
df.columns = [clean_feature_name(col) for col in df.columns]
df = df.drop(['ID'], axis=1)
df = df.dropna(axis=1,how='all')

train_set = df[df['class']==1]
cross_validation(df, train_set)

print("--- %s seconds ---" % round((time.time() - start_time),2))