import numpy as np
import random
import pandas as pd
from data_process import clean_feature_name
import lightgbm as lgb
from para_set import para_set
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dt', default='', type=str)   # dataset
parser.add_argument('--QTL_path', default='', type=str)  # QTL path
parser.add_argument('--trait', default='', type=str)  # select trait: PH, FT and TBN
parser.add_argument('--result_path', default='', type=str)  # path to save the result
args = parser.parse_args()

dt = args.dt
QTL_path = args.QTL_path
trait = args.trait
write_path = args.result_path

df = pd.read_csv(dt)
df.columns = [clean_feature_name(col) for col in df.columns]
df = df.dropna(axis=1,how='all')

Validation_set_ID = pd.read_csv(QTL_path,names=['ID','class'],header=None) 

# exclude genes not in the feature list
Validation_set=pd.DataFrame()
for i in range(len(Validation_set_ID)):
    Validation_set=Validation_set.append(df[df.ID==Validation_set_ID.ID[i]])
for i in range(len(Validation_set)):
    Validation_set.iloc[i,len(Validation_set.columns)-1]= int(Validation_set_ID[Validation_set_ID.ID==Validation_set.iloc[i,0]]['class'])

# number of gene on the QTL 
original_length = len(Validation_set_ID)
print('Number of genes: '+str(original_length-1)) 

# assign validation_set and train_set
df = df.drop(['ID'], axis=1) 
train_set = df[df['class']==1]

Validation_set_ID_uni = list(Validation_set.ID) 
Validation_set = Validation_set.drop(['ID'], axis=1) 
Validation_set_i = Validation_set.reset_index(drop=True)

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

neg_inter = 5000
prediction_list = (len(Validation_set_i))*[0]

for i in range(0, neg_inter):
            train_data = train_set 
            # randomly select negatives from genome genes 
            training_negative = random.sample(list(df[df['class']==0].index), len(train_data)*scale)
            train_data = train_data.append(df.iloc[training_negative]) 
            train_feature = train_data.drop(['class'], axis=1)
            # model fitting
            clf.fit(train_feature, train_data['class'])
            validation_feature_all = Validation_set_i.drop(['class'], axis=1)
            validation_pred = clf.predict_proba(validation_feature_all)[:,1]
            prediction_list += validation_pred
            
prediction_list_all=np.array(prediction_list)
Ranks=prediction_list_all.argsort()[::-1].argsort()
# extract causal index
causal_index = Validation_set_i.index[Validation_set_i['class']==1].tolist()

weights = []
for i in causal_index:
    # ranking of each causal gene in QTL
    print('Causal gene {0} rank {1} ({2}%), with {3} interation'.format(Validation_set_ID_uni[i],Ranks[i]+1,int(((Ranks[i])/original_length)*100),neg_inter))
    weights.append(1 - (Ranks[i])/original_length)
    with open(write_path, 'a') as file:
        file.write("1:"+ str(scale) +"\n")
        file.write(QTL_path.split('/')[-1].split('.')[0]+'\n')
        file.write('Number of genes: '+str(original_length))
        file.write('\nCausal gene {0} rank {1} ({2}%), with {3} interation'.format(Validation_set_ID_uni[i],Ranks[i]+1,int(((Ranks[i])/original_length)*100),neg_inter))
        file.write('\n')

