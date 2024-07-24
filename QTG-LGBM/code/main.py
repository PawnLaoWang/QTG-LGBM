
import argparse
import lightgbm as lgb
from function.train import train
from function.para_set import para_set
from function.rank import predict_rank
import os
parser = argparse.ArgumentParser()
parser.add_argument('--dt', default='', type=str)   # dataset
parser.add_argument('--QTL_path', default='', type=str) # QTL_path
parser.add_argument('--species', default='', type=str)  # species: Maize or more
parser.add_argument('--trait', default='', type=str)  # phenotype FT, PH and TBN or more
parser.add_argument('--result_path', default='', type=str)  # path to save
args = parser.parse_args()

dt = args.dt
QTL_path= args.QTL_path
species = args.species
trait = args.trait
result_path = args.result_path

if species == 'Maize' and trait == 'PH':
    clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=15, reg_alpha=0.0, reg_lambda=1,
            max_depth=4, n_estimators=30, objective='binary',
            subsample=1, colsample_bytree=1, subsample_freq=1,is_unbalance=True,
            learning_rate=0.09, min_child_weight=5, random_state=20)
    scale = 30
    
elif species == 'Maize' and trait == 'FT':
    clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=15, reg_alpha=0.0, reg_lambda=1,
            max_depth=9, n_estimators=50, objective='binary',
            subsample=1, colsample_bytree=1, subsample_freq=1,is_unbalance=True,
            learning_rate=0.07, min_child_weight=1, random_state=20)
    scale = 130
    
elif species == 'Maize' and trait == 'TBN':
    clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=15, reg_alpha=0.0, reg_lambda=1,
            max_depth=7, n_estimators=50, objective='binary',
            subsample=1, colsample_bytree=1, subsample_freq=1,is_unbalance=True,
            learning_rate=0.07, min_child_weight=1, random_state=20)
    scale = 120

else:
    if os.path.exists('./code/model/'+ '_' + trait + '_' + species +'_model.dat'):
        pass
    else:
        clf, scale = para_set(dt)

# train model
if os.path.exists('./code/model/'+ '_' + trait + '_' + species +'_model.dat'):
    pass
else:
    model_path = train(dt, clf, scale, trait, species)

# use QTL to predict
predict_rank(model_path, QTL_path, trait, species, result_path)