import random
import pandas as pd
import re
import pickle
from function.data_process import clean_feature_name

def train(dt, clf, scale,  trait, species):
    save_path = './code/model/' + trait + '_' + species +'_model.dat'
    with open(save_path, 'wb') as pik_f:
        df = pd.read_csv(dt)
        df.columns = [clean_feature_name(col) for col in df.columns]
        pickle.dump(df, pik_f)
        df=df.dropna(axis=1,how='all') 
        df = df.drop(['ID'], axis=1) 
        train_set = df[df['class']==1]
        neg_inter=5000
        pickle.dump(neg_inter, pik_f)
        for i in range(0, neg_inter):
                    train_data = train_set
                    training_negative = random.sample(list(df[df['class']==0].index), int(len(train_data)*scale))
                    train_data=train_data.append(df.iloc[training_negative]) 
                    train_feature=train_data.drop(['class'], axis=1) 
                    clf.fit(train_feature, train_data['class'])  # model fitting
                    pickle.dump(clf, pik_f)
        return save_path
