import numpy as np
import pandas as pd
import pickle

def predict_rank(model_path, QTL, trait, species, result_path):
    with open(QTL, 'r') as f, open(model_path, 'rb') as pik_f:
        df = pickle.load(pik_f)
        QTL_list = [tmp.strip('\n') for tmp in f]
        QTL_name = species + '-' + trait
        QTL_length = len(QTL_list)
        Validation_set = pd.DataFrame()
        
        for i in range(QTL_length):
            Validation_set = Validation_set.append(df[df.ID == QTL_list[i]])
        
        df = df.drop(['ID'], axis=1)
        ind_for_exclusion = []
        
        for t in range(len(Validation_set.index)):
            if Validation_set['class'].iloc[t] == 1:
                ind_for_exclusion.append(t)
                Validation_set.at[Validation_set.index[t], 'ID'] = '__exist in positive gene list__' + Validation_set['ID'].iloc[t]
        
        Validation_set_ID = list(Validation_set.ID)
        gene_ex = set(QTL_list) - set(Validation_set_ID)
        Validation_set = Validation_set.drop(['ID'], axis=1)
        Validation_set_i = Validation_set.reset_index(drop=True)
        
        prediction_list = [0] * len(Validation_set_i)
        validation_feature_all = Validation_set_i.drop(['class'], axis=1)
        
        if len(validation_feature_all) > 0:
            neg_inter = pickle.load(pik_f)
            for _ in range(neg_inter):
                single_model = pickle.load(pik_f)
                validation_pred = single_model.predict_proba(validation_feature_all)[:, 1]
                prediction_list += validation_pred
            
            prediction_list_all = np.array(prediction_list)
            Ranks = prediction_list_all.argsort()[::-1].argsort()
            prediction_list_freq = [l / neg_inter for l in prediction_list]
            rank_list_df = pd.DataFrame({'ID': Validation_set_ID, 'freq': prediction_list_freq})
            
            for gene in gene_ex:
                rank_list_df = rank_list_df.append({'ID': gene, 'freq': 0}, ignore_index=True)
            
            rank_list_df['Rank'] = rank_list_df['freq'].rank(ascending=0, method='average')
            rank_list_df_sorted = rank_list_df.sort_values(by=['Rank'], ascending=True).reset_index(drop=True)
            
        else:
            if len(gene_ex) != 0:
                rank_list_df_sorted = pd.DataFrame({'ID': [], 'freq': [], 'Rank': []})
                for gene in gene_ex:
                    rank_list_df_sorted = rank_list_df_sorted.append({'ID': gene, 'freq': 'NA', 'Rank': 'NA'}, ignore_index=True)
    
    with open(result_path, 'w') as file:
        file.write(QTL_name + '\n')
        file.write('Rank_in_a_QTL,ID\n')
        # file.write ('ID'+','+'Rank_in_a_QTL'+','+'Score'+'\n')
        for i in range(len(rank_list_df_sorted)):
            file.write (rank_list_df_sorted['ID'][i]+','+str(int(rank_list_df_sorted['Rank'][i])) + '\n')   
            # file.write (rank_list_df_sorted['ID'][i]+','+str(int(rank_list_df_sorted['Rank'][i]))+','+ str(rank_list_df_sorted['freq'][i]) +'\n')
