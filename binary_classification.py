import os
import pandas as pd
import numpy as np
import random
from human_ISH_config import *
import h5py
import time
from shutil import copyfile
import operator
import matplotlib.pyplot as plt
import math
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def get_sz_labels_image_and_donor_level(label):

    #path_to_sz_info = os.path.join(DATA_DIR, STUDY, "human_ISH_info.csv")
    general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/sz"
    path_to_sz_info = os.path.join(general_path, "human_ISH_info.csv")
    sz_info_df = pd.read_csv(path_to_sz_info)



    if label == 'disease_diagnosis':

        new_df = pd.DataFrame(columns=['ID', label])

        # --------------- image level
        new_df['ID'] = sz_info_df['image_id']
        diagnosis = list(sz_info_df['description'])

        image_sz_count = 0
        image_no_sz_count = 0
        for i in range(len(diagnosis)):
            if "schizophrenia" in diagnosis[i]:
                diagnosis[i] = True
                image_sz_count +=1

            elif "control" in diagnosis[i]:
                diagnosis[i] = False
                image_no_sz_count +=1
            else:
                diagnosis[i] = None

        new_df[label] = diagnosis
        file_name = "sz_diagnosis_image_level.csv"
        new_df.to_csv(os.path.join(general_path, file_name), index=None)

        print ("image sz count: ", image_sz_count)
        print ("image no sz count: ", image_no_sz_count)
        print ("total: ", image_sz_count + image_no_sz_count)


        # --------------- donor level
        group_by_donor = sz_info_df.groupby('donor_id')
        donor_list=[]
        diagnosis_list = []

        donor_sz_count = 0
        donor_no_sz_count = 0

        for key, item in group_by_donor:
            donor_list.append(key)
            diagnosis = list(item['description'])[0]
            if "schizophrenia" in diagnosis:
                diagnosis_list.append(True)
                donor_sz_count +=1

            elif "control" in diagnosis:
                diagnosis_list.append(False)
                donor_no_sz_count +=1
            else:
                diagnosis_list.append(None)

        new_df = pd.DataFrame(columns=['ID', label])
        new_df['ID']= donor_list
        new_df[label] = diagnosis_list

        file_name = "sz_diagnosis_donor_level.csv"
        new_df.to_csv(os.path.join(general_path, file_name), index=None)

        print ("donor sz count: ", donor_sz_count)
        print ("donor no sz count: ", donor_no_sz_count)
        print ("total: ", donor_sz_count + donor_no_sz_count)



    elif label in ['donor_age', 'donor_sex', 'smoker', 'pmi', 'tissue_ph', 'donor_race']:
        new_df = pd.DataFrame(columns=['ID', label])

        # --------------- image level
        new_df['ID'] = sz_info_df['image_id']
        new_df[label] = list(sz_info_df[label])

        file_name = label + "_as_label_image_level.csv"
        new_df.to_csv(os.path.join(general_path, file_name), index=None)

        # --------------- donor level
        group_by_donor = sz_info_df.groupby('donor_id')
        donor_list = []
        label_list = []


        for key, item in group_by_donor:
            donor_list.append(key)
            label_list.append(list(item[label])[0])

        new_df = pd.DataFrame(columns=['ID', label])
        new_df['ID'] = donor_list
        new_df[label] = label_list

        file_name = label + "_as_label_donor_level.csv"
        new_df.to_csv(os.path.join(general_path, file_name), index=None)


def get_sz_labels_gene_level():
    general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/"
    path_to_sz_info = os.path.join(general_path, "sz", "human_ISH_info.csv")
    sz_info_df = pd.read_csv(path_to_sz_info)
    sz_genes = sz_info_df['gene_symbol'].unique()
    sz_genes_dict ={item:None for item in sz_genes}
    print ("sz genes: ", len(sz_genes))

    general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/"
    path_to_cortex_info = os.path.join(general_path, "cortex", "human_ISH_info.csv")
    cortex_info_df = pd.read_csv(path_to_cortex_info)
    cortex_genes = cortex_info_df['gene_symbol'].unique()
    not_sz_genes= [item for item in cortex_genes if item not in sz_genes_dict]
    print("not_sz_genes: ", len(not_sz_genes))

    total_genes = []
    total_diagnosis = []

    for item in sz_genes:
        total_genes.append(item)
        total_diagnosis.append(True)

    for item in not_sz_genes:
        total_genes.append(item)
        total_diagnosis.append(False)

    new_df = pd.DataFrame(columns=['ID', 'disease_diagnosis'])
    new_df['ID'] = total_genes
    new_df['disease_diagnosis'] = total_diagnosis

    file_name = "sz_diagnosis_gene_level.csv"
    new_df.to_csv(os.path.join(general_path, "sz", file_name), index=None)


def sz_diagnosis_create_training_files(path_to_embeddings, path_to_labels, levels, path_to_save_files):

    label_files = os.listdir(path_to_labels)
    embed_files = os.listdir(path_to_embeddings)
    label_df = pd.DataFrame()
    embed_df = pd.DataFrame()

    for level in levels:
        print ("----"*10)
        print ("level: ", level)

        # --- get label file
        level_label_file = None

        for item in label_files:
            if level in item:
                level_label_file = item
                break

        if level_label_file == None:
            print ("Could not find a label file at this level")
        else:
            print ("label file: ", level_label_file)
            label_df = pd.read_csv(os.path.join(path_to_labels, level_label_file))


        # --- get embed file
        if level == "image":
            level_embed_file = None
            for item in embed_files:
                if level in item and "schizophrenia" in item:
                    level_embed_file = item
                    break

            if level_embed_file == None:
                print ("Could not find a embed file at this level")
            else:
                print ("embed file: ", level_embed_file)
                embed_df = pd.read_csv(os.path.join(path_to_embeddings, level_embed_file))
                embed_df = embed_df.rename(columns={'image_id' : 'ID'})


        elif level == "donor":
            level_embed_file = None
            for item in embed_files:
                if level in item and "schizophrenia" in item:
                    level_embed_file = item
                    break

            if level_embed_file == None:
                print("Could not find a embed file at this level")
            else:
                print("embed file: ", level_embed_file)
                embed_df = pd.read_csv(os.path.join(path_to_embeddings, level_embed_file))
                embed_df = embed_df.rename(columns={'donor_id': 'ID'})


        elif level == "gene":

            level_no_sz_embed_file = None
            level_sz_embed_file = None
            for item in embed_files:
                if level in item and "schizophrenia" not in item:
                    level_no_sz_embed_file = item

                if level in item and "schizophrenia" in item:
                    level_sz_embed_file = item

            if level_no_sz_embed_file == None:
                print("Could not find a no-sz embed file at this level")

            if level_sz_embed_file == None:
                print("Could not find a sz embed file at this level")

            if level_no_sz_embed_file and level_sz_embed_file:

                no_sz_embed_df = pd.read_csv(os.path.join(path_to_embeddings, level_no_sz_embed_file))
                no_sz_embed_df = no_sz_embed_df.rename(columns={'gene_symbol': 'ID'})
                print (len(no_sz_embed_df))
                sz_embed_df = pd.read_csv(os.path.join(path_to_embeddings, level_sz_embed_file))
                sz_embed_df = sz_embed_df.rename(columns={'gene_symbol': 'ID'})
                print (len(sz_embed_df))


                embed_df = pd.concat([no_sz_embed_df, sz_embed_df], ignore_index=True)
                #print (embed_df.head())

            else:
                embed_df = None



        # at this point, i have the lavel df and the embed df
        # i want to merge them on their ID column
        # i want to keep the id, the 128 embedding vector values and the label


        merged_df = embed_df.merge(label_df, how="left", on="ID")

        # check number of columns
        cols = list(merged_df)
        if len(cols) != 130:
            print ("Something is wrong. There should be 130 columns (id, embeds, label")


        else:
            file_name = "sz_diagnosis_binary_class_" + level + "_level.csv"
            merged_df.to_csv(os.path.join(path_to_save_files, file_name), index=None)



def embeddings_per_gene_per_donor(path_to_embeddings):
    general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/sz"

    path_to_sz_info = os.path.join(general_path, "human_ISH_info.csv")
    sz_info_df = pd.read_csv(path_to_sz_info)

    embeddings_df = pd.read_csv(os.path.join(path_to_embeddings, "triplet_patches_schizophrenia_embeddings_image_level.csv"))

    # I want to add two extra columns: gene_symbol and donor_id

    left = embeddings_df
    right = sz_info_df
    merge_res = pd.merge(left, right[['image_id','gene_symbol', 'donor_id']], how='left', on='image_id')
    merge_res = merge_res.drop(columns=['image_id'])

    genes = list(merge_res['gene_symbol'].unique())

    """
    per_gene_per_donor_path = os.path.join(general_path, "per_gene_per_donor")
    group_by_gene = merge_res.groupby('gene_symbol')
    for key, item in group_by_gene:
        gene_name = key
        item= item.drop(columns=['gene_symbol'])
        group_by_donor = item.groupby('donor_id').mean()
        group_by_donor.to_csv(os.path.join(per_gene_per_donor_path, gene_name+".csv"))
        
    """

    return genes

def demog_info_as_training(list_of_columns_to_get, ts):
    general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation"

    path_to_sz_info = os.path.join(general_path, "dummy_4/sz", "human_ISH_info.csv")
    sz_info_df = pd.read_csv(path_to_sz_info)

    list_of_columns_to_get = ['donor_id'] + list_of_columns_to_get
    filtered = sz_info_df[list_of_columns_to_get]

    grouped_by_donor = filtered.groupby('donor_id')
    demog_df = grouped_by_donor.first().reset_index()

    file_name = "demog_info_as_training_donor_level.csv"
    demog_df.to_csv(os.path.join(general_path, "dummy_4/sz", file_name), index= None)


    # ---- merge with embeddings

    donor_level_embeddings_path = os.path.join(general_path, "dummy_3", ts, "triplet_patches_schizophrenia_embeddings_donor_level.csv")
    embeds_df = pd.read_csv(donor_level_embeddings_path)

    left = embeds_df
    right = demog_df

    merged_res = pd.merge(left, right, how='left', on='donor_id')
    file_name = "demog_info_and_embeddings_as_training_donor_level.csv"
    merged_res.to_csv(os.path.join(general_path, "dummy_4/sz", file_name), index=None)




def perform_logistic_regression(ts,level, n_splits =5, n_jobs = 1):
    general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/"

    path_to_embed_file = os.path.join(general_path, "dummy_3", ts, "triplet_patches_schizophrenia_embeddings_"+level+"_level.csv")
    path_to_labels = os.path.join(general_path, "dummy_4", "sz", "sz_diagnosis_" + level +"_level.csv")

    embeds_df = pd.read_csv(path_to_embed_file)
    labels = pd.read_csv(path_to_labels)
    labels = labels.rename(columns={'ID': level+'_id'})
    left = embeds_df
    right = labels
    merge_res = pd.merge(left, right, how='left', on=level+"_id")

    scores = []
    skf = StratifiedKFold(n_splits=n_splits)

    col_titles = [str(item) for item in range(128)]
    X = merge_res[col_titles]
    Y = merge_res['disease_diagnosis']

    y_test_total = pd.Series([])
    preds_total = []
    probas_total = pd.DataFrame()

    for i, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
        model = LogisticRegression(penalty='none', n_jobs=n_jobs, max_iter=1000)
        X_train = X.iloc[train_idx, :]
        y_train = Y.iloc[train_idx]
        X_test = X.iloc[test_idx, :]
        y_test = Y.iloc[test_idx]

        model.fit(X_train, y_train)

        # Extract predictions from fitted model
        preds = list(model.predict(X_test))
        # probs for classes ordered in same manner as model.classes_
        # model.classes_  >>  array([False,  True])
        probas = pd.DataFrame(model.predict_proba(
            X_test), columns=model.classes_)

        y_test_total = y_test_total.append(y_test)
        preds_total += preds
        probas_total = probas_total.append(probas)

        print ("Finished fold: ", i+1)

    print ("----" * 20)


    preds_total = np.array(preds_total)

    f1 = f1_score(y_test_total, preds_total)
    auc = roc_auc_score(y_test_total, probas_total[True])

    measures = {'level': level,
                'f1': f1,
                'AUC': auc}


    scores.append(measures)

    return pd.DataFrame(scores,columns=['level', 'AUC', 'f1']).sort_values(by=['AUC'],ascending=False).reset_index().drop(columns=['index'])



def perform_logistic_regression_per_gene_per_donor(genes_list, n_splits =5, n_jobs=1):
    """
    n_splits: number of folds to be used in cross validation
    n_jobs: int, default=1
    Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”.
    """

    general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/sz"

    invalid_genes_count = 0
    scores = []
    skf = StratifiedKFold(n_splits=n_splits)
    for gene in genes_list:

        print ("Gene is: ", gene)
        path_to_donor_level_embeds = os.path.join(general_path,"per_gene_per_donor", gene+".csv")
        embeds = pd.read_csv(path_to_donor_level_embeds)


        if len(embeds) < 40:
            invalid_genes_count +=1
            pass
        else:

            number_of_donors = len(embeds)
            left = embeds

            path_to_labels = os.path.join(general_path, "sz_diagnosis_donor_level.csv")
            labels = pd.read_csv(path_to_labels)
            labels = labels.rename(columns={'ID': 'donor_id'})
            right = labels

            merge_res = pd.merge(left, right, how='left', on='donor_id')

            col_titles =[str(item) for item in range(128)]
            X = merge_res[col_titles]
            Y = merge_res['disease_diagnosis']

            y_test_total = pd.Series([])
            preds_total = []
            probas_total = pd.DataFrame()

            for i, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
                model = LogisticRegression(penalty='none', n_jobs=n_jobs, max_iter=500)
                X_train = X.iloc[train_idx, :]
                y_train = Y.iloc[train_idx]
                X_test = X.iloc[test_idx, :]
                y_test = Y.iloc[test_idx]

                model.fit(X_train, y_train)

                # Extract predictions from fitted model
                preds = list(model.predict(X_test))
                # probs for classes ordered in same manner as model.classes_
                # model.classes_  >>  array([False,  True])
                probas = pd.DataFrame(model.predict_proba(
                    X_test), columns=model.classes_)

                y_test_total = y_test_total.append(y_test)
                preds_total += preds
                probas_total = probas_total.append(probas)

                print ("Finished fold: ", i+1)

            print ("----" * 20)


            preds_total = np.array(preds_total)

            f1 = f1_score(y_test_total, preds_total)
            auc = roc_auc_score(y_test_total, probas_total[True])

            measures = {'gene_symbol': gene,
                        'number_of_donors': number_of_donors,
                        'f1': f1,
                        'AUC': auc}

            scores.append(measures)


    print (invalid_genes_count)
    return pd.DataFrame(scores,
                        columns=['gene_symbol', 'number_of_donors', 'AUC', 'f1']).sort_values(by=['AUC'],
                                                                                       ascending=False).reset_index().drop(columns=['index'])



if __name__ == "__main__":

    """
    levels = ['gene']
    path_to_embeddings = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_3/1603427490"
    path_to_labels =  "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/sz"
    path_to_save_files = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/binary"
    sz_diagnosis_create_training_files(path_to_embeddings, path_to_labels, levels, path_to_save_files)
    
    """

    """
    general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/sz"

    path_to_embeddings = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_3/1603427490"
    genes = embeddings_per_gene_per_donor(path_to_embeddings)
    diagnosis_prediction_res = perform_logistic_regression(genes)

    diagnosis_prediction_res.to_csv(os.path.join(general_path, "per_gene_per_donor_diagnosis_prediction_scores.csv"), index=False)
    """




    # ---- donor level
    #general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/sz"
    #donor_level_prediction_res=perform_logistic_regression("1603427490", "donor", n_splits=5, n_jobs=1)
    #donor_level_prediction_res.to_csv(os.path.join(general_path, "per_donor_diagnosis_prediction_scores.csv"),index=False)



    # ---- image level
    #general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/sz"
    #image_level_prediction_res=perform_logistic_regression("1603427490", "image", n_splits=5, n_jobs=1)
    #image_level_prediction_res.to_csv(os.path.join(general_path, "per_image_diagnosis_prediction_scores.csv"),index=False)


    labels= ['donor_age', 'donor_sex', 'smoker', 'pmi', 'tissue_ph', 'donor_race']
    #for label in labels:
        #get_sz_labels_image_and_donor_level(label)

    demog_info_as_training(labels, "1603427490")




