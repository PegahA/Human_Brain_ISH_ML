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
import sklearn


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

print(sklearn.__version__)


general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/"
sz_general_path = os.path.join(general_path, "dummy_4/sz")

def get_sz_labels_image_and_donor_level(label):

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



def embeddings_per_gene_per_donor(input_type, ts, embeddings_df):
    """
    This function gets an image-level embedding file and outputs a donor-level csv file for each gene.
    Each gene will have a separate csv file: gene_name.csv
    Each row in the csv file will represent a donor.
    The number of rows in the csv file is the number of donors on which this specific gene was tested.

    We will use image level embeddings, then group them by gene. So each group will be all the images that assay the same gene.
    Then, within each group, will group the images again by donor_id and use the mean() function to take the average of the embeddings.

    :param input_type: str. Determine the type of input vectors.
    Could be: ['embed' ,'demog' , 'demog_and_embed', 'random', 'plain_resnet', 'demog_without_smoker',
                   'demog_and_embed_without_smoker', 'demog_without_sex', 'demog_and_embed_without_sex']

    :param ts: str. The timestamp that indicates which files to use.
    :param embeddings_df: pandas data frame. Image-level embeddings.
    :return: a list of genes
    """

    # the embeddings are image level
    path_to_sz_info = os.path.join(sz_general_path, "human_ISH_info.csv")
    sz_info_df = pd.read_csv(path_to_sz_info)

    # I want to add two extra columns: gene_symbol and donor_id to the embeddings
    # if the file already has donor_id, don't add it
    left = embeddings_df
    left_cols = list(embeddings_df)
    right = sz_info_df
    if 'donor_id' in left_cols:
        merge_res = pd.merge(left, right[['image_id', 'gene_symbol']], how='left', on='image_id')
    else:

        merge_res = pd.merge(left, right[['image_id','gene_symbol', 'donor_id']], how='left', on='image_id')
    print (merge_res.head())
    print ("lll")
    print (list(merge_res))

    genes = list(merge_res['gene_symbol'].unique())

    if input_type == 'random' or input_type == 'resnet':
        # random and resnet do not require a timestamp

        per_gene_per_donor_general_path = os.path.join(sz_general_path, "per_gene_per_donor")
        if (not os.path.exists(per_gene_per_donor_general_path)):
            os.mkdir(per_gene_per_donor_general_path)

        per_gene_per_donor_path = os.path.join(per_gene_per_donor_general_path, input_type + "_per_gene_per_donor")
        if (not os.path.exists(per_gene_per_donor_path)):
            os.mkdir(per_gene_per_donor_path)

    else:
        per_gene_per_donor_general_path = os.path.join(sz_general_path, "per_gene_per_donor")
        if (not os.path.exists(per_gene_per_donor_general_path)):
            os.mkdir(per_gene_per_donor_general_path)

        per_gene_per_donor_path = os.path.join(per_gene_per_donor_general_path, ts+ "_" + input_type +"_per_gene_per_donor")
        if (not os.path.exists(per_gene_per_donor_path)):
            os.mkdir(per_gene_per_donor_path)

    group_by_gene = merge_res.groupby('gene_symbol')
    for key, item in group_by_gene:
        # key is gene_symbol
        # item is the group of images that assay that gene
        item = item.drop(columns=['image_id'])
        group_by_donor = item.groupby('donor_id').mean()
        gene_name = key

        group_by_donor.to_csv(os.path.join(per_gene_per_donor_path, gene_name + ".csv"))

    return genes


def embeddings_per_gene_per_donor_old_bug(input_type, ts, embeddings_df):
    """
    This function is WRONG.
    It expected a donor-level embedding data frame. Merging it will info_df will not give out the desired output.
    It would take the embeddings for the donor from the embeddings_df and put that same embedding vector for every gene.
    Therefore, even for different gene_name.csv files, the rows (which represent the donors) will be the same across different genes.
    """

    path_to_sz_info = os.path.join(sz_general_path, "human_ISH_info.csv")
    sz_info_df = pd.read_csv(path_to_sz_info)

    # I want to add two extra columns: gene_symbol and donor_id

    left = embeddings_df
    right = sz_info_df
    merge_res = pd.merge(left, right[['gene_symbol', 'donor_id']], how='left', on='donor_id')

    print (left.head())
    print (right.head())
    print (merge_res.head())

    genes = list(merge_res['gene_symbol'].unique())


    if input_type == 'random' or input_type == 'resnet':
        per_gene_per_donor_path = os.path.join(sz_general_path, input_type + "_per_gene_per_donor")

    else:

        per_gene_per_donor_path = os.path.join(sz_general_path, ts+ "_" + input_type +"_per_gene_per_donor")
    if (not os.path.exists(per_gene_per_donor_path)):
        os.mkdir(per_gene_per_donor_path)

    group_by_gene = merge_res.groupby('gene_symbol')
    for key, item in group_by_gene:
        gene_name = key
        item= item.drop(columns=['gene_symbol'])
        group_by_donor = item.groupby('donor_id').mean()
        group_by_donor.to_csv(os.path.join(per_gene_per_donor_path, gene_name+".csv"))
        

    return genes


def demog_info_as_training(list_of_columns_to_get, ts):
    """
    For every image, it also extracts the demographics info and adds them as new columns.
    For 'smoker', 'donor_sex', and 'donor_race', it performs one-hot coding.
    Everything needs to be image-level

    :param list_of_columns_to_get:
    :param ts:
    :param without:
    :return:
    """

    path_to_sz_info = os.path.join(sz_general_path, "human_ISH_info.csv")
    sz_info_df = pd.read_csv(path_to_sz_info)

    list_of_columns_to_get = ['image_id', 'donor_id'] + list_of_columns_to_get
    demog_df = sz_info_df[list_of_columns_to_get]


    # ------ handle one-hot encoding ------

    columns_needing_one_hot = ['smoker', 'donor_sex', 'donor_race']
    keep = []
    for item in columns_needing_one_hot:
        if item in list_of_columns_to_get:
            keep.append(item)

    one_hot_dfs = []
    for item in keep:
        item_one_hot = pd.get_dummies(demog_df[item], prefix=item)
        print (item_one_hot.head())
        one_hot_dfs.append(item_one_hot)


    for item in one_hot_dfs:
        demog_df = pd.concat([demog_df, item], axis=1)

    print (demog_df.head())
    print (list(demog_df))

    smoker_one_hot = pd.get_dummies(demog_df['smoker'], prefix='smoker')
    sex_one_hot = pd.get_dummies(demog_df['donor_sex'], prefix='sex')
    race_one_hot = pd.get_dummies(demog_df['donor_race'], prefix='race')

    demog_df = demog_df.drop(columns=['smoker', 'donor_sex', 'donor_race'])

    demog_df = pd.concat([demog_df, smoker_one_hot], axis=1)
    demog_df = pd.concat([demog_df, sex_one_hot], axis=1)
    demog_df = pd.concat([demog_df, race_one_hot], axis=1)

    # -------------------------------------

    file_name =  ts+ "_demog_info_as_training_image_level.csv"
    demog_df.to_csv(os.path.join(sz_general_path, file_name), index=None)

    # ---- merge with image-level embeddings ----

    image_level_embeddings_path = os.path.join(general_path, "dummy_3", ts,
                                               "triplet_patches_schizophrenia_embeddings_image_level.csv")
    embeds_df = pd.read_csv(image_level_embeddings_path)

    left = embeds_df
    right = demog_df

    merged_res = pd.merge(left, right, how='left', on='image_id')
    file_name = ts + "_demog_info_and_embeddings_as_training_image_level.csv"
    merged_res.to_csv(os.path.join(sz_general_path, file_name), index=None)

    # -------------------------------------



def demog_info_as_training_old(list_of_columns_to_get, ts, without=None):
    """
    This function adds the demographics info to donor-level embeddings.
    It's not wrong, it's just not the type of file we would need for per_gene_per_donor analysis.
    """

    path_to_sz_info = os.path.join(sz_general_path, "human_ISH_info.csv")
    sz_info_df = pd.read_csv(path_to_sz_info)

    list_of_columns_to_get = ['donor_id'] + list_of_columns_to_get
    filtered = sz_info_df[list_of_columns_to_get]

    grouped_by_donor = filtered.groupby('donor_id')
    demog_df = grouped_by_donor.first().reset_index()
    # ------ handle one-hot encoding ------

    columns_needing_one_hot = ['smoker', 'donor_sex', 'donor_race']
    keep = []
    for item in columns_needing_one_hot:
        if item in list_of_columns_to_get:
            keep.append(item)

    one_hot_dfs = []
    for item in keep:
        item_one_hot = pd.get_dummies(demog_df[item], prefix=item)
        one_hot_dfs.append(item_one_hot)

    demog_df = demog_df.drop(columns=keep)

    for item in one_hot_dfs:
        demog_df = pd.concat([demog_df, item], axis=1)

    smoker_one_hot = pd.get_dummies(demog_df['smoker'], prefix='smoker')
    sex_one_hot = pd.get_dummies(demog_df['donor_sex'], prefix='sex')
    race_one_hot = pd.get_dummies(demog_df['donor_race'], prefix='race')

    demog_df = demog_df.drop(columns=['smoker', 'donor_sex', 'donor_race'])

    demog_df = pd.concat([demog_df, smoker_one_hot], axis=1)
    demog_df = pd.concat([demog_df, sex_one_hot], axis=1)
    demog_df = pd.concat([demog_df, race_one_hot], axis=1)

    # -------------------------------------

    # ------ handle one-hot encoding ------

    columns_needing_one_hot = ['smoker', 'donor_sex', 'donor_race']
    keep = []
    for item in columns_needing_one_hot:
        if item in list_of_columns_to_get:
            keep.append(item)

    one_hot_dfs = []
    for item in keep:
        item_one_hot = pd.get_dummies(demog_df[item], prefix=item)
        one_hot_dfs.append(item_one_hot)

    demog_df = demog_df.drop(columns=keep)

    for item in one_hot_dfs:
        demog_df = pd.concat([demog_df, item], axis=1)


    smoker_one_hot= pd.get_dummies(demog_df['smoker'], prefix='smoker')
    sex_one_hot = pd.get_dummies(demog_df['donor_sex'], prefix='sex')
    race_one_hot = pd.get_dummies(demog_df['donor_race'], prefix='race')


    demog_df = demog_df.drop(columns=['smoker', 'donor_sex', 'donor_race'])

    demog_df = pd.concat([demog_df, smoker_one_hot], axis=1)
    demog_df = pd.concat([demog_df, sex_one_hot], axis=1)
    demog_df = pd.concat([demog_df, race_one_hot], axis=1)

    # -------------------------------------

    removed = ""
    if without != None:
        removed = "_without_"
        for item in without:
            removed +=item
            removed+="_"
        removed = removed[:-1]

    file_name = ts+ removed +"_demog_info_as_training_donor_level.csv"
    demog_df.to_csv(os.path.join(sz_general_path, file_name), index= None)


    # ---- merge with donor-level embeddings ----

    donor_level_embeddings_path = os.path.join(general_path, "dummy_3", ts, "triplet_patches_schizophrenia_embeddings_donor_level.csv")
    embeds_df = pd.read_csv(donor_level_embeddings_path)

    left = embeds_df
    right = demog_df

    merged_res = pd.merge(left, right, how='left', on='donor_id')
    file_name = ts+removed+"_demog_info_and_embeddings_as_training_donor_level.csv"
    merged_res.to_csv(os.path.join(sz_general_path, file_name), index=None)

    # -------------------------------------








def perform_logistic_regression(path_to_embed_file, path_to_labels_file,level, n_splits =5, n_jobs = 1):

    embeds_df = pd.read_csv(path_to_embed_file)
    labels = pd.read_csv(path_to_labels_file)
    labels = labels.rename(columns={'ID': level+'_id'})
    left = embeds_df
    right = labels
    merge_res = pd.merge(left, right, how='left', on=level+"_id")

    scores = []
    skf = StratifiedKFold(n_splits=n_splits)

    col_titles = list(embeds_df)[1:]
    #col_titles = [str(item) for item in range(128)]
    X = merge_res[col_titles]
    Y = merge_res['disease_diagnosis']

    f1_score_values = []
    auc_values = []

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


        #y_test_total = y_test_total.append(y_test)
        #preds_total += preds
        #probas_total = probas_total.append(probas)

        # Get metrics for each model
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probas[True])

        f1_score_values.append(f1)
        auc_values.append(auc)

        print ("THIS FOLD: ", f1, auc)
        print ("Finished fold: ", i+1)

    print ("----" * 20)


    #preds_total = np.array(preds_total)

    #f1 = f1_score(y_test_total, preds_total)
    #auc = roc_auc_score(y_test_total, probas_total[True])

    f1 = np.mean(f1_score_values)
    auc = np.mean(auc_values)


    print ("FINAL: ", f1, auc)

    measures = {'level': level,
                'f1': f1,
                'AUC': auc}


    scores.append(measures)

    #return pd.DataFrame(scores,columns=['level', 'AUC', 'f1']).sort_values(by=['AUC'],ascending=False).reset_index().drop(columns=['index'])

    return auc



def perform_logistic_regression_per_gene_per_donor(input_type, ts,genes_list, classifier , rnd_state, n_splits =5, n_jobs=1):
    """
    n_splits: number of folds to be used in cross validation
    n_jobs: int, default=1
    Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”.
    """


    invalid_genes_count = 0
    scores = []


    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,  random_state=rnd_state)
    for gene in genes_list:

        print ("Gene is: ", gene)

        if input_type == 'random' or input_type == 'resnet':
            path_to_per_gene_per_donor = os.path.join(sz_general_path, "per_gene_per_donor")
            path_to_donor_level_embeds = os.path.join( path_to_per_gene_per_donor, input_type + "_per_gene_per_donor",
                                                      gene + ".csv")
        else:
            path_to_per_gene_per_donor = os.path.join(sz_general_path, "per_gene_per_donor")
            path_to_donor_level_embeds = os.path.join(path_to_per_gene_per_donor,ts+ "_" + input_type +"_per_gene_per_donor", gene+".csv")

        embeds = pd.read_csv(path_to_donor_level_embeds)


        if len(embeds) < 40:
            invalid_genes_count +=1
            pass
        else:

            number_of_donors = len(embeds)
            left = embeds

            path_to_labels = os.path.join(sz_general_path, "sz_diagnosis_donor_level.csv")
            labels = pd.read_csv(path_to_labels)
            labels = labels.rename(columns={'ID': 'donor_id'})
            right = labels

            merge_res = pd.merge(left, right, how='left', on='donor_id')

            col_titles = list(embeds)[1:]
            #col_titles =[str(item) for item in range(128)]


            X = merge_res[col_titles]
            Y = merge_res['disease_diagnosis']

            #y_test_total = pd.Series([])
            #preds_total = []
            #probas_total = pd.DataFrame()

            f1_score_values = []
            auc_values = []


            for i, (train_idx, test_idx) in enumerate(skf.split(X, Y)):

                if classifier == 'lr':
                    model = LogisticRegression(penalty='none', n_jobs=n_jobs, max_iter=500, random_state=rnd_state)
                elif classifier == 'rf':
                    model = RandomForestClassifier(n_estimators=100,
                                                   bootstrap=True,
                                                   max_features='sqrt', random_state=rnd_state)
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

                #y_test_total = y_test_total.append(y_test)
                #preds_total += preds
                #probas_total = probas_total.append(probas)

                # Get metrics for each model
                f1 = f1_score(y_test, preds)
                auc = roc_auc_score(y_test, probas[True])

                f1_score_values.append(f1)
                auc_values.append(auc)


                
                print ("Finished fold: ", i+1)

            print ("----" * 20)


            #preds_total = np.array(preds_total)

            #f1 = f1_score(y_test_total, preds_total)
            #auc = roc_auc_score(y_test_total, probas_total[True])

            f1 = np.mean(f1_score_values)
            auc = np.mean(auc_values)

            measures = {'gene_symbol': gene,
                        'number_of_donors': number_of_donors,
                        'f1': f1,
                        'AUC': auc}

            scores.append(measures)


    print (invalid_genes_count)
    return pd.DataFrame(scores,
                        columns=['gene_symbol', 'number_of_donors', 'AUC', 'f1']).sort_values(by=['AUC'],
                                                                                       ascending=False).reset_index().drop(columns=['index'])



def check_genes_and_donors():
    general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/sz"

    path_to_sz_info = os.path.join(general_path, "human_ISH_info.csv")
    sz_info_df = pd.read_csv(path_to_sz_info)

    sz_donors_genes = sz_info_df[['donor_id','gene_symbol']]

    # ------
    group_by_genes = sz_donors_genes.groupby('gene_symbol')['donor_id'].apply(list).to_dict()


    potential_genes = {}

    for gene in group_by_genes:
        donors = group_by_genes[gene]
        if len(donors) < 40:
            pass

        else:
            potential_genes[gene] = donors


    print (len(potential_genes))

    donors_list = [potential_genes[genes] for genes  in potential_genes]
    commons = set.intersection(*map(set, donors_list))

    commons = list(set(commons))
    print (len(commons))

    # ---------
    print ("______" *20 +"\n")

    group_by_donors = sz_donors_genes.groupby('donor_id')['gene_symbol'].apply(list).to_dict()

    potential_donors = {}

    for donor in group_by_donors:
        genes = group_by_donors[donor]

        potential_donors[donor] = genes

    print(len(potential_donors))

    genes_list = [potential_donors[donors] for donors in potential_donors]
    commons = set.intersection(*map(set, genes_list))

    commons = list(set(commons))


    return commons



def temp_function():
    path_to_info_csv = os.path.join(sz_general_path, "human_ISH_info.csv")
    info_csv = pd.read_csv(path_to_info_csv, )
    left = info_csv[['image_id', 'donor_id']]

    print (len(left))

    res_img_level_path = os.path.join(sz_general_path, 'plain_resnet', "resnet50_embeddings_image_level.csv")
    res_img_level_df = pd.read_csv(res_img_level_path)
    right = res_img_level_df
    print (len(right))

    merged_df = pd.merge(left, right, how='right', on='image_id')
    merged_df = merged_df.drop(columns=['image_id'])
    print (len(merged_df))

    grouped_df = merged_df.groupby(['donor_id']).mean()


    grouped_df.to_csv( os.path.join(sz_general_path, 'plain_resnet', "resnet50_embeddings_donor_level.csv"))





def generate_random_embeddings_for_disease_dataset(embeddings_length):

    path_to_info_csv = os.path.join(sz_general_path, "human_ISH_info.csv")
    info_csv = pd.read_csv(path_to_info_csv,)

    id_column = list(info_csv['donor_id'].unique())
    print (id_column)

    n_rows = len(id_column)

    cols = np.arange(0, embeddings_length)
    cols = list(map(str, cols))
    cols = ['donor_id'] + cols

    random_embed_file = pd.DataFrame(columns=cols)
    random_embed_file['donor_id'] = id_column

    print (random_embed_file.head())

    for i in range(embeddings_length):
        sample = np.random.uniform(size=(n_rows,))
        random_embed_file[str(i)] = sample


    path_to_random = os.path.join(sz_general_path, "random")

    random_embed_file.to_csv(os.path.join(path_to_random, "random_embeddings_donor_level_2.csv"),index=None)



def perform_random_forest(path_to_embed_file, path_to_labels_file, level):

    random.seed(10)
    #np.random.RandomState(42)
    print ("____________ RANDOM FOREST _______________")

    embed_df = pd.read_csv(path_to_embed_file)
    label_df = pd.read_csv(path_to_labels_file)

    training_ratio = 0.8

    number_of_samples = len(embed_df)


    row_range = [item for item in range(number_of_samples)]

    num_of_training_rows = int(training_ratio * number_of_samples)
    training_rows = random.sample(row_range, num_of_training_rows)
    test_rows = [item for item in row_range if item not in training_rows]

    print ("Test rows: ", test_rows)


    print ("There are {} samples in the training set and {} samples in the test set.".format(len(training_rows), len(test_rows)))


    training_input_df = embed_df.iloc[training_rows]
    training_labels_df = label_df.iloc[training_rows]

    training_input_df = training_input_df.drop(columns=[level+ '_id'])
    training_labels_df = training_labels_df.drop(columns=['ID'])

    test_input_df = embed_df.iloc[test_rows]
    test_labels_df = label_df.iloc[test_rows]


    test_input_df = test_input_df.drop(columns=[level + '_id'])
    test_labels_df = test_labels_df.drop(columns=['ID'])

    # Create the model with 100 trees
    model = RandomForestClassifier(n_estimators=100,
                                   bootstrap=True,
                                   max_features='sqrt', random_state = 10)
    # Fit on training data
    model.fit(training_input_df.to_numpy(), training_labels_df.values.ravel())

    # Actual class predictions
    rf_predictions = model.predict(test_input_df.to_numpy())
    # Probabilities for each class
    rf_probs = model.predict_proba(test_input_df.to_numpy())[:, 1]

    # Calculate roc auc
    roc_value = roc_auc_score(test_labels_df.values.ravel(), rf_probs)

    print (roc_value)


    return roc_value



def get_avg_AUCs(gene_type, input_type_list):

    for input_type in input_type_list:
        path_to_scores = os.path.join(sz_general_path ,gene_type ,input_type)

        files = os.listdir(path_to_scores)
        for file in files:
            path_to_file = os.path.join(path_to_scores , file)
            score_df = pd.read_csv(path_to_file)
            auc_col = list(score_df['AUC'])

            print (path_to_file)
            print (np.mean(auc_col))


def compare_scores(gene_type, input_type_1, input_type_2, classifier):

    path_to_input_1_scores = os.path.join(sz_general_path, gene_type, input_type_1)
    path_to_input_2_scores =  os.path.join(sz_general_path, gene_type, input_type_2)

    input_1_files = os.listdir(path_to_input_1_scores)
    input_2_files = os.listdir(path_to_input_2_scores)


    for file in input_1_files:
        if classifier in file:
            score_1_path = os.path.join(path_to_input_1_scores, file)
            score_1_df = pd.read_csv(score_1_path)

    for file in input_2_files:
        if classifier in file:
            score_2_path = os.path.join(path_to_input_2_scores, file)
            score_2_df = pd.read_csv(score_2_path)



    genes_1 = list(score_1_df['gene_symbol'])
    auc_1 = (score_1_df['AUC'])

    dict_1 = {}
    for i in range(len(genes_1)):
        dict_1[genes_1[i]] = auc_1[i]


    genes_2 = list(score_2_df['gene_symbol'])
    auc_2 = (score_2_df['AUC'])

    dict_2 = {}
    for i in range(len(genes_2)):
        dict_2[genes_2[i]] = auc_2[i]




    first_greater_than_second = []


    for gene in dict_1:
        auc_1 = dict_1[gene]
        auc_2 = dict_2[gene]

        if auc_1 > auc_2:
            first_greater_than_second.append(gene)


    print (first_greater_than_second)


def get_patches_that_activate_neuron_the_most_and_the_least(ts, top_gene, path_to_labels_file, path_to_patch_level_embeddings):
    """
    First, get the donor level embeddings of the top gene (embeddings of the images that assay the top gene, aggregated to donor-level).
    This files is used as the data to get feature importance.

    Standardize the embeddings.

    Then, identify the most important feature by comparing the coefficients for a logistic regression model that has been mapped to the data.
    Next, for all the patches that assay the top gene, see which 2 patches have the highest value and which 2 patches have the lowest values
    for that features

    :param ts: str. Timestamp to indicate which set of embeddings to use.
    :param top_gene: str. The gene with the highest AUC score from the logistic regression binary classifier that predicts case vs control
    :param path_to_labels_file: str. Path to the labels csv file. Labels are True/False (case/control) on a donor level.
    :param path_to_patch_level_embeddings: str. Path to the patch level sz embeddings
    :return: None
    """
    path_to_per_gene_per_donor_file = os.path.join(sz_general_path,"per_gene_per_donor", ts + "_embed_per_gene_per_donor", top_gene + ".csv")
    max_feature, max_score = feature_importance_with_lr(path_to_per_gene_per_donor_file, path_to_labels_file, standardize = False)

    print ('MAX')
    print (max_feature)
    path_to_info_file = os.path.join(sz_general_path, "human_ISH_info.csv")
    get_feature_value_from_patch_level(path_to_info_file,path_to_patch_level_embeddings, top_gene, max_feature)



def feature_importance_with_lr(path_to_embed_file, path_to_labels_file, standardize):

    if standardize == False:
        embed_df = pd.read_csv(path_to_embed_file)

    else:
        embed_df = pd.read_csv(path_to_embed_file)
        donor_id_col = embed_df['donor_id']
        embed_df = embed_df.drop(columns=['donor_id'])
        embed_df = embed_df.apply(lambda  x: (x - x.mean())/(x.std()), axis=1)
        embed_df['donor_id'] = donor_id_col

    label_df = pd.read_csv(path_to_labels_file)

    label_df = label_df.rename(columns={'ID': 'donor_id'})
    print(embed_df.head())
    print(label_df.head())

    left = embed_df
    right = label_df

    label_df =  pd.merge(left, right, how='left', on='donor_id')[['donor_id','disease_diagnosis']]

    embed_df = embed_df.drop(columns=['donor_id'])
    label_df = label_df.drop(columns=['donor_id'])



    X = embed_df
    Y = label_df

    print (embed_df.head())
    print (label_df.head())

    model = LogisticRegression()
    # fit the model
    model.fit(X, Y)
    # get importance
    importance = model.coef_[0]
    # summarize feature importance

    max_score = math.inf * -1
    max_feature = None

    for i, v in enumerate(importance):
        print('Feature: {}, Score: {}'.format(i, abs(v)))
        if abs(v) >max_score:
            max_score = abs(v)
            max_feature = i

    print ("----")
    print ("Max:" , max_feature, max_score)

    return (max_feature, max_score)



def get_feature_value_from_patch_level(path_to_info_file, path_to_patch_level_embeddings, top_gene, max_feature):

    patch_level_embed_df = pd.read_csv(path_to_patch_level_embeddings)

    info_df = pd.read_csv(path_to_info_file)

    # ---- among all patches --------

    info_of_top_gene_df = info_df[info_df['gene_symbol']==top_gene]
    images_of_top_gene = list(info_of_top_gene_df['image_id'])
    images_of_top_gene = {str(item):None for item in images_of_top_gene}


    all_patches = list(patch_level_embed_df['image_id'])

    patches_of_top_gene = []
    for patch in all_patches:
        image_segment = patch.split("_")[0]
        if image_segment in images_of_top_gene:
            patches_of_top_gene.append(patch)

    patch_level_embed_of_top_gene = patch_level_embed_df[patch_level_embed_df['image_id'].isin(patches_of_top_gene)]



    max_feature_col_df = patch_level_embed_of_top_gene[['image_id', str(max_feature)]]

    patch_and_max_feature_df = max_feature_col_df.sort_values(by=[str(max_feature)], ascending=False)

    print("------ ALL -------")

    print (patch_and_max_feature_df.head())
    print (patch_and_max_feature_df.iloc[0])
    print(patch_and_max_feature_df.iloc[1])
    print(patch_and_max_feature_df.iloc[2])
    print(patch_and_max_feature_df.iloc[3])
    print(patch_and_max_feature_df.iloc[4])


    print(patch_and_max_feature_df.iloc[-1])
    print(patch_and_max_feature_df.iloc[-2])
    print(patch_and_max_feature_df.iloc[-3])
    print(patch_and_max_feature_df.iloc[-4])
    print(patch_and_max_feature_df.iloc[-5])

    """
    
    image_id    101291310_3    0.00133308

    image_id    81335473_21    0.00129161

    image_id    101057447_10   -0.00808748

    image_id    81348503_21    -0.00780414
        
    
    """




    # ---- filter case and control --------

    info_of_top_gene_df = info_df[info_df['gene_symbol'] == top_gene]
    info_of_top_gene_df_case = info_of_top_gene_df[info_of_top_gene_df['description'] == 'disease categories - schizophrenia']
    info_of_top_gene_df_control = info_of_top_gene_df[info_of_top_gene_df['description'] == 'disease categories - control']

    images_of_top_gene_case = list(info_of_top_gene_df_case['image_id'])
    images_of_top_gene_case = {str(item): None for item in images_of_top_gene_case}

    images_of_top_gene_control = list(info_of_top_gene_df_control['image_id'])
    images_of_top_gene_control = {str(item): None for item in images_of_top_gene_control}

    all_patches = list(patch_level_embed_df['image_id'])

    patches_of_top_gene_case = []
    patches_of_top_gene_control = []

    for patch in all_patches:
        image_segment = patch.split("_")[0]
        if image_segment in images_of_top_gene_case:
            patches_of_top_gene_case.append(patch)

        if image_segment in images_of_top_gene_control:
            patches_of_top_gene_control.append(patch)


    patch_level_embed_of_top_gene_case = patch_level_embed_df[patch_level_embed_df['image_id'].isin(patches_of_top_gene_case)]

    patch_level_embed_of_top_gene_control = patch_level_embed_df[
        patch_level_embed_df['image_id'].isin(patches_of_top_gene_control)]

    max_feature_col_df_case = patch_level_embed_of_top_gene_case[['image_id', str(max_feature)]]
    max_feature_col_df_control = patch_level_embed_of_top_gene_control[['image_id', str(max_feature)]]

    patch_and_max_feature_df_case = max_feature_col_df_case.sort_values(by=[str(max_feature)], ascending=False)
    patch_and_max_feature_df_control = max_feature_col_df_control.sort_values(by=[str(max_feature)], ascending=False)

    print ("------ CASE -------")

    print(patch_and_max_feature_df_case.head())
    print(patch_and_max_feature_df_case.iloc[0])
    print(patch_and_max_feature_df_case.iloc[1])
    print(patch_and_max_feature_df_case.iloc[2])
    print(patch_and_max_feature_df_case.iloc[3])
    print(patch_and_max_feature_df_case.iloc[4])

    print(patch_and_max_feature_df_case.iloc[-1])
    print(patch_and_max_feature_df_case.iloc[-2])
    print(patch_and_max_feature_df_case.iloc[-3])
    print(patch_and_max_feature_df_case.iloc[-4])
    print(patch_and_max_feature_df_case.iloc[-5])


    """
    image_id    101291310_3    0.00133308
    image_id    81112230_31    0.000201117
    image_id    81430398_47    -0.00756251
    image_id    81239257_17    -0.00745863
    
    """



    print("------ CONTROL -------")

    print(patch_and_max_feature_df_control.head())
    print(patch_and_max_feature_df_control.iloc[0])
    print(patch_and_max_feature_df_control.iloc[1])
    print(patch_and_max_feature_df_control.iloc[2])
    print(patch_and_max_feature_df_control.iloc[3])
    print(patch_and_max_feature_df_control.iloc[4])

    print(patch_and_max_feature_df_control.iloc[-1])
    print(patch_and_max_feature_df_control.iloc[-2])
    print(patch_and_max_feature_df_control.iloc[-3])
    print(patch_and_max_feature_df_control.iloc[-4])
    print(patch_and_max_feature_df_control.iloc[-5])

    """
    image_id    81335473_21    0.00129161
    image_id    81173430_38    0.00115357
    image_id    101057447_10   -0.00808748
    image_id    81348503_21    -0.00780414
    
    """


def get_feature_value_from_patch_level_all_genes(path_to_patch_level_embeddings, max_feature):
    patch_level_embed_df = pd.read_csv(path_to_patch_level_embeddings)
    print ("000" *20)
    print (patch_level_embed_df.head())
    print (len(patch_level_embed_df))
    print("000" * 20)

    # ---- among all patches --------

    max_feature_col_df = patch_level_embed_df[['image_id', str(max_feature)]]

    patch_and_max_feature_df = max_feature_col_df.sort_values(by=[str(max_feature)], ascending=False)

    print ("MAX FEATURE: ", max_feature)
    print("------ ALL -------")

    print(patch_and_max_feature_df.head())
    print(patch_and_max_feature_df.iloc[0])
    print(patch_and_max_feature_df.iloc[1])
    print(patch_and_max_feature_df.iloc[2])
    print(patch_and_max_feature_df.iloc[3])
    print(patch_and_max_feature_df.iloc[4])

    print(patch_and_max_feature_df.iloc[-1])
    print(patch_and_max_feature_df.iloc[-2])
    print(patch_and_max_feature_df.iloc[-3])
    print(patch_and_max_feature_df.iloc[-4])
    print(patch_and_max_feature_df.iloc[-5])






if __name__ == "__main__":


    #generate_random_embeddings_for_disease_dataset(128)
    #temp_function()

    """
    levels = ['gene']
    path_to_embeddings = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_3/1603427490"
    path_to_labels =  "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/sz"
    path_to_save_files = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/binary"
    sz_diagnosis_create_training_files(path_to_embeddings, path_to_labels, levels, path_to_save_files)
    
    """

    gene_type = "all_genes"
    input_type_list = ['embed','demog' , 'demog_and_embed', 'random', 'plain_resnet']
    #get_avg_AUCs(gene_type, input_type_list)

    #compare_scores(gene_type,'plain_resnet', 'embed', 'lr')
    print ("----")

    #compare_scores(gene_type, 'plain_resnet', 'demog', 'lr')
    print("----")

    #compare_scores(gene_type, 'plain_resnet', 'demog_and_embed', 'lr')
    print("----")



    #genes_common_in_all = check_genes_and_donors()

    list_of_columns_to_get = ['donor_age', 'pmi', 'tissue_ph', 'smoker', 'donor_race','donor_sex']
    without = []
    ts = "1603427156"  # with SZ

    #demog_info_as_training(list_of_columns_to_get, ts)


    #--------- Cortex embeddings patch activation ---------------
    path_to_patch_level_embeddings = os.path.join("/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_3/1603427156",
                                                 "triplet_all_training_embeddings.csv" )
    max_feature = 116
    get_feature_value_from_patch_level_all_genes(path_to_patch_level_embeddings, max_feature)



    labels = ['donor_age', 'donor_sex', 'smoker', 'pmi', 'tissue_ph', 'donor_race']
    #for label in labels:
        #get_sz_labels_image_and_donor_level(label)

    ts = "1603427156"  # with SZ
    #demog_info_as_training(labels, ts)

    gene_types =  []#['all_genes'] #,'top_20_genes']
    input_types = []#['embed' ,'demog' , 'demog_and_embed' , 'random', 'plain_resnet']#, 'demog_without_smoker',
                   #'demog_and_embed_without_smoker', 'demog_without_sex', 'demog_and_embed_without_sex']

    rnd_state = 6


    rf_auc_dict= {}
    lr_auc_dict = {}

    for input_type in input_types:
        for gene_type in gene_types:
            print ("____" * 50)
            print (input_type)
            print("____" * 50)

            gene_type_path = os.path.join(sz_general_path, gene_type)
            if (not os.path.exists(gene_type_path)):
                os.mkdir(gene_type_path)

            input_type_path = os.path.join(sz_general_path, gene_type, input_type)
            if (not os.path.exists(input_type_path)):
                os.mkdir(input_type_path)

            if input_type == 'embed':



                # ------ per gene per donor ---------------

                """

                path_to_embeddings = os.path.join(general_path, "dummy_3", ts,
                                                  "triplet_patches_schizophrenia_embeddings_image_level.csv")
                embeddings_df = pd.read_csv(path_to_embeddings)

                if gene_type == "all_genes":
                    genes = embeddings_per_gene_per_donor(input_type, ts, embeddings_df)
                elif gene_type == "top_20_genes":
                    genes = genes_common_in_all


                diagnosis_prediction_res = perform_logistic_regression_per_gene_per_donor(input_type,ts, genes,'lr',rnd_state, n_splits=5, n_jobs=1)
    
                diagnosis_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                             ts + "_"+input_type+"_per_gene_per_donor_diagnosis_prediction_scores_"
                                                             +"lr"+ "_rnd_state_" + str(rnd_state)+".csv"), index=False)

                diagnosis_prediction_res = perform_logistic_regression_per_gene_per_donor(input_type, ts, genes, 'rf', rnd_state,
                                                                                          n_splits=5, n_jobs=1)

                diagnosis_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                             ts + "_" + input_type + "_per_gene_per_donor_diagnosis_prediction_scores_" + "rf" + "_rnd_state_" + str(rnd_state)+ ".csv"),
                                                index=False)
                                                

                """



                # ------ donor level ---------------------

                path_to_embed_file = os.path.join(general_path, "dummy_3", ts,"triplet_patches_schizophrenia_embeddings_" + "donor" + "_level.csv")
                path_to_labels_file = os.path.join(sz_general_path, "sz_diagnosis_" + "donor" + "_level.csv")
                path_to_patch_level_embeddings = os.path.join(general_path, "dummy_3", ts, "triplet_patches_schizophrenia_embeddings.csv")

                #donor_level_prediction_res=perform_logistic_regression(path_to_embed_file, path_to_labels_file,  "donor", n_splits=5, n_jobs=1)
                #donor_level_prediction_res.to_csv(os.path.join(input_type_path, ts+"_" + gene_type + "_" +input_type + "_per_donor_diagnosis_prediction_scores.csv"),index=False)

                #lr = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "donor", n_splits=5, n_jobs=1)
                #lr_auc_dict[input_type] = lr

                #rf = perform_random_forest(path_to_embed_file, path_to_labels_file, 'donor')
                #rf_auc_dict[input_type] = rf

                get_patches_that_activate_neuron_the_most_and_the_least(ts, 'SCN4B', path_to_labels_file,
                                                                        path_to_patch_level_embeddings)
                
                





                """
                # ---- image level ---------------------
    
                path_to_embed_file = os.path.join(general_path, "dummy_3", ts,"triplet_patches_schizophrenia_embeddings_" + "image" + "_level.csv")
                path_to_labels_file = os.path.join(sz_general_path, "sz_diagnosis_" + "image" + "_level.csv")
    
                image_level_prediction_res=perform_logistic_regression(path_to_embed_file, path_to_labels_file, "image", n_splits=5, n_jobs=1)
                #image_level_prediction_res.to_csv(os.path.join(input_type_path, ts+ "_" + input_type + "_per_image_diagnosis_prediction_scores.csv"),index=False)
    
                """


            elif input_type == 'demog':
                path_to_embeddings = os.path.join(sz_general_path, ts + "_demog_info_as_training_image_level.csv")
                embeddings_df = pd.read_csv(path_to_embeddings)



                # ------ per gene per donor ---------------
                if gene_type == "all_genes":
                    genes = embeddings_per_gene_per_donor(input_type, ts, embeddings_df)
                elif gene_type == "top_20_genes":
                    genes = genes_common_in_all



                diagnosis_prediction_res = perform_logistic_regression_per_gene_per_donor(input_type, ts, genes, 'lr',rnd_state, n_splits=5, n_jobs=1)
                diagnosis_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                             ts + "_" + input_type + "_per_gene_per_donor_diagnosis_prediction_scores_"+ "lr"+ "_rnd_state_" + str(rnd_state)+".csv"),
                                                index=False)

                diagnosis_prediction_res = perform_logistic_regression_per_gene_per_donor(input_type, ts, genes, 'rf', rnd_state,
                                                                                          n_splits=5, n_jobs=1)

                diagnosis_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                             ts + "_" + input_type + "_per_gene_per_donor_diagnosis_prediction_scores_" + "rf" + "_rnd_state_" + str(rnd_state)+ ".csv"),
                                              index=False)
                                              



                """
                # ------ donor level ---------------------

                path_to_embed_file = path_to_embeddings
                path_to_labels_file = os.path.join(sz_general_path, "sz_diagnosis_" + "donor" + "_level.csv")

                #donor_level_prediction_res = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "donor",n_splits=5, n_jobs=1)
                #donor_level_prediction_res.to_csv(os.path.join(input_type_path, ts + "_"  + gene_type + "_" + input_type + "_per_donor_diagnosis_prediction_scores.csv"),index=False)

                #lr = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "donor", n_splits=5, n_jobs=1)
                #lr_auc_dict[input_type] = lr

                rf = perform_random_forest(path_to_embed_file, path_to_labels_file, 'donor')
                rf_auc_dict[input_type] = rf
                """

            elif input_type == 'demog_and_embed':

                path_to_embeddings = os.path.join(sz_general_path, ts + "_demog_info_and_embeddings_as_training_image_level.csv")
                embeddings_df = pd.read_csv(path_to_embeddings)



                # ------ per gene per donor ---------------
                if gene_type == "all_genes":
                    genes = embeddings_per_gene_per_donor(input_type, ts, embeddings_df)
                elif gene_type == "top_20_genes":
                    genes = genes_common_in_all




                diagnosis_prediction_res = perform_logistic_regression_per_gene_per_donor(input_type, ts, genes, 'lr', rnd_state, n_splits=5, n_jobs=1)
                diagnosis_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                             ts + "_" + input_type + "_per_gene_per_donor_diagnosis_prediction_scores_" + "lr"+ "_rnd_state_" + str(rnd_state)+ ".csv"),
                                                index=False)

                diagnosis_prediction_res = perform_logistic_regression_per_gene_per_donor(input_type, ts, genes, 'rf', rnd_state,
                                                                                          n_splits=5, n_jobs=1)

                diagnosis_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                             ts + "_" + input_type + "_per_gene_per_donor_diagnosis_prediction_scores_" + "rf" + "_rnd_state_" + str(rnd_state)+ ".csv"),
                                                index=False)


                """

                # ------ donor level ---------------------

                path_to_embed_file = path_to_embeddings
                path_to_labels_file = os.path.join(sz_general_path, "sz_diagnosis_" + "donor" + "_level.csv")

                #donor_level_prediction_res = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "donor",n_splits=5, n_jobs=1)
                #donor_level_prediction_res.to_csv(os.path.join(input_type_path, ts +  "_" + gene_type + "_" + input_type + "_per_donor_diagnosis_prediction_scores.csv"),index=False)

                #lr = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "donor", n_splits=5, n_jobs=1)
                #lr_auc_dict[input_type] = lr

                rf = perform_random_forest(path_to_embed_file, path_to_labels_file, 'donor')
                rf_auc_dict[input_type] = rf
                """

            elif input_type == 'random':


                # ------ per gene per donor ---------------
                path_to_embeddings = os.path.join(sz_general_path, 'random',  "random_embeddings_image_level.csv")
                embeddings_df = pd.read_csv(path_to_embeddings)

                embeddings_df = embeddings_df.rename(columns={'id': 'image_id'})

                if gene_type == "all_genes":
                    genes = embeddings_per_gene_per_donor(input_type, ts, embeddings_df)
                elif gene_type == "top_20_genes":
                    genes = genes_common_in_all



                diagnosis_prediction_res = perform_logistic_regression_per_gene_per_donor(input_type, ts, genes, 'lr', rnd_state, n_splits=5, n_jobs=1)

                diagnosis_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                              input_type+"_per_gene_per_donor_diagnosis_prediction_scores_" + "lr" + "_rnd_state_" + str(rnd_state)+ ".csv"), index=False)

                diagnosis_prediction_res = perform_logistic_regression_per_gene_per_donor(input_type, ts, genes, 'rf', rnd_state,
                                                                                          n_splits=5, n_jobs=1)

                diagnosis_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                             input_type + "_per_gene_per_donor_diagnosis_prediction_scores_" + "rf" + "_rnd_state_" + str(rnd_state)+ ".csv"),
                                                index=False)


                """
                # ------ donor level ---------------------

                path_to_embed_file = os.path.join(sz_general_path, 'random' ,"random_embeddings_donor_level_2.csv")
                path_to_labels_file = os.path.join(sz_general_path, "sz_diagnosis_" + "donor" + "_level.csv")

                #donor_level_prediction_res=perform_logistic_regression(path_to_embed_file, path_to_labels_file,  "donor", n_splits=5, n_jobs=1)
                #donor_level_prediction_res.to_csv(os.path.join(input_type_path, gene_type + "_" + input_type + "_per_donor_diagnosis_prediction_scores_2.csv"),index=False)

                #lr = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "donor", n_splits=5, n_jobs=1)
                #lr_auc_dict[input_type] = lr

                rf = perform_random_forest(path_to_embed_file, path_to_labels_file, 'donor')
                rf_auc_dict[input_type] = rf
                
                
                """

                """
                # ---- image level ---------------------

                path_to_embed_file = os.path.join(sz_general_path, 'random', "random_embeddings_image_level.csv")
                path_to_labels_file = os.path.join(sz_general_path, "sz_diagnosis_" + "image" + "_level.csv")

                image_level_prediction_res=perform_logistic_regression(path_to_embed_file, path_to_labels_file, "image", n_splits=5, n_jobs=1)
                #image_level_prediction_res.to_csv(os.path.join(input_type_path, input_type + "_per_image_diagnosis_prediction_scores.csv"),index=False)
                """



            elif input_type == 'plain_resnet':


                """
                # ------ per gene per donor ---------------
                path_to_embeddings = os.path.join(sz_general_path, 'plain_resnet', "resnet50_embeddings_image_level.csv")
                embeddings_df = pd.read_csv(path_to_embeddings)

                if gene_type == "all_genes":
                    genes = embeddings_per_gene_per_donor(input_type, ts, embeddings_df)
                elif gene_type == "top_20_genes":
                    genes = genes_common_in_all

                diagnosis_prediction_res = perform_logistic_regression_per_gene_per_donor(input_type, ts, genes,'lr',rnd_state,  n_splits=5,
                                                                                          n_jobs=1)

                diagnosis_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                             input_type + "_per_gene_per_donor_diagnosis_prediction_scores_" + "lr" + "_rnd_state_" + str(rnd_state)+ ".csv"),
                                                index=False)

                diagnosis_prediction_res = perform_logistic_regression_per_gene_per_donor(input_type, ts, genes, 'rf', rnd_state,
                                                                                          n_splits=5, n_jobs=1)

                diagnosis_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                             input_type + "_per_gene_per_donor_diagnosis_prediction_scores_" + "rf" + "_rnd_state_" + str(rnd_state)+ ".csv"),
                                                index=False)

                """


                # ------ donor level ---------------------

                path_to_embed_file = os.path.join(sz_general_path, 'plain_resnet', "resnet50_embeddings_donor_level.csv")
                path_to_labels_file = os.path.join(sz_general_path, "sz_diagnosis_" + "donor" + "_level.csv")
                path_to_patch_level_embeddings = os.path.join("/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_3/patch_max_feature/sz",
                                                              "resnet50_embeddings.csv")

                #donor_level_prediction_res = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "donor",n_splits=5, n_jobs=1)
                #donor_level_prediction_res.to_csv(os.path.join(input_type_path, gene_type + "_" +input_type + "_per_donor_diagnosis_prediction_scores.csv"),index=False)

                #lr = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "donor", n_splits=5, n_jobs=1)
                #lr_auc_dict[input_type] = lr

                #rf = perform_random_forest(path_to_embed_file, path_to_labels_file, 'donor')
                #rf_auc_dict[input_type] = rf

                get_patches_that_activate_neuron_the_most_and_the_least(ts, 'NEFH', path_to_labels_file,
                                                                        path_to_patch_level_embeddings)


                """
                # ---- image level ---------------------

                path_to_embed_file = os.path.join(sz_general_path, 'plain_resnet', "resnet50_embeddings_image_level.csv")
                path_to_labels_file = os.path.join(sz_general_path, "sz_diagnosis_" + "image" + "_level.csv")

                image_level_prediction_res = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "image",
                                                                         n_splits=5, n_jobs=1)
                #image_level_prediction_res.to_csv(os.path.join(input_type_path, input_type + "_per_image_diagnosis_prediction_scores.csv"),index=False)


                """




            elif input_type == 'demog_without_smoker':
                path_to_embeddings = os.path.join(sz_general_path, ts + "_without_smoker" +"_demog_info_as_training_donor_level.csv")
                embeddings_df = pd.read_csv(path_to_embeddings)

                """
                # ------ per gene per donor ---------------
                #genes = embeddings_per_gene_per_donor(input_type, ts, embeddings_df)
                genes = genes_common_in_all
                diagnosis_prediction_res = perform_logistic_regression_per_gene_per_donor(input_type, ts, genes, n_splits=5, n_jobs=1)
                diagnosis_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                             ts + "_" + input_type + "_per_gene_per_donor_diagnosis_prediction_scores.csv"),
                                                index=False)

                """
                """
                # ------ donor level ---------------------

                print ("HEREEEE")

                path_to_embed_file = path_to_embeddings
                path_to_labels_file = os.path.join(sz_general_path, "sz_diagnosis_" + "donor" + "_level.csv")

                #donor_level_prediction_res = perform_logistic_regression(path_to_embed_file, path_to_labels_file,"donor",n_splits=5, n_jobs=1)

                #donor_level_prediction_res.to_csv(os.path.join(input_type_path,ts + "_" + gene_type + "_" + input_type + "_per_donor_diagnosis_prediction_scores.csv"),index=False)

                #lr = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "donor", n_splits=5, n_jobs=1)
                #lr_auc_dict[input_type] = lr


                rf = perform_random_forest(path_to_embed_file, path_to_labels_file, 'donor')
                rf_auc_dict[input_type] = rf

                """


            elif input_type ==  'demog_and_embed_without_smoker':
                path_to_embeddings = os.path.join(sz_general_path,
                                                  ts + "_without_smoker" + "_demog_info_and_embeddings_as_training_donor_level.csv")
                embeddings_df = pd.read_csv(path_to_embeddings)

                """

                # ------ per gene per donor ---------------
                #genes = embeddings_per_gene_per_donor(input_type, ts, embeddings_df)
                genes= genes_common_in_all
                diagnosis_prediction_res = perform_logistic_regression_per_gene_per_donor(input_type, ts, genes, n_splits=5, n_jobs=1)
                diagnosis_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                             ts + "_" + input_type + "_per_gene_per_donor_diagnosis_prediction_scores.csv"),
                                                index=False)

                """


                """
                # ------ donor level ---------------------

                path_to_embed_file = path_to_embeddings
                path_to_labels_file = os.path.join(sz_general_path, "sz_diagnosis_" + "donor" + "_level.csv")

                #donor_level_prediction_res = perform_logistic_regression(path_to_embed_file, path_to_labels_file,"donor",n_splits=5, n_jobs=1)
                #donor_level_prediction_res.to_csv(os.path.join(input_type_path,ts + "_" + gene_type + "_" + input_type + "_per_donor_diagnosis_prediction_scores.csv"),index=False)

                lr = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "donor", n_splits=5, n_jobs=1)
                lr_auc_dict[input_type] = lr

                #rf = perform_random_forest(path_to_embed_file, path_to_labels_file, 'donor')
                #rf_auc_dict[input_type] = rf
                
                """



            elif input_type == 'demog_without_sex':
                path_to_embeddings = os.path.join(sz_general_path, ts + "_without_donor_sex" +"_demog_info_as_training_donor_level.csv")
                embeddings_df = pd.read_csv(path_to_embeddings)

                """
                # ------ per gene per donor ---------------
                #genes = embeddings_per_gene_per_donor(input_type, ts, embeddings_df)
                genes = genes_common_in_all
                diagnosis_prediction_res = perform_logistic_regression_per_gene_per_donor(input_type, ts, genes, n_splits=5, n_jobs=1)
                diagnosis_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                             ts + "_" + input_type + "_per_gene_per_donor_diagnosis_prediction_scores.csv"),
                                                index=False)

                """
                """
                # ------ donor level ---------------------


                path_to_embed_file = path_to_embeddings
                path_to_labels_file = os.path.join(sz_general_path, "sz_diagnosis_" + "donor" + "_level.csv")

                #donor_level_prediction_res = perform_logistic_regression(path_to_embed_file, path_to_labels_file,"donor",n_splits=5, n_jobs=1)

                #donor_level_prediction_res.to_csv(os.path.join(input_type_path,ts + "_" + gene_type + "_" + input_type + "_per_donor_diagnosis_prediction_scores.csv"),index=False)

                #lr = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "donor", n_splits=5, n_jobs=1)
                #lr_auc_dict[input_type] = lr

                rf = perform_random_forest(path_to_embed_file, path_to_labels_file, 'donor')
                rf_auc_dict[input_type] = rf
                """


            elif input_type ==  'demog_and_embed_without_sex':
                path_to_embeddings = os.path.join(sz_general_path,
                                                  ts + "_without_donor_sex" + "_demog_info_and_embeddings_as_training_donor_level.csv")
                embeddings_df = pd.read_csv(path_to_embeddings)

                """

                # ------ per gene per donor ---------------
                #genes = embeddings_per_gene_per_donor(input_type, ts, embeddings_df)
                genes= genes_common_in_all
                diagnosis_prediction_res = perform_logistic_regression_per_gene_per_donor(input_type, ts, genes, n_splits=5, n_jobs=1)
                diagnosis_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                             ts + "_" + input_type + "_per_gene_per_donor_diagnosis_prediction_scores.csv"),
                                                index=False)

                """

                """

                # ------ donor level ---------------------

                path_to_embed_file = path_to_embeddings
                path_to_labels_file = os.path.join(sz_general_path, "sz_diagnosis_" + "donor" + "_level.csv")

                #donor_level_prediction_res = perform_logistic_regression(path_to_embed_file, path_to_labels_file,"donor",n_splits=5, n_jobs=1)
                #donor_level_prediction_res.to_csv(os.path.join(input_type_path,ts + "_" + gene_type + "_" + input_type + "_per_donor_diagnosis_prediction_scores.csv"),index=False)

                #lr = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "donor", n_splits=5, n_jobs=1)
                #lr_auc_dict[input_type] = lr

                rf = perform_random_forest(path_to_embed_file, path_to_labels_file, 'donor')
                rf_auc_dict[input_type] = rf

                """


    print (rf_auc_dict)
    print ("@" *20)
    print (lr_auc_dict)







