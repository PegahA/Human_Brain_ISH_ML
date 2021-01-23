
import pandas as pd
import numpy as np
import random
from human_ISH_config import *
import math
import os
import sklearn


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier


#DATA_DIR: This is defined in human_ISJ_config.py. This is the directory you have defined to store all the data.

PATH_TO_SZ_STUDY = os.path.join(DATA_DIR, "schizophrenia")
PATH_TO_SZ_POST_PROCESS = os.path.join(PATH_TO_SZ_STUDY, "post_process_on_sz")


def get_sz_labels_image_and_donor_level(label):
    """
    This function is used to select a certain column from the info csv file to be later used as a label in downstream tasks.
    The main columns that we were interested are: "description" and "smoker"
    "description" indicates whether the donor was case or control, and "smoker" indicates whether they smoked or not.
    This information is available from the Allen website.

    :param label: string. The column name to be used as label
    :return: None
    """


    path_to_sz_info = os.path.join(PATH_TO_SZ_STUDY, "human_ISH_info.csv")
    sz_info_df = pd.read_csv(path_to_sz_info)



    if label == 'description':

        new_df = pd.DataFrame(columns=['ID', label])

        # --------------- image level ---------------
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
        file_name = "sz_diagnosis_as_label_image_level.csv"
        new_df.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, file_name), index=None)

        print ("image sz count: ", image_sz_count)
        print ("image no sz count: ", image_no_sz_count)
        print ("total: ", image_sz_count + image_no_sz_count)


        # --------------- donor level ---------------
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

        file_name = "sz_diagnosis_as_label_donor_level.csv"
        new_df.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, file_name), index=None)

        print ("donor sz count: ", donor_sz_count)
        print ("donor no sz count: ", donor_no_sz_count)
        print ("total: ", donor_sz_count + donor_no_sz_count)



    elif label in ['donor_age', 'donor_sex', 'smoker', 'pmi', 'tissue_ph', 'donor_race']:
        new_df = pd.DataFrame(columns=['ID', label])

        # --------------- image level ---------------
        new_df['ID'] = sz_info_df['image_id']
        new_df[label] = list(sz_info_df[label])

        file_name = label + "_as_label_image_level.csv"
        new_df.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, file_name), index=None)

        # --------------- donor level ---------------
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
        new_df.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, file_name), index=None)


def embeddings_per_gene_per_donor(path_to_per_gene_per_donor_level_files, input_type, ts, embeddings_df):
    """
    This function gets an image-level embedding file and outputs a donor-level csv file for each gene.
    Each gene will have a separate csv file: gene_name.csv
    Each row in the csv file will represent a donor.
    The number of rows in the csv file is the number of donors on which this specific gene was tested.

    We will use image level embeddings, then group them by gene. So each group will be all the images that assay the same gene.
    Then, within each group, we will group the images again by donor_id and use the mean() function to take the average of the embeddings.

    :param path_to_per_gene_per_donor_level_files: the path in which per gene donor-level files should be saved.
    The directory will be created if it doesn't alredy exist.

    :param input_type: str. Determine the type of input vectors.
    Could be: ['embed','demog','demog_and_embed','random','plain_resnet']

    :param ts: str. The timestamp that indicates which files to use.
    :param embeddings_df: pandas data frame. Image-level embeddings.
    :return: a list of genes
    """

    # the embeddings are image level
    path_to_sz_info = os.path.join(PATH_TO_SZ_STUDY, "human_ISH_info.csv")
    sz_info_df = pd.read_csv(path_to_sz_info)

    # add two extra columns: gene_symbol and donor_id to the embeddings
    # if the file already has donor_id, don't add it
    left = embeddings_df
    left_cols = list(embeddings_df)
    right = sz_info_df
    if 'donor_id' in left_cols:
        merge_res = pd.merge(left, right[['image_id', 'gene_symbol']], how='left', on='image_id')
    else:

        merge_res = pd.merge(left, right[['image_id','gene_symbol', 'donor_id']], how='left', on='image_id')

    genes = list(merge_res['gene_symbol'].unique())

    if input_type == 'random' or input_type == 'resnet':
        # random and resnet do not require a timestamp

        if (not os.path.exists(path_to_per_gene_per_donor_level_files)):
            os.mkdir(path_to_per_gene_per_donor_level_files)

        per_gene_per_donor_path = os.path.join(path_to_per_gene_per_donor_level_files, input_type + "_per_gene_per_donor")
        if (not os.path.exists(per_gene_per_donor_path)):
            os.mkdir(per_gene_per_donor_path)

    else:

        if (not os.path.exists(path_to_per_gene_per_donor_level_files)):
            os.mkdir(path_to_per_gene_per_donor_level_files)

        per_gene_per_donor_path = os.path.join(path_to_per_gene_per_donor_level_files, ts+ "_" + input_type +"_per_gene_per_donor")
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



def demog_info_as_training(list_of_columns_to_get, path_to_image_level_embeddings,  ts):
    """
    For every image, it extracts the demographics info and adds them as new columns to the embeddings.
    For 'smoker', 'donor_sex', and 'donor_race', it performs one-hot coding.
    Everything needs to be image-level
    Once we have the image-level embeddings, we can then aggregate to donor-level

    :param path_to_image_level_embeddings: str. Patch to image-level embeddings. These are the embeddings that you want
    to concatenae the demographic info to.
    :param list_of_columns_to_get: list of demographic info columns to use
    :param ts: timestamp indicating which model's embeddings to use
    :return: None
    """

    path_to_sz_info = os.path.join(PATH_TO_SZ_STUDY, "human_ISH_info.csv")
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
        one_hot_dfs.append(item_one_hot)


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

    file_name =  ts+ "_demog_info_as_training_image_level.csv"
    demog_df.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, file_name), index=None)

    grouped_df = demog_df.groupby(['donor_id']).mean()
    grouped_df = grouped_df.drop(columns=['image_id'])

    file_name = ts + "_demog_info_as_training_donor_level.csv"
    grouped_df.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, file_name))

    # ---- merge with image-level embeddings ----

    embeds_df = pd.read_csv( path_to_image_level_embeddings)

    left = embeds_df
    right = demog_df

    merged_res = pd.merge(left, right, how='left', on='image_id')
    file_name = ts + "_demog_info_and_embeddings_as_training_image_level.csv"
    merged_res.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, file_name), index=None)

    grouped_df = merged_res.groupby(['donor_id']).mean()
    grouped_df = grouped_df.drop(columns=['image_id'])

    file_name = ts + "_demog_info_and_embeddings_as_training_donor_level.csv"
    grouped_df.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, file_name))

    # -------------------------------------



def perform_logistic_regression(path_to_embed_file, path_to_labels_file,level, n_splits =5, n_jobs = 1):
    """

    This function performs logistic regression on a given dataset.

    :param path_to_embed_file: str. Path to the donor-level embedding file that will be used as training input for the logistic regression model.
    :param path_to_labels_file: str. Path to the file that will be used as ground truth lables for the logistic regression model.
    :param level: the aggregation level of embeddings. It could be 'donor', 'gene', or 'image'
    :param n_splits: number of splits for cross-validation
    :param n_jobs: number of jobs
    :return: a pandas data frame that has the AUC and f1 score.
    """

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




        # Get metrics for each model
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probas[True])

        f1_score_values.append(f1)
        auc_values.append(auc)

        print ("Finished fold: ", i+1)

    print ("----" * 20)



    f1 = np.mean(f1_score_values)
    auc = np.mean(auc_values)


    print ("FINAL: ", f1, auc)

    measures = {'level': level,
                'f1': f1,
                'AUC': auc}


    scores.append(measures)

    return pd.DataFrame(scores,columns=['level', 'AUC', 'f1']).sort_values(by=['AUC'],ascending=False).reset_index().drop(columns=['index'])



def get_genes_common_in_all_donors_and_donors_common_in_all_genes():
    """
    This function returns a list of genes that have been tested in all donors, and a list of donors on whom all the genes have been tested.
    This information is useful for downstream tasks.
    In the sz dataset, not all genes have been tested on every donor, and not every donor's sample has been used for every gene.
    :return: 2 lists
    """

    path_to_sz_info = os.path.join(PATH_TO_SZ_STUDY, "human_ISH_info.csv")
    sz_info_df = pd.read_csv(path_to_sz_info)
    sz_donors_genes = sz_info_df[['donor_id','gene_symbol']]

    # ------------------
    group_by_genes = sz_donors_genes.groupby('gene_symbol')['donor_id'].apply(list).to_dict()

    potential_genes = {}  # a dictionary: keys are genes, and values are list of donors on whom that gene was tested

    for gene in group_by_genes:
        donors = group_by_genes[gene]
        if len(donors) < 40: # ignore genes that have less than 40 donors
            pass

        else:
            potential_genes[gene] = donors


    donors_list = [potential_genes[genes] for genes  in potential_genes]
    common_donors_in_all_genes = set.intersection(*map(set, donors_list))


    # ------------------

    group_by_donors = sz_donors_genes.groupby('donor_id')['gene_symbol'].apply(list).to_dict()

    potential_donors = {} # a dictionary: keys are donors, and values are list of genes that were tested on that donor

    for donor in group_by_donors:
        genes = group_by_donors[donor]

        potential_donors[donor] = genes


    genes_list = [potential_donors[donors] for donors in potential_donors]
    common_genes_in_all_donors = set.intersection(*map(set, genes_list))



    return common_genes_in_all_donors, common_donors_in_all_genes



def get_patches_that_activate_neuron_the_most_and_the_least(path_to_per_gene_per_donor_level_files, ts, top_gene, path_to_labels_file, path_to_patch_level_embeddings, k):
    """
    The goal of this function is to get the patch ID of patches of a certain gene that activate the most important embedding feature for sz diagnosis classification.

    First, get the donor level embeddings of the top gene (embeddings of the images that assay the top gene, aggregated to donor-level).
    This file is used as the data to get feature importance.

    Standardize the embeddings.

    Then, identify the most important embedding feature by comparing the coefficients for a logistic regression model that has been mapped to the data.
    Next, for all the patches that assay the top gene, see which k patches have the highest value and which k patches have the lowest values
    for that features

    :param path_to_per_gene_per_donor_level_files: path to where the per gene donor level files are stored.
    :param ts: str. Timestamp to indicate which set of embeddings to use.
    :param top_gene: str. The gene with the highest AUC score from the logistic regression binary classifier that predicts case vs control
    If it is set to None, it means we are not looking for patches of a certain gene, and all patches of all the genes should be looked into.

    :param path_to_labels_file: str. Path to the labels csv file. Labels are True/False (case/control) on a donor level.
    :param path_to_patch_level_embeddings: str. Path to the patch level sz embeddings
    :param k: number of most and least activating patches to return

    :return: None
    """
    path_to_per_gene_per_donor_file = os.path.join( path_to_per_gene_per_donor_level_files, ts + "_embed_per_gene_per_donor", top_gene + ".csv")
    max_feature, max_score = feature_importance_with_lr(path_to_per_gene_per_donor_file, path_to_labels_file, standardize = False)
    get_feature_value_from_patch_level(path_to_patch_level_embeddings, top_gene, max_feature, k)


def feature_importance_with_lr(path_to_embed_file, path_to_labels_file, standardize):
    """
    This function fits a logistic regression model on a set of data and uses the coefficients to determine the most important embedding feature.

    :param path_to_embed_file: str. Path to the donor-level embedding file that will be used as training input for the logistic regression model.
    :param path_to_labels_file: str. Path to the file that will be used as ground truth lables for the logistic regression model.
    :param standardize: boolean. Determines whether standardization needs to be performed on the training input.
    :return: a tuple: the most important feature and its associated score.
    """

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


    left = embed_df
    right = label_df

    label_df =  pd.merge(left, right, how='left', on='donor_id')[['donor_id','disease_diagnosis']]

    embed_df = embed_df.drop(columns=['donor_id'])
    label_df = label_df.drop(columns=['donor_id'])



    X = embed_df
    Y = label_df

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



def get_feature_value_from_patch_level(path_to_patch_level_embeddings, top_gene, max_feature, k):
    """
    Knowing the most important feature, this function will grab the k patches that activate that feature the most and the least.
    It will compare the patches in 3 ways:
    1.among all the patches
    2.among patches belonging to cases
    3.among patches belonging to controls

    :param path_to_patch_level_embeddings: str. Patch to patch level embeddings of the sz study.
    :param top_gene: str. Gene symbol of the specific gene that we are interested in.
    If it is set to None, it means we are not looking for patches of a certain gene, and all patches of all the genes should be looked into.
    :param max_feature: int. The most important feature according to logistic regression.
    :param k: the number of most activating and least activating patches to be returned. The exact number is 2k.
    :return:  6 lists: the k most activating patches among all patches, the k least activating patches among all patches,
    the k most activating patches among case patches, the k least activating patches among case patches,
    the k most activating patches among control patches, the k least activating patches among control patches,

    """

    patch_level_embed_df = pd.read_csv(path_to_patch_level_embeddings)

    info_df = pd.read_csv(os.path.join(PATH_TO_SZ_STUDY, "human_ISH_info.csv"))

    # ---- among all patches --------
    if top_gene == None:
        info_of_top_gene_df = info_df
    else:
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

    among_all_top_k = []
    among_all_bottom_k = []

    for i in range(k):
        among_all_top_k.append(patch_and_max_feature_df.iloc[i])
        among_all_bottom_k.append(patch_and_max_feature_df.iloc[i-1])



    # ---- filter case and control --------

    if top_gene == None:
        info_of_top_gene_df = info_df
    else:
        info_of_top_gene_df = info_df[info_df['gene_symbol']==top_gene]

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

    among_cases_top_k = []
    among_cases_bottom_k = []

    for i in range(k):
        among_cases_top_k.append(patch_and_max_feature_df_case.iloc[i])
        among_cases_bottom_k.append(patch_and_max_feature_df_case.iloc[i-1])



    print("------ CONTROL -------")

    among_controls_top_k = []
    among_controls_bottom_k = []

    for i in range(k):
        among_controls_top_k.append(patch_and_max_feature_df_control.iloc[i])
        among_controls_bottom_k.append(patch_and_max_feature_df_control.iloc[i - 1])


    return among_all_top_k, among_all_bottom_k, among_cases_top_k, among_cases_bottom_k, among_controls_top_k, among_controls_bottom_k



def get_sz_study_donors_with_unknown_smoking_status():
    """
    This function returns donors with "unknown" smoking status in the info file.
    :return: np array
    """
    info_df = pd.read_csv(os.path.join(PATH_TO_SZ_STUDY, 'human_ISH_info.csv'))
    info_df_smoking_unk = info_df[info_df['smoker'] == 'unknown']
    unk_smokers = info_df_smoking_unk['donor_id'].unique()

    return unk_smokers



def perform_logistic_regresson_per_gene_donor_level_on_label(path_to_per_gene_per_donor_level_files, label, input_type, ts,
                                                           genes_list,  rnd_state, classifier ="lr" ,n_splits =5, n_jobs=1):
    """
    This function performs logistic regression per gene, per donor.
    It means that for a given list of genes, it grabs the donor-level csv file associated with that gene and performs the
    logistic regression on that file.

    :param path_to_per_gene_per_donor_level_files: path to where the per gene donor level files are stored.
    :param label: column to be used as label for prediction. It could be any of the columns in the human_ISH_info.csv file.
    e.g: 'smoker', 'donor sex', ...
    If you want to use the 'description' column which indicates cases vs controls, use "sz_diagnosis" as label.

    :param input_type: indicates the type of training info that will be used.
    Options are: ['embed','demog','demog_and_embed','random','plain_resnet']
    :param ts: str. Timestamp to indicate which set of embeddings to use.
    :param genes_list: list of genes to explore.
    :param rnd_state: random state
    :param classifier: it is 'lr' by default which is logistic regression. You could also use 'rf' which would be random forest.
    :param n_splits: number of splits for cross-validation
    :param n_jobs: number of jobs

    :return: a pandas data frame which will have the list of genes and their associated AUC and f1 score.
    """


    if label == "smoker":
        unk_smokers = get_sz_study_donors_with_unknown_smoking_status()

    invalid_genes_count = 0
    scores = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rnd_state)
    for gene in genes_list:

        print("Gene is: ", gene)

        if input_type == 'random' or input_type == 'resnet':
            path_to_donor_level_embeds = os.path.join(path_to_per_gene_per_donor_level_files, input_type + "_per_gene_per_donor",
                                                      gene + ".csv")
        else:
            path_to_donor_level_embeds = os.path.join(path_to_per_gene_per_donor_level_files,
                                                      ts + "_" + input_type + "_per_gene_per_donor", gene + ".csv")

        embeds = pd.read_csv(path_to_donor_level_embeds)



        if len(embeds) < 40:  # if this gene has been tested on less than 40 donors, ignore it
            invalid_genes_count += 1
            pass
        else:

            number_of_donors = len(embeds)
            left = embeds

            # label = "smoker"  "doner_sex"
            path_to_labels = os.path.join(PATH_TO_SZ_POST_PROCESS, label+ "_as_label_donor_level.csv")
            labels = pd.read_csv(path_to_labels)
            labels = labels.rename(columns={'ID': 'donor_id'})
            label_col = labels[label].to_numpy()

            if label == "smoker":
                label_col = np.where(label_col == "yes", True, False)
            elif label == "donor_sex":
                label_col = np.where(label_col == "F", True, False)

            labels[label] = label_col
            right = labels



            merge_res = pd.merge(left, right, how='left', on='donor_id')

            if label == "smoker":
                merge_res = merge_res[~merge_res['donor_id'].isin(unk_smokers)]

            col_titles = list(embeds)[1:]

            X = merge_res[col_titles]


            Y = merge_res[label]


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

                # Get metrics for each model
                f1 = f1_score(y_test, preds)
                auc = roc_auc_score(y_test, probas[True])

                f1_score_values.append(f1)
                auc_values.append(auc)

                print("Finished fold: ", i + 1)

            print("----" * 20)


            f1 = np.mean(f1_score_values)
            auc = np.mean(auc_values)

            measures = {'gene_symbol': gene,
                        'number_of_donors': number_of_donors,
                        'f1': f1,
                        'AUC': auc}

            scores.append(measures)

    return pd.DataFrame(scores,
                        columns=['gene_symbol', 'number_of_donors', 'AUC', 'f1']).sort_values(by=['AUC'],
                                                                                              ascending=False).reset_index().drop(
        columns=['index'])




if __name__ == "__main__":

    ts = "1603427156"  # a model that has been trained on all the images, with overlaps with SZ genes
    # options: ["1603427156", "1503427490", "1596183933", "1602225390"]

    # although we are processing schizophrenia embeddings, those embeddings have been saved in the cortex directory.
    # because the model that was used to generate the sz embeddings was trained on cortex data

    path_to_sz_image_level_embeddings = os.path.join(DATA_DIR, "cortex", "segmentation_embeddings", ts, "triplet_patches_schizophrenia_embeddings_image_level.csv")
    path_to_sz_donor_level_embeddings = os.path.join(DATA_DIR, "cortex", "segmentation_embeddings", ts, "triplet_patches_schizophrenia_embeddings_donor_level.csv")
    path_to_sz_patch_level_embeddings = os.path.join(DATA_DIR, "cortex", "segmentation_embeddings", ts, "triplet_patches_schizophrenia_embeddings.csv")

    path_to_per_gene_per_donor_level_files = os.path.join(PATH_TO_SZ_STUDY, "per_gene_donor_level_embeddings")

    genes_common_in_all_donors, donors_common_in_all_genes = get_genes_common_in_all_donors_and_donors_common_in_all_genes()

    list_of_demog_columns_to_get = ['donor_age', 'pmi', 'tissue_ph', 'smoker', 'donor_race','donor_sex']

    demog_info_as_training(list_of_demog_columns_to_get, path_to_sz_image_level_embeddings, ts)


    labels = ['description','smoker']   # options: ['description', 'donor_age', 'donor_sex', 'smoker', 'pmi', 'tissue_ph', 'donor_race']
    for label in labels:
        get_sz_labels_image_and_donor_level(label)



    gene_types = []#["all_genes", "genes_common_in_all_donors"]
    input_type_list = []#['embed', 'demog', 'demog_and_embed', 'random', 'plain_resnet']

    rnd_state = 1

    for input_type in input_type_list:
        for gene_type in gene_types:

            gene_type_path = os.path.join(PATH_TO_SZ_POST_PROCESS, gene_type)
            if (not os.path.exists(gene_type_path)):
                os.mkdir(gene_type_path)

            input_type_path = os.path.join(PATH_TO_SZ_POST_PROCESS, gene_type, input_type)
            if (not os.path.exists(input_type_path)):
                os.mkdir(input_type_path)

            if input_type == 'embed':

                # ------ per gene per donor ---------------
                embeddings_df = pd.read_csv(path_to_sz_image_level_embeddings)
                genes = embeddings_per_gene_per_donor(path_to_per_gene_per_donor_level_files, input_type, ts, embeddings_df)

                if gene_type == "all_genes":
                    genes = genes

                elif gene_type == "genes_common_in_all_donors":
                    genes = genes_common_in_all_donors

                target_label_list = ['smoker', 'donor_sex', 'disease_diagnosis']
                for target in target_label_list:
                    target_prediction_res = perform_logistic_regresson_per_gene_donor_level_on_label(target, input_type, ts, genes, "lr", rnd_state, n_splits =5, n_jobs=1)
                    target_prediction_res.to_csv(os.path.join(gene_type, input_type_path,
                                                             ts + "_"+input_type+"_per_gene_donor_level_" + target+ "_prediction_scores_"
                                                             +"lr"+ "_rnd_state_" + str(rnd_state)+".csv"), index=False)



                # ------ donor level ---------------------
                path_to_labels_file = os.path.join(PATH_TO_SZ_POST_PROCESS, "sz_diagnosis_as_label_donor_level.csv")

                donor_level_prediction_res=perform_logistic_regression(path_to_sz_donor_level_embeddings, path_to_labels_file,  "donor", n_splits=5, n_jobs=1)
                donor_level_prediction_res.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, input_type_path,
                                                               ts+"_" + gene_type + "_" +input_type + "_per_donor_diagnosis_prediction_scores.csv"),index=False)


                

            elif input_type == 'demog':

                # ------ per gene per donor ---------------
                path_to_embeddings = os.path.join(PATH_TO_SZ_POST_PROCESS, ts + "_demog_info_as_training_image_level.csv")
                embeddings_df = pd.read_csv(path_to_embeddings)

                if gene_type == "all_genes":
                    genes = embeddings_per_gene_per_donor(path_to_per_gene_per_donor_level_files, input_type, ts, embeddings_df)
                elif gene_type == "genes_common_in_all_donors":
                    genes = genes_common_in_all_donors


                diagnosis_prediction_res = perform_logistic_regresson_per_gene_donor_level_on_label('sz_diagnosis',input_type, ts, genes, 'lr',rnd_state, n_splits=5, n_jobs=1)
                diagnosis_prediction_res.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, input_type_path,
                                                             ts + "_" + input_type + "_per_gene_per_donor_diagnosis_prediction_scores_"+ "lr"+ "_rnd_state_" + str(rnd_state)+".csv"),
                                                index=False)


                # ------ donor level ---------------------

                path_to_embeddings = os.path.join(PATH_TO_SZ_POST_PROCESS, ts + "_demog_info_as_training_donor_level.csv")
                embeddings_df = pd.read_csv(path_to_embeddings)
                path_to_labels_file = os.path.join(PATH_TO_SZ_POST_PROCESS, "sz_diagnosis_as_label_donor_level.csv")

                donor_level_prediction_res = perform_logistic_regression(path_to_embeddings, path_to_labels_file, "donor",n_splits=5, n_jobs=1)
                donor_level_prediction_res.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, input_type_path, ts + "_"  + gene_type + "_" + input_type + "_per_donor_diagnosis_prediction_scores.csv"),index=False)



            elif input_type == 'demog_and_embed':

                # ------ per gene per donor ---------------
                path_to_embeddings = os.path.join(PATH_TO_SZ_POST_PROCESS,ts + "_demog_info_and_embeddings_as_training_image_level.csv")
                embeddings_df = pd.read_csv(path_to_embeddings)

                if gene_type == "all_genes":
                    genes = embeddings_per_gene_per_donor(path_to_per_gene_per_donor_level_files, input_type, ts, embeddings_df)
                elif gene_type == "genes_common_in_all_donors":
                    genes = genes_common_in_all_donors



                diagnosis_prediction_res =  perform_logistic_regresson_per_gene_donor_level_on_label('sz_diagnosis',input_type, ts, genes, 'lr', rnd_state, n_splits=5, n_jobs=1)
                diagnosis_prediction_res.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, input_type_path,
                                                             ts + "_" + input_type + "_per_gene_per_donor_diagnosis_prediction_scores_" + "lr"+ "_rnd_state_" + str(rnd_state)+ ".csv"),
                                                index=False)


                # ------ donor level ---------------------

                path_to_embeddings = os.path.join(PATH_TO_SZ_POST_PROCESS, ts + "_demog_info_and_embeddings_as_training_donor_level.csv")
                path_to_labels_file = os.path.join(PATH_TO_SZ_POST_PROCESS, "sz_diagnosis_as_label_donor_level.csv")

                donor_level_prediction_res = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "donor",n_splits=5, n_jobs=1)
                donor_level_prediction_res.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, input_type_path, ts +  "_" + gene_type + "_" + input_type + "_per_donor_diagnosis_prediction_scores.csv"),index=False)


            elif input_type == 'random':


                # ------ per gene per donor ---------------
                path_to_embeddings = os.path.join(PATH_TO_SZ_STUDY, 'random',  "random_embeddings_image_level.csv")
                embeddings_df = pd.read_csv(path_to_embeddings)

                embeddings_df = embeddings_df.rename(columns={'id': 'image_id'})

                if gene_type == "all_genes":
                    genes = embeddings_per_gene_per_donor(path_to_per_gene_per_donor_level_files, input_type, ts, embeddings_df)
                elif gene_type == "genes_common_in_all_donors":
                    genes = genes_common_in_all_donors



                diagnosis_prediction_res = perform_logistic_regresson_per_gene_donor_level_on_label('sz_diagnosis',input_type, ts, genes, 'lr', rnd_state, n_splits=5, n_jobs=1)
                diagnosis_prediction_res.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, input_type_path, gene_type + "_" + input_type+ "_per_gene_per_donor_diagnosis_prediction_scores_" + "lr" + "_rnd_state_" + str(rnd_state)+ ".csv"), index=False)


                # ------ donor level ---------------------

                path_to_embed_file = os.path.join(PATH_TO_SZ_STUDY, 'random' ,"random_embeddings_donor_level.csv")
                path_to_labels_file = os.path.join(PATH_TO_SZ_POST_PROCESS, "sz_diagnosis_as_label_donor_level.csv")

                donor_level_prediction_res=perform_logistic_regression(path_to_embed_file, path_to_labels_file,  "donor", n_splits=5, n_jobs=1)
                donor_level_prediction_res.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, input_type_path, gene_type + "_" + input_type + "_per_donor_diagnosis_prediction_scores_2.csv"),index=False)



            elif input_type == 'plain_resnet':

                # ------ per gene per donor ---------------
                path_to_embeddings = os.path.join(PATH_TO_SZ_STUDY, 'plain_resnet', "resnet50_embeddings_image_level.csv")
                embeddings_df = pd.read_csv(path_to_embeddings)

                if gene_type == "all_genes":
                    genes = embeddings_per_gene_per_donor(path_to_per_gene_per_donor_level_files, input_type, ts, embeddings_df)
                elif gene_type == "genes_common_in_all_donors":
                    genes = genes_common_in_all_donors

                diagnosis_prediction_res = perform_logistic_regresson_per_gene_donor_level_on_label('sz_diagnosis', input_type, ts, genes,'lr',rnd_state,  n_splits=5,
                                                                                          n_jobs=1)

                diagnosis_prediction_res.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, input_type_path, gene_type + "_" +
                                                             input_type + "_per_gene_per_donor_diagnosis_prediction_scores_" + "lr" + "_rnd_state_" + str(rnd_state)+ ".csv"),
                                                index=False)



                # ------ donor level ---------------------

                path_to_embed_file = os.path.join(PATH_TO_SZ_STUDY, 'plain_resnet', "resnet50_embeddings_donor_level.csv")
                path_to_labels_file = os.path.join(PATH_TO_SZ_POST_PROCESS, "sz_diagnosis_as_label_donor_level.csv")


                donor_level_prediction_res = perform_logistic_regression(path_to_embed_file, path_to_labels_file, "donor",n_splits=5, n_jobs=1)
                donor_level_prediction_res.to_csv(os.path.join(PATH_TO_SZ_POST_PROCESS, input_type_path, gene_type + "_"
                                                               +input_type + "_per_donor_diagnosis_prediction_scores.csv"),index=False)








