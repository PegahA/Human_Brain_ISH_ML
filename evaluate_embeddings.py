from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from human_ISH_config import *
import pandas as pd
import numpy as np
import scipy
from sklearn import metrics
import json
import os
import datetime
#print (pd.show_versions())




#EMBEDDING_DEST = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_2/"

def create_diagonal_mask(low_to_high_map, target_value=1):
    """
    Create a block diagonal mask matrix from the input mapping.

    The input pandas data frame has only two columns, the first is the
    low level id (image, sample, or probe_id) and the second is the
    high level mapping (gene, region, donor). The target_value argument can
    be set to np.nan.

    The output will be a matrix sized the number of low level ID's squared.
    The column and row order will have to be rearranged to match your distance matrix.

    """
    low_to_high_map.drop_duplicates()
    grouped = low_to_high_map.groupby(low_to_high_map.columns[1])
    ordered_low_level_names = list()
    group_matrices = []
    for name, group in grouped:
        group_size = group.shape[0]
        # build up row/col names, order doesn't matter within a group = they are all equal
        ordered_low_level_names = ordered_low_level_names + group.iloc[:, 0].tolist()
        # set the diagonal matrix to be the target value
        single_group_matrix = np.full(shape=(group_size, group_size), fill_value=target_value)
        group_matrices.append(single_group_matrix)
    # add the individual matrices along the diagonal
    relationship_matrix = scipy.linalg.block_diag(*group_matrices)
    # convert to pandas dataframe and set names
    relationship_df = pd.DataFrame(relationship_matrix, columns=ordered_low_level_names, index=ordered_low_level_names)

    return relationship_df

def get_general_distance_and_relationship_matrix(path_to_embeddings,image_level_embed_file_name):
    """
    This function uses the image_level_embeddings to create a distance matrix. It also creates a relationship matrix
    for images that we have an embedding for.
    It calculates the euclidean distance for every possible pair of images and also gives the relationship
    (same gene/different gene) for every possible pair of images.

    :param path_to_embeddings:  path to the image_level_embeddings
    :return: 2 pandas Data frames: distance matrix and the relationship matrix.
    """
    images_info = pd.read_csv(os.path.join(DATA_DIR,STUDY,"human_ISH_info.csv"))
    #images_info = pd.read_csv(os.path.join("/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_2", "human_ISH_info.csv"))

    dist_matrix_df = build_distance_matrix(os.path.join(path_to_embeddings,  image_level_embed_file_name))

    dist_matrix_rows = list(dist_matrix_df.index)  # list of image IDs
    dist_matrix_columns = list(dist_matrix_df) # list of image IDs

    # --- sanity check -------------
    if dist_matrix_rows != dist_matrix_columns:
        print ("Something is wrong, the number and order of image IDs in distance matrix's rows and columns should the same.")
        return None
    # ------------------------------


    genes = images_info[images_info['image_id'].isin(dist_matrix_rows)]['gene_symbol']

    low_to_high_map = pd.DataFrame(list(zip(dist_matrix_rows, genes))) # create a 2-column df of image IDs and genes
    relationship_df = create_diagonal_mask(low_to_high_map, target_value=1)


    # --- check to see if rows and columns of dist matrix match the relationship matrix. ---------------------
    # if they don't re-arrange them in the relationship matrix to match the dist matrix

    dist_matrix_df, relationship_df =  match_matrices(dist_matrix_df, relationship_df)

    # ---------------------------------------------------------------------------------------------------------

    return dist_matrix_df,relationship_df


def match_matrices(first_matrix_df, second_matrix_df):
    """
    Checks to see if the two matrices match.
    Matching means the number and order of rows and columns are the same.
    If they do not match, the function re-arranges the order of rows and columns of the second matrix to make it similar
    to the first matrix.

    The function does not modify the values inside the matrices. It jus re-arranges the order of columns and rows.
    :param first_matrix_df: pandas Dataframe.
    :param second_matrix_df: pandas Dataframe.
    :return: 2 pandas Dataframes.
    """

    first_matrix_array = first_matrix_df.to_numpy()
    second_matrix_array = second_matrix_df.to_numpy()

    first_matrix_rows = list(first_matrix_df.index)
    first_matrix_columns = list(first_matrix_df)

    second_matrix_rows = list(second_matrix_df.index)
    second_matrix_columns = list(second_matrix_df)

    if first_matrix_rows == second_matrix_rows and first_matrix_columns == second_matrix_columns:
        print("They match!")

    else:
        print("They don't match. Re-arranging ...")

        desired_permutation = []
        for item in second_matrix_columns:
            ind = first_matrix_columns.index(item)  # get the correct order of image IDs from distance matrix columns
            desired_permutation.append(ind)

        idx = np.empty_like(desired_permutation)
        idx[desired_permutation] = np.arange(len(desired_permutation))
        second_matrix_array[:] = second_matrix_array[:, idx]
        second_matrix_array[:] = second_matrix_array[idx, :]

        second_matrix_df = pd.DataFrame(second_matrix_array, columns=first_matrix_columns, index=first_matrix_rows)

    return first_matrix_df, second_matrix_df


def apply_mask(mask_matrix_df, original_matrix_df):
    """
   Changes elements of the original array based on the mask array.
   The original array and the mask array should have the same shape.

   :param mask_matrix_df: pandas Data frame. Boolean mask. It has to be the same shape as the target array.
   :param original_matrix_df: pandas Data frame. Original matrix that we want to apply the mask to.
   :return: pandas data frame. This is the original array after masking. It is converted to pandas df.
   """

    print("Applying the mask ...")

    original_matrix_columns = list(original_matrix_df)
    original_matrix_rows = list(original_matrix_df.index)

    mask_array = mask_matrix_df.to_numpy()
    original_array = original_matrix_df.to_numpy().astype(float)


    # Note: np.nan cannot be inserted into an array of type int. The array needs to be float.
    np.putmask(original_array, mask_array, np.nan)


    after_masking_df = pd.DataFrame(original_array, columns=original_matrix_columns, index=original_matrix_rows)
    return after_masking_df


def AUC(dist_matrix_df, label_matrix_df, label):
    """
    Calculates the AUC using the positive and negative pairs.
    It gets the actual labels of the pairs from the label matrix and the predicted labels based on the distance matrix.
    It will ignore np.nan values in the two matrices.
    :param dist_matrix_df: pandas data frame that has the euclidean distance between pairs of images. Some cells might be Nan.
    :param label_matrix_df:  pandas data frame that has the actual labels of the pairs of images. Some cells might be Nan.

    These two matrices should completely match. Meaning they should have the same number of rows and columns, the order of rows and
    columns should be the same, also any cell that is Nan in one matrix should also be Nan in the other one.
    :return: float. The AUC value.
    """




    print ("Calculating AUC ...")

    dist_matrix_columns = list(dist_matrix_df)
    dist_matrix_rows = list(dist_matrix_df.index)

    label_matrix_columns = list(label_matrix_df)
    label_matrix_rows = list(label_matrix_df.index)



    if dist_matrix_columns == label_matrix_columns and dist_matrix_rows == label_matrix_rows:
        print ("The two matrix match. Will continue to calculate AUC ...")

        dist_matrix_array = dist_matrix_df.to_numpy()
        label_matrix_array = label_matrix_df.to_numpy()


        top_tri_ind_list = np.triu_indices(len(dist_matrix_columns),1)
        top_tri_dist_matrix = dist_matrix_array[top_tri_ind_list]
        top_tri_dist_matrix = top_tri_dist_matrix[~np.isnan(top_tri_dist_matrix)]  # remove NaNs
        print ("top triangle of dist matrix without NaNs:", len(top_tri_dist_matrix))


        top_tri_ind_list = np.triu_indices(len(label_matrix_array), 1)
        top_tri_label_matrix = label_matrix_array[top_tri_ind_list]
        top_tri_label_matrix = top_tri_label_matrix [~np.isnan(top_tri_label_matrix)]  # remove NaNs
        print ("top triangle of label matrix without NaNs:",len(top_tri_label_matrix))

        top_tri_label_matrix = top_tri_label_matrix.astype(int)  # convert values to int so they are binary {0,1}

        # #positive label is 0 because distance is closer for positive pairs.
        fpr, tpr, thresholds = metrics.roc_curve(top_tri_label_matrix, top_tri_dist_matrix, pos_label=0)
        auc_val = metrics.auc(fpr, tpr)




        # --------

        #plot_roc_curve_2(fpr, tpr,label)

        # ----

        l = len(top_tri_dist_matrix)
        print (l)
        l_sub = len(top_tri_dist_matrix) // 10
        res = np.random.choice(l, l_sub)

        top_tri_dist_matrix = [top_tri_dist_matrix[i] for i in res]
        top_tri_label_matrix = [top_tri_label_matrix[i] for i in res]

        print (len(top_tri_label_matrix))

        # ----

        #plot_roc_curve_2(top_tri_label_matrix, top_tri_dist_matrix, label)

        # --------
        
        """
        dist_matrix_flatten = dist_matrix_array.flatten()
        dist_matrix_flatten = dist_matrix_flatten[~np.isnan(dist_matrix_flatten)]  # remove NaNs
        
        label_matrix_flatten = label_matrix_array.flatten()
        label_matrix_flatten = label_matrix_flatten[~np.isnan(label_matrix_flatten)]  # remove NaNs
        label_matrix_flatten = label_matrix_flatten.astype(int) # convert values to int so they are binary {0,1}

        # #positive label is 0 because distance is closer for positive pairs.
        fpr, tpr, thresholds = metrics.roc_curve(label_matrix_flatten, dist_matrix_flatten, pos_label=0)
        auc_val = metrics.auc(fpr, tpr)
        return auc_val
        """

        return auc_val

    else:
        print ("The two matrices do not match.")
        return None


def first_hit_percentage(dist_matrix_df):
    """
    This function finds the image ID of the closest image to every image and then check to see what percentage of
    these pairs are positive, meaning they have the same gene.
    It finds the closest image by calling the 'find_closest_image()' function and passing the distance matrix to it.
    The distance matrix may have np.nan values in some cells. Those cells will be ignored when looking for the closest image.

    Which cells in the distance matrix might be np.nan? That depends on the criteria of the evaluation which determines the
    universe of the genes to be considered.

    :param dist_matrix_df: data frame that has the euclidean distance between every possible pair of images in the dataset.
    The euclidean distances are calculated from the embedding vectors of each images.
    :return: float. The percentage of images for which the closest image has the same gene.
    """

    print ("Calculating first hit match percentage ...")

    images_info = pd.read_csv(os.path.join(DATA_DIR,STUDY, "human_ISH_info.csv"))
    #images_info = pd.read_csv( os.path.join("/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_2", "human_ISH_info.csv"))

    min_indexes_df = find_closest_image(dist_matrix_df) # min_indexes_df has two columns: an image ID and the ID of the closest image to that image

    total_count = len(min_indexes_df)  #total number of rows (== number of images)

    image_gene_mapping = images_info[['image_id', 'gene_symbol']]
    min_indexes_df = pd.merge(min_indexes_df, image_gene_mapping, left_on='id1', right_on='image_id')
    min_indexes_df = pd.merge(min_indexes_df, image_gene_mapping, left_on='id2', right_on='image_id')

    same_gene = min_indexes_df.query('gene_symbol_x == gene_symbol_y') # definition of positive

    match_count = len(same_gene)
    proportion = (match_count / total_count) * 100.0

    return proportion


def first_hit_match_percentage_and_AUC_results(path_to_embeddings ,image_level_embed_file_name):

    general_distance_matrix , general_relationship_matrix = get_general_distance_and_relationship_matrix(path_to_embeddings, image_level_embed_file_name)



    # ---- General ----------------------------------------------------------------------------------
    # General means only look at gene. Is it the same gene or different gene. Do not check donor_id.

    print ("---------------------------------- General ---------------------------------- ")
    general_first_hit_percentage = first_hit_percentage(general_distance_matrix)
    general_AUC = AUC(general_distance_matrix, general_relationship_matrix, "General")

    general_res = [general_first_hit_percentage, general_AUC]

    # ---- Among Other Donors ------------------------------------------------------------------------

    print ("---------------------------------- Other Donors ----------------------------- ")
    images_info = pd.read_csv(os.path.join( DATA_DIR,STUDY,"human_ISH_info.csv"))

    #images_info = pd.read_csv(os.path.join("/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_2", "human_ISH_info.csv"))

    dist_matrix_rows = list(general_distance_matrix.index)

    donors = images_info[images_info['image_id'].isin(dist_matrix_rows)]['donor_id']

    low_to_high_map = pd.DataFrame(list(zip(dist_matrix_rows, donors)))  # create a 2-column df of image IDs and genes
    mask_df = create_diagonal_mask(low_to_high_map, target_value=1) # the pairs that have the same donor will have label 1

    general_relationship_matrix, arranged_mask_df = match_matrices(general_relationship_matrix, mask_df)

    # after applying the mask, any cell that corresponds to a pair with the same donor will be Nan.
    # therefore, we are limiting our universe of pairs to those that have different donors.
    distance_matrix_after_masking = apply_mask(arranged_mask_df, general_distance_matrix)
    relationship_matrix_after_masking = apply_mask(arranged_mask_df, general_relationship_matrix)

  
    among_other_donors_first_hit_percentage = first_hit_percentage(distance_matrix_after_masking)
    among_other_donors_AUC = AUC(distance_matrix_after_masking, relationship_matrix_after_masking, "Other Donors")

    among_other_donors_res = [among_other_donors_first_hit_percentage, among_other_donors_AUC]


    # ---- Within Donor ----------------------------------------------------------------------------
    # so far, in the masked_df, the pairs that have the same donor will have label 1, and when we use this as a mask,
    # these pairs will be set to Nan. But we need the opposite of that here.
    # we need the pairs that have a different donor to be 1, so later when we actually apply the mask, the corresponding pairs would be set to Na.
    # the idea is to convert every 0 into 1 and every 1 into 0

    print("---------------------------------- Within Donor ------------------------------ ")

    inverted_mask_df =  np.logical_not(mask_df).astype(int)

    general_relationship_matrix, arranged_inverted_mask_df = match_matrices(general_relationship_matrix, inverted_mask_df)

    # after applying the mask, any cell that corresponds to a pair with the same donor will be Nan.
    # therefore, we are limiting our universe of pairs to those that have different donors.
    distance_matrix_after_masking = apply_mask(arranged_inverted_mask_df, general_distance_matrix)
    relationship_matrix_after_masking = apply_mask(arranged_inverted_mask_df, general_relationship_matrix)

    withing_donor_first_hit_percentage = first_hit_percentage(distance_matrix_after_masking)
    within_donor_brains_AUC = AUC(distance_matrix_after_masking, relationship_matrix_after_masking, "Within Donor")

    within_donor_res = [withing_donor_first_hit_percentage, within_donor_brains_AUC]


    return [general_res, among_other_donors_res, within_donor_res]



def build_distance_matrix(path_to_embeddings):
    """
    Distance from one item to itself shows up as inf.
    :param filename: String. This is the name of the folder in the EMBEDDING_DEST folder which contains the embeddings csv file
    :return: pandas DataFrame. A distance matrix that has the euclidean distance between all the possible pairs of embedding vectors
    """

    embed_df = pd.read_csv(path_to_embeddings)
    print ("length is: ", len(embed_df))
    columns = list(embed_df)

   
    distances = euclidean_distances(embed_df.iloc[:, 1:], embed_df.iloc[:, 1:])
    embed_df = embed_df.set_index([columns[0]])
    # format distance matrix
    distances_df = pd.DataFrame(distances)
    distances_df.columns = list(embed_df.index)
    distances_df.index = list(embed_df.index)

    print ("finished building the distance matrix ...")

    return distances_df


def find_closest_image(distances_df):
    """
    :param distances_df: pandas DataFrame. A distance matrix that has the euclidean distance between all the possible
    pairs of embedding vectors
    :return: pandas DataFrame. Has 2 columns. The first column is an image_id, the second column is the image_id of
    the corresponding closest image.
    """

    # find the closest image in each row

    default_value_for_diagonal = distances_df.iloc[0,0]

    # set the distance between each image to itself as inf to make sure it doesn't get picked as closest
    distances_df.values[[np.arange(distances_df.shape[0])] * 2] = float("inf")
    min_indexes = distances_df.idxmin(axis=1, skipna=True)
    min_indexes_df = pd.DataFrame(min_indexes).reset_index()
    min_indexes_df.columns = ["id1", "id2"]
    #min_indexes_df = min_indexes_df.applymap(str)

    # set the distance between each image to itself back to the default
    distances_df.values[[np.arange(distances_df.shape[0])] * 2] = float(default_value_for_diagonal)
    print("finished finding the closest image ...")
    return min_indexes_df


def not_the_same_gene(min_indexes_df, level):
    if level == 'image':

        total_count = len(min_indexes_df)
        print ("total number of images: ", total_count)
        info_csv_path = os.path.join(DATA_DIR, STUDY, "human_ISH_info.csv")
        info_csv = pd.read_csv(info_csv_path, index_col=None)

        gene_donor_mapping = info_csv[['gene_symbol', 'donor_id', 'image_id']]
        gene_donor_mapping['image_id']=gene_donor_mapping['image_id'].astype(str)
        min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='id1', right_on='image_id')
        min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='id2', right_on='image_id')

        not_the_same_image = min_indexes_df.query('image_id_x != image_id_y')
        not_the_same_gene = not_the_same_image.query('gene_symbol_x != gene_symbol_y')
        print(not_the_same_gene)

        match_count = len(not_the_same_gene)
        print("number of matches with not the same gene is: ", match_count)
        proportion = (match_count / total_count) * 100.0
        
        print ("proportion is: ", proportion)
        return proportion




def get_creation_time(ts):

    path_to_embed_file = os.path.join(DATA_DIR, STUDY, "experiment_files", "experiment_"+ ts, "triplet_training_validation_embeddings.csv")

    if os.path.exists(path_to_embed_file):
        stat = os.stat(path_to_embed_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime
    else:
        return None

def evaluate(ts):
    path_to_embeddings = os.path.join(EMBEDDING_DEST, ts)
    image_level_files_list = []

    contents = os.listdir(path_to_embeddings)
    for item in contents:
        if item.endswith("embeddings_image_level.csv"):
           image_level_files_list.append(item)

    for item in image_level_files_list:
        image_level_embed_file_name = item
        results = first_hit_match_percentage_and_AUC_results(path_to_embeddings,image_level_embed_file_name)


        # from args.json :
        args_names, args_val = get_arguments_from_json(ts)


        columns = ["ts", "number of embeddings", "duration", "general_first_hit_percentage", "general_AUC", "among_other_donors_first_hit_percentage",
                   "among_other_donors_AUC", "within_donor_first_hit_percentage", "within_donor_AUC"]


        # ------ number of embeddings and duration ----
        df = pd.read_csv(os.path.join(path_to_embeddings, image_level_embed_file_name))
        number_of_embeddings = len(df)

        creation_time  = get_creation_time(ts)
        if creation_time != None:
            creation_time = int(creation_time)
            duration = creation_time - int(ts)
        else:
            duration = 0
        # ---------------------------------------------

        if args_names != None and args_val != None:
            columns = columns[0:3] + args_names + columns[3:]

        eval_results_df = pd.DataFrame(columns=columns)

        general_res = results[0]
        among_other_donors_res = results[1]
        within_donor_res = results[2]

        if args_names != None and args_val != None:
            eval_results_df.loc[0] = [ts, number_of_embeddings, duration] + args_val + [general_res[0], general_res[1], among_other_donors_res[0],
                                      among_other_donors_res[1],
                                      within_donor_res[0], within_donor_res[1]]
        else:
            eval_results_df.loc[0] = [ts, number_of_embeddings, duration, general_res[0], general_res[1], among_other_donors_res[0],
                                                        among_other_donors_res[1],
                                                        within_donor_res[0], within_donor_res[1]]

        eval_result_file_name = item.split(".")[0] + "_evaluation_result_top_tri.csv"
        eval_path = os.path.join(EMBEDDING_DEST, ts)
        eval_results_df.to_csv(os.path.join(eval_path, eval_result_file_name), index=None)

        #----------




def get_arguments_from_json(ts):
    list_of_arguments_to_get = ["segmentation_training_samples","patch_count_per_image", "learning_rate", "batch_k",
                                "batch_p", "flip_augment", "standardize"]

    args_val_list = []

    path_to_embeddings = os.path.join(EMBEDDING_DEST, ts)
    args_file = os.path.join(path_to_embeddings, "args.json")
    if not os.path.exists(args_file):
        print ("There is no args.json file in ", path_to_embeddings)
        return None, None

    if os.path.isfile(args_file):
        with open(args_file, 'r+') as f:
            args_resumed = json.load(f)
            for arg in list_of_arguments_to_get:
                if arg in args_resumed:
                    args_val_list.append(args_resumed[arg])
                else:
                    args_val_list.append(0)

            f.close()

    return list_of_arguments_to_get, args_val_list


def concat_all_evaluation_results():

    """

    :return:
    """
    """
    list_of_folders= ["1584753511", "1583770480", "1585521837", "1584025762", "1586831151", "1586740776", "1587686591",
                      "1587462051", "1589259198", "1589258734", "1589222258", "1591130418", "1591130635", "1591132845",
                      "1591188766", "1591234815", "1591250445", "1591297149", "1591329662", "1591342075", "1591395395",
                      "1591423439", "1591434031", "1591490025", "1591509560", "1591521386", "1591588276", "1591600820",
                      "1591615341", "1591684726", "1591695239", "1591712071", "1591783517", "1591793952", "1591813151",
                      "1591881335", "1591897361", "1591914659", "1591986392", "1591997885", "1592014294", "1592079079",
                      "1592090557", "1592105924", "1592178919"]
    """

    list_of_folders = ["1593023060", "1593023112", "1593023149", "1593132703", "1593133440", "1593134313", "1593242622",
               "1593244389", "1593245325", "1593349242", "1593353302", "1593355864", "random"]

    train_eval_df_list = []
    val_eval_df_list = []
    train_val_eval_df_list = []


    for item in list_of_folders:
        path_to_eval_folder = os.path.join(EMBEDDING_DEST, item)
        files = os.listdir(path_to_eval_folder)

        for f in files:
            if f.endswith("image_level_evaluation_result_top_tri.csv"):

                if "triplet" in f:
                    df = pd.read_csv(os.path.join(path_to_eval_folder, f))
                    train_val_eval_df_list.append(df)

                elif "training" in f:
                    df = pd.read_csv(os.path.join(path_to_eval_folder, f))
                    train_eval_df_list.append(df)

                elif "validation" in f:
                    df = pd.read_csv(os.path.join(path_to_eval_folder, f))
                    val_eval_df_list.append(df)


    columns = list(train_val_eval_df_list[0])
    train_columns = ["training_"+item for item in columns[1:]]
    train_columns = [columns[0]] + train_columns
    train_columns_dict ={}
    
    val_columns = ["validation_"+item for item in columns[1:]]
    val_columns = [columns[0]] + val_columns
    val_columns_dict ={}

    #train_and_val_columns = ["train_and_validation_"+item for item in columns[1:]]
    #train_and_val_columns = [columns[0]] + train_and_val_columns
    #train_and_val_columns_dict ={}


    for i in range(len(columns)):
        train_columns_dict[columns[i]] = train_columns[i]
        val_columns_dict[columns[i]] = val_columns[i]
        #train_and_val_columns_dict[columns[i]] = train_and_val_columns[i]


    concatenated_training_df = pd.concat(train_eval_df_list)
    concatenated_training_df = concatenated_training_df.rename(columns=train_columns_dict)

    concatenated_validation_df = pd.concat(val_eval_df_list)
    concatenated_validation_df = concatenated_validation_df.rename(columns=val_columns_dict)
    
    concatenated_train_and_validation_df = pd.concat(train_val_eval_df_list)
    #concatenated_train_and_validation_df =  concatenated_train_and_validation_df.rename(columns=train_and_val_columns_dict)


    concatenated_training_df.to_csv(os.path.join(EMBEDDING_DEST,"grids","training_all_evaluation_result_top_tri.csv"),index=None)
    concatenated_validation_df.to_csv(os.path.join(EMBEDDING_DEST,"grids","validation_all_evaluation_result_top_tri.csv"),index=None)
    concatenated_train_and_validation_df.to_csv(os.path.join(EMBEDDING_DEST,"grids","training_and_validation_all_evaluation_result_top_tri.csv"), index=None)

    # ---------
    # If you have columns on arguments, keep them in training but drop them in validation and train_and_val to prevent duplicates
    list_of_cols_in_validation_df = list(concatenated_validation_df)
    list_of_cols_in_train_val_df = list(concatenated_train_and_validation_df)
    args_cols = ["segmentation_training_samples","patch_count_per_image", "learning_rate", "batch_k",
                                "batch_p", "flip_augment", "standardize"]
    args_cols_val = ["validation_"+item for item in args_cols]
    
    if len(list_of_cols_in_train_val_df) == len(list_of_cols_in_validation_df) and len(list_of_cols_in_train_val_df) > 7:
        concatenated_validation_df = concatenated_validation_df.drop(args_cols_val, axis=1)
        concatenated_train_and_validation_df = concatenated_train_and_validation_df.drop(args_cols, axis=1)


    # ---------

    all_three_df_list = [concatenated_training_df, concatenated_validation_df, concatenated_train_and_validation_df]
    concatenated_all_df = pd.concat(all_three_df_list, axis=1)
    concatenated_all_df.to_csv(os.path.join(EMBEDDING_DEST,"grids","all_evaluation_result_top_tri.csv"), index=None)



def plot_roc_curve_2(x, y, label):

    print ("here in plotting ...")
    import matplotlib.pyplot as plt

    plt.scatter(x, y)
    plt.title(label+" - Training")
    plt.xlabel("label")
    plt.ylabel("distance")
    plt.show()

def plot_roc_curve(label_matrix, dist_matrix):

    print ("Here in plotting ...")
    import scikitplot as skplt
    import matplotlib.pyplot as plt

    y_true =  label_matrix
    y_pred=  dist_matrix

    print (type(y_true))
    print (type(y_pred))

    print (y_true.shape)
    print (y_pred.shape)
    skplt.metrics.plot_roc_curve(y_true, y_pred)
    plt.show()





def main():
    #ts_list = ["1584753511", "1583770480", "1585521837", "1584025762", "1586831151", "1586740776",
              #"1587686591", "1587462051", "1589259198", "1589258734" , 
    #ts_list = ["1589222258","1591130418", "1591130635", "1591132845", "1591188766", "1591234815", "1591250445","1591297149", "1591329662", "1591342075", "1591395395", "1591423439", "1591434031","1591490025", "1591509560", "1591521386", "1591588276","1591600820", "1591615341", "1591684726", "1591695239"]

    #ts_list = ["1591712071", "1591783517", "1591793952", "1591813151", "1591881335", "1591897361", "1591914659", "1591986392" ,
               #"1591997885", "1592014294", "1592079079", "1592090557", "1592105924", "1592178919" ]

    #ts_list = ["1593023060", "1593023112", "1593023149", "1593132703", "1593133440", "1593134313", "1593242622",
               #"1593244389", "1593245325", "1593349242", "1593353302", "1593355864"]

    ts_list = ["random"]
    for ts in ts_list:
        print ("ts is: ", ts)
        evaluate(ts)





if __name__ == '__main__':

    #main()
    #concat_all_evaluatio
    # n_results()

    #main()
    concat_all_evaluation_results()


    





