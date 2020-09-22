from sklearn.metrics.pairwise import euclidean_distances
from human_ISH_config import *
import pandas as pd
import numpy as np
import scipy
from sklearn import metrics
import json
import os
#print (pd.show_versions())



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

def get_general_distance_and_relationship_matrix(path_to_embeddings,image_level_embed_file_name, study=None):
    """
    This function uses the image_level_embeddings to create a distance matrix. It also creates a relationship matrix
    for images that we have an embedding for.
    It calculates the euclidean distance for every possible pair of images and also gives the relationship
    (same gene/different gene) for every possible pair of images.

    :param path_to_embeddings:  path to the image_level_embeddings
    :return: 2 pandas Data frames: distance matrix and the relationship matrix.
    """

    if study == None:
        images_info = pd.read_csv(os.path.join(DATA_DIR,STUDY,"human_ISH_info.csv"))
    else:
        images_info = pd.read_csv(os.path.join(DATA_DIR, study, "human_ISH_info.csv"))


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
    Matching means the number and order of rows and columns are the same (based on titles)
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


def AUC(dist_matrix_df, label_matrix_df, title, image_level_embed_file_name):
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


        """
        
        if 'training_validation' in image_level_embed_file_name:
            set_type = 'Training and Validation'
        elif 'training' in image_level_embed_file_name:
            set_type = 'Training'
        elif 'validation' in image_level_embed_file_name:
            set_type = 'Validation'
        else:
            set_type = None
            
        #plot_curve(fpr, tpr, title, ['fpr', 'tpr'], set_type)  #to generate tpr over fpr graphs

        l = len(top_tri_dist_matrix)
        l_sub = len(top_tri_dist_matrix) // 10
        res = np.random.choice(l, l_sub)

        top_tri_dist_matrix = [top_tri_dist_matrix[i] for i in res]
        top_tri_label_matrix = [top_tri_label_matrix[i] for i in res]
        #plot_curve(top_tri_label_matrix, top_tri_dist_matrix, title, ['label', 'distance'], set_type) # to generate distance over actual label graphs
        """



        """
        # This piece of code will use the whole dist and label matrix and not just the top triangle.
        # This is less efficient because we know that the matrices are symmetric. 
        
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


def first_hit_percentage(dist_matrix_df, study=None):
    """
    This function finds the image ID of the closest image to every image and then checks to see what percentage of
    these pairs are positive, meaning they have the same gene.
    It finds the closest image by calling the 'find_closest_image()' function and passing the distance matrix to it.
    The distance matrix may have np.nan values in some cells. Those cells will be ignored when looking for the closest image.

    Which cells in the distance matrix might be np.nan? That depends on the criteria of the evaluation which determines the
    universe of the genes to be considered.

    :param dist_matrix_df: data frame that has the euclidean distance between every possible pair of images in the dataset.
    The euclidean distances are calculated from the embedding vectors of each images.
    :param study: the study that the embeddings belong to (i.e autism, schizophrenia)

    # -----------------------
    One step of the project is to pass disease images through a model that has been trained on healthy cortex images.
    Because these models are trained on cortex, their corresponding files and information are in the cortex folder.
    The disease embeddings that are generated by these models will also be stored in the cortex folder.
    Therefore, the STUDY argument in the human_ISH_config.py is set to "cortex".

    However, the main files of the disease dataset are in its own directory. That is why we have this argument 'study'.

    Example: I have a model that has been trained on cortex images at some time stamp.
    The info of that model such as its check points and its generated cortex embeddings are in:
    DATA_DIR/cortex/segmentation_embeddings/timestamp
    Also, when I pass the SZ and autism images through this model, I will store their embeddings in the same directory:
    DATA_DIR/cortex/segmentation_embeddings/timestamp

    However, he info.csv file which has the information of schizophrenia images and needs to be used in evaluation is in:
    DATA_DIR/schizophrenia


    To summarize: in this scenario, the disease embeddings that we want to evaluate are in:
    DATA_DIR//STUDY/segmentation_embeddings/timestamp
    but the disease info file is in:
    DATA_DIR/study/
    # -----------------------


    :return: float. The percentage of images for which the closest image has the same gene.
    """

    print ("Calculating first hit match percentage ...")

    if study == None:
        images_info = pd.read_csv(os.path.join(DATA_DIR,STUDY, "human_ISH_info.csv"))
    else:
        images_info = pd.read_csv(os.path.join(DATA_DIR, study, "human_ISH_info.csv"))

    min_indexes_df = find_closest_image(dist_matrix_df) # min_indexes_df has two columns: an image ID and the ID of the closest image to that image

    total_count = len(min_indexes_df)  #total number of rows (== number of images)

    image_gene_mapping = images_info[['image_id', 'gene_symbol']]
    min_indexes_df = pd.merge(min_indexes_df, image_gene_mapping, left_on='id1', right_on='image_id')
    min_indexes_df = pd.merge(min_indexes_df, image_gene_mapping, left_on='id2', right_on='image_id')

    same_gene = min_indexes_df.query('gene_symbol_x == gene_symbol_y') # definition of positive

    match_count = len(same_gene)
    proportion = (match_count / total_count) * 100.0

    return proportion


def first_hit_match_percentage_and_AUC_results(path_to_embeddings ,image_level_embed_file_name, study = None):
    """
    :param path_to_embeddings:
    :param image_level_embed_file_name:
    :param study: the study that the embeddings belong to (i.e autism, schizophrenia)

    # -----------------------
    One step of the project is to pass disease images through a model that has been trained on healthy cortex images.
    Because these models are trained on cortex, their corresponding files and information are in the cortex folder.
    The disease embeddings that are generated by these models will also be stored in the cortex folder.
    Therefore, the STUDY argument in the human_ISH_config.py is set to "cortex".

    However, the main files of the disease dataset are in its own directory. That is why we have this argument 'study'.

    Example: I have a model that has been trained on cortex images at some time stamp.
    The info of that model such as its check points and its generated cortex embeddings are in:
    DATA_DIR/cortex/segmentation_embeddings/timestamp
    Also, when I pass the SZ and autism images through this model, I will store their embeddings in the same directory:
    DATA_DIR/cortex/segmentation_embeddings/timestamp

    However, he info.csv file which has the information of schizophrenia images and needs to be used in evaluation is in:
    DATA_DIR/schizophrenia


    To summarize: in this scenario, the disease embeddings that we want to evaluate are in:
    DATA_DIR//STUDY/segmentation_embeddings/timestamp
    but the disease info file is in:
    DATA_DIR/study/
    # -----------------------
    :return:
    """

    general_distance_matrix , general_relationship_matrix = get_general_distance_and_relationship_matrix(path_to_embeddings, image_level_embed_file_name, study)



    # ---- General ----------------------------------------------------------------------------------
    # General means only look at gene. Is it the same gene or different gene. Do not check donor_id.

    print ("---------------------------------- General ---------------------------------- ")
    general_first_hit_percentage = first_hit_percentage(general_distance_matrix, study)
    general_AUC = AUC(general_distance_matrix, general_relationship_matrix, "General", image_level_embed_file_name)

    general_res = [general_first_hit_percentage, general_AUC]

    # ---- Among Other Donors ------------------------------------------------------------------------

    print ("---------------------------------- Other Donors ----------------------------- ")
    if study == None:
        images_info = pd.read_csv(os.path.join( DATA_DIR,STUDY,"human_ISH_info.csv"))
    else:
        images_info = pd.read_csv(os.path.join(DATA_DIR, study, "human_ISH_info.csv"))


    dist_matrix_rows = list(general_distance_matrix.index)

    donors = images_info[images_info['image_id'].isin(dist_matrix_rows)]['donor_id']


    low_to_high_map = pd.DataFrame(list(zip(dist_matrix_rows, donors)))  # create a 2-column df of image IDs and genes
    mask_df = create_diagonal_mask(low_to_high_map, target_value=1) # the pairs that have the same donor will have label 1

    general_relationship_matrix, arranged_mask_df = match_matrices(general_relationship_matrix, mask_df)

    # after applying the mask, any cell that corresponds to a pair with the same donor will be Nan.
    # therefore, we are limiting our universe of pairs to those that have different donors.
    distance_matrix_after_masking = apply_mask(arranged_mask_df, general_distance_matrix)
    relationship_matrix_after_masking = apply_mask(arranged_mask_df, general_relationship_matrix)

    among_other_donors_first_hit_percentage = first_hit_percentage(distance_matrix_after_masking, study)
    among_other_donors_AUC = AUC(distance_matrix_after_masking, relationship_matrix_after_masking, "Other Donors", image_level_embed_file_name)

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

    withing_donor_first_hit_percentage = first_hit_percentage(distance_matrix_after_masking, study)
    within_donor_brains_AUC = AUC(distance_matrix_after_masking, relationship_matrix_after_masking, "Within Donor", image_level_embed_file_name)

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

    print ("///////////////////")
    print (len(distances_df))

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
    """
    This function returns the proportion of images for which the closest image (based on the distance matrix) has a different gene.

    This is a helper function to better understand the metrics and results.

    Ideally, the closest image to an image should have the same gene. (same gene ==> same pattern ==> less distance)
    :param min_indexes_df: a dataframe with two columns: image id, and image id of the closest image
    :param level: the integration level in at which we are comparing the embeddings.
    :return: float.
    """
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
    """
    This function gets the creation time of the embedding csv file.
    It is designed to be used for embeddings that are generated by the triplet model.

    I am using the embedding file in the experiment_files folder. The reason is that in linux, there is no simple way
    of getting the creation time of a file. Instead, we can get the last time it was modifies.
    Every embedding file generated by the triplet model is saved in EXPERIMENT_ROOT. From there, it is also copied inside
    EMBEDDING_DEST.
    After copying, I use the copied version in EMBEDDING_DEST so the initial one in EXPERIMENT_ROOT is probably never
    accessed and modified and its last-access-time will be almost the same as its creation time.

    :param ts: folder name.
    :return:  creation time stamp.
    """
    path_to_embed_file = os.path.join(DATA_DIR, STUDY, "experiment_files", "experiment_"+ ts, "triplet_training_validation_embeddings.h5")

    if os.path.exists(path_to_embed_file):
        stat = os.stat(path_to_embed_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime
    else:
        print ("here, path is: ", path_to_embed_file)
        return None



def disease_embed_evaluate(study):

    #ts_list =  ['1596374295', '1595171169', '1596183933', '1595636690', '1596630544', '1596890418', '1596929673',
              #  '1595570961', '1596258245', '1593570490', '1596444832', '1596335814', '1595941978', '1596795103',
             #   '1595326272', '1596946785', '1596553484', '1595472034', '1593133440', '1595107729']

    #ts_list =  ['1596374295', '1595171169', '1596183933', '1595636690', '1596630544']

    ts_list = ["random", "resnet50_50_patches"]

    for ts in ts_list:

        if ts == "random" or "resnet" in ts:
            path_to_embeddings = os.path.join(DATA_DIR, study, "segmentation_embeddings", ts)
            eval_path = os.path.join(DATA_DIR, study, "segmentation_embeddings", ts)
            base_case = True

        else:
            path_to_embeddings = os.path.join(EMBEDDING_DEST, ts)
            eval_path = os.path.join(EMBEDDING_DEST, ts)
            base_case = False

        image_level_files_list = []

        print ("ts is: ", ts)
        print ("path to embeddings is: ", path_to_embeddings)

        contents = os.listdir(path_to_embeddings)

        print (contents)

        for item in contents:

            if base_case:
                if item.endswith("embeddings_image_level.csv"):
                    image_level_files_list.append(item)
            else:

                if item.endswith("embeddings_image_level.csv") and study in item:
                    image_level_files_list.append(item)

        print (image_level_files_list)

        for item in image_level_files_list:
            # for every image level embedding file, call another function to calculate first hit match percentage and AUC
            image_level_embed_file_name = item
            results = first_hit_match_percentage_and_AUC_results(path_to_embeddings, image_level_embed_file_name, study)

            # list of columns to have in the evaluation table.
            columns = ["ts", "dataset","number of embeddings", "general_first_hit_percentage", "general_AUC",
                       "among_other_donors_first_hit_percentage","among_other_donors_AUC",
                       "within_donor_first_hit_percentage", "within_donor_AUC"]

            df = pd.read_csv(os.path.join(path_to_embeddings, image_level_embed_file_name))
            number_of_embeddings = len(df)

            eval_results_df = pd.DataFrame(columns=columns)

            general_res = results[0]
            among_other_donors_res = results[1]
            within_donor_res = results[2]


            eval_results_df.loc[0] = [ts, study, number_of_embeddings, general_res[0], general_res[1],
                                          among_other_donors_res[0],
                                          among_other_donors_res[1],
                                          within_donor_res[0], within_donor_res[1]]

            eval_result_file_name = item.split(".")[0] + "_evaluation_result_top_tri.csv"
            eval_results_df.to_csv(os.path.join(eval_path, eval_result_file_name), index=None)




def evaluate(ts, not_found_list):
    """
    The function evaluates the embeddings. It gets a folder name as input. If embeddings are generated by the triplet model,
    the input folder name is a time stamp.

    The function then reads all the image level embedding csv files within that folder. Normally, there should be 3 files:
    One for training set embeddings, one for validation set embeddings, and one for training+validation.

    :param ts: Folder name
    :return: None. For every image level embedding file inside this folder, the function creates an evaluation csv file.
    """
    path_to_embeddings = os.path.join(EMBEDDING_DEST, ts)
    image_level_files_list = []

    if not os.path.exists(path_to_embeddings):
        print ("Could not find ", path_to_embeddings)
        not_found_list.append(path_to_embeddings)
        pass
    else:

        contents = os.listdir(path_to_embeddings)
        for item in contents:
            if item.endswith("embeddings_image_level.csv"):
               image_level_files_list.append(item)

        for item in image_level_files_list:

            # for every image level embedding file, call another function to calculate first hit match percentage and AUC
            image_level_embed_file_name = item
            results = first_hit_match_percentage_and_AUC_results(path_to_embeddings,image_level_embed_file_name)

            # from args.json :
            args_names, args_val = get_arguments_from_json(ts)


            # list of columns to have in the evaluation table.
            columns = ["ts", "number of embeddings", "duration", "general_first_hit_percentage", "general_AUC", "among_other_donors_first_hit_percentage",
                       "among_other_donors_AUC", "within_donor_first_hit_percentage", "within_donor_AUC"]


            # ------ number of embeddings and duration ----
            df = pd.read_csv(os.path.join(path_to_embeddings, image_level_embed_file_name))
            number_of_embeddings = len(df)

            # duration means the amount of time between when the folder was created and when the embeddings were generated.
            # it is the amount of time that it took the model to generate these embeddings.
            # this argument is valid for embeddings that were generated by the triplet model.

            print ("/////////////////////////////////")
            print (args_names)
            if args_names != None and "finish_time" in args_names:
                idx = args_names.index("finish_time")
                creation_time = int(args_val[idx])
                duration = creation_time - int(ts)


            else:
                creation_time  = get_creation_time(ts)
                if creation_time != None:
                    creation_time = int(creation_time)
                    duration = creation_time - int(ts)
                else:
                    duration = -1
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

    return not_found_list


def get_json_argument_list():
    """
    Returns a list of arguments from json files that we are interested in and we want to keep as columns in the evaluation tables.
    :return: list of arguments
    """
    list_of_arguments_to_get = ["finish_time", "segmentation_training_samples", "patch_count_per_image", "learning_rate", "batch_k",
                                "batch_p", "flip_augment", "standardize", "margin", "metric"]

    return list_of_arguments_to_get



def get_arguments_from_json(ts):
    """
    Embeddings that are generated with the triplet model are saved in folders that have time stamp as name.
    There is an args.json file in each folder that has the values for the arguments.

    The function checks to see if there is an args.json file within that folder. If yes, it looks for arguments
    from a list of arguments and returns the argument value. If that argument does not exist in the json file, it is returned as -1.

    :param ts: The embedding folder's name which is usually a time stamp.
    :return: two lists. The first list is a list of arguments, the second list is those arguments' values.
    """
    list_of_arguments_to_get = get_json_argument_list()

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
                    args_val_list.append(-1)

            f.close()

    return list_of_arguments_to_get, args_val_list


def get_all_ts_folders():
    path_to_ts_embed_folders = os.path.join(EMBEDDING_DEST)
    folders = os.listdir(path_to_ts_embed_folders)
    ts_folders = []

    for f in folders:
        if "159" in f:
            ts_folders.append(f)

    return ts_folders


def concat_disease_evaluation_results(study):
    list_of_folders= ['1596374295', '1595171169', '1596183933', '1595636690', '1596630544', '1596890418', '1596929673',
                      '1595570961', '1596258245', '1593570490', '1596444832', '1596335814', '1595941978', '1596795103',
                      '1595326272', '1596946785', '1596553484', '1595472034', '1593133440', '1595107729', "random", "resnet50_50_patches"]

    eval_df_list = []

    for item in list_of_folders:
        if item == "random" or "resnet" in item:
            path_to_eval_folder = os.path.join(DATA_DIR, study, "segmentation_embeddings", item)
            base_case = True
        else:
            path_to_eval_folder = os.path.join(EMBEDDING_DEST, item)
            base_case = False

        files = os.listdir(path_to_eval_folder)

        for f in files:

            # for each evaluation result csv file, see whether it is from training set, or validation set, or training+validation

            if base_case == True:
                if f.endswith("image_level_evaluation_result_top_tri.csv"):
                    df = pd.read_csv(os.path.join(path_to_eval_folder, f))
                    eval_df_list.append(df)
            else:
                if f.endswith("image_level_evaluation_result_top_tri.csv") and study in f:
                    df = pd.read_csv(os.path.join(path_to_eval_folder, f))
                    eval_df_list.append(df)


    columns = list(eval_df_list[0])

    concatenated_df = pd.concat(eval_df_list, sort=False)
    
    concatenated_df.to_csv(os.path.join(EMBEDDING_DEST, study+ "_all_evaluation_result_top_tri.csv"),index=None)


def concat_all_evaluation_results():
    """
    The function uses a list of folders, goes through each folder and reads its evaluation csv files.
    Normally, there should be 3 files in each folder: one for training set evaluation, one for validation set evaluation,
    and one for training+validation set evaluation.

    The function first concatenates each set's results from all the folders (concatenates vertically) into csv files.
    So there will be 3 csv files, one for training, one for validation, and one for training+validation.
    Number of rows == number of folder.

    The function then concatenates those 3 csv files horizontally into a final general csv file.


    :return: None. The function generates 4 csv files.

    training_all_evaluation_result_top_tri.csv
    validation_all_evaluation_result_top_tri.csv
    training_and_validation_all_evaluation_result_top_tri.csv
    all_evaluation_result_top_tri.csv

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

    """
    list_of_folders = ["1593023060", "1593023112", "1593023149", "1593132703", "1593133440", "1593134313", "1593242622",
                       "1593244389", "1593245325", "1593349242", "1593353302", "1593355864", "1593458519", "1593462661",
                       "1593470584", "1593570490", "1593581711", "1593585268", "1593683948", "1593695731", "1593696278",
                       "1593798768", "1593804603", "1593813177", "1593929477", "1593929501", "1594019525", "1594033616",
                       "1594113452", "1594118066", "1594132422", "1594165757", "1594192645", "1594199191", "1594232848",
                       "1594694428", "1594694844", "1594695178", "random"]
    """

    #list_of_folders = ["1594920479", "1594920854", "1594921222", "1594957148", "1594957337", "1594957873", "1594990440",
     #                  "1594991833", "1594992442", "1595027778", "1595029308", "1595029898", "1595035644", "1595061900",
     #                  "1595063681", "1595064319", "1595071590", "1595099038", "1595101976", "1595102546", "1595107729",
     #                  "1595132851", "1595136249", "1595136799", "1595143205", "1595171169", "1595175053", "1595175523",
     #                  "1595287279", "1595287977", "1595288363", "1595326272", "1595326978", "1595327354", "1595360634",
     #                  "1595361328", "1595361718", "1595398605", "1595399328", "1595399723", "1595431794", "1595432150",
     #                  "1595434064", "1595469825", "1595470197", "1595472034", "1595503244", "1595503323", "1595536453",
     #                  "1595536980", "1595570417", "1595570961", "1595602850", "1595603756", "1595635727", "1595636690",
     #                  "1595668008", "1595669221", "1595669221", "1595883396", "1595904365", "1595904737", "1595919239",
     #                  "1595941978", "1595942353", "1595954945", "1595989172", "1596024687", "1596058492", "random"]

    #list_of_folders = ["1596182551", "1596182973", "1596183379", "1596183933", "1596184224", "1596187834", "1596221527",
    # "1596221771", "1596223288", "1596225537", "1596256485", "1596256723", "1596258245", "1596260525",
    # "1596300288", "1596300554", "1596302071", "1596304056", "1596335566", "1596335814", "1596337331",
    # "1596339342", "1596374295", "1596375453", "1596375695", "1596379176", "1596409725", "1596410763",
    # "1596410988", "1596414379", "1596444832", "1596450560", "1596450802", "1596454143", "1596479945",
    # "1596485467", "1596485699", "1596489082", "1596516521", "1596525946", "1596526192", "1596529501",
    # "1596553484", "1596561093", "1596561322", "1596564704", "1596595541", "1596604431", "1596604622",
    # "1596607509", "1596630544", "1596639464", "1596639649", "1596642538", "1596672248", "1596683659",
    # "1596683840", "1596686401", "1596709811", "1596718871", "1596719042", "1596721616", "1596746123",
    # "1596759993", "1596760102", "1596762700", "1596784123", "1596795103", "1596795150", "1596797763",
    # "1596819094", "1596835082", "1596835093", "1596837656", "1596854477", "1596869949", "1596869960",
    # "1596872548", "1596890418", "1596911384", "1596911583", "1596914043", "1596929673", "1596946541",
    # "1596946785", "1596949246", "1596987783", "1596988070", "1596989989", "1597020996", "1597021394",
    # "1597023204", "1597059046", "1597059501", "1597061142", "random"]



    #list_of_folders = ["1598148899", "1598149283", "1598149994", "1598163229", "1598169861", "1598176752", "1598176969",
     #          "1598189473", "1598190556", "1598202465", "1598208605", "1598225452", "random"]


    list_of_folders = get_all_ts_folders()

    train_eval_df_list = []
    val_eval_df_list = []
    train_val_eval_df_list = []


    for item in list_of_folders:
        path_to_eval_folder = os.path.join(EMBEDDING_DEST, item)
        files = os.listdir(path_to_eval_folder)

        for f in files:

            # for each evaluation result csv file, see whether it is from training set, or validation set, or training+validation
            if f.endswith("image_level_evaluation_result_top_tri.csv"):

                if "random" in f:
                    if "random_training_validation" in f:
                        df = pd.read_csv(os.path.join(path_to_eval_folder, f))
                        train_val_eval_df_list.append(df)

                    elif "random_training" in f:
                        df = pd.read_csv(os.path.join(path_to_eval_folder, f))
                        train_eval_df_list.append(df)


                    elif "random_validation" in f:
                        df = pd.read_csv(os.path.join(path_to_eval_folder, f))
                        val_eval_df_list.append(df)


                else:
                    if "triplet" in f:
                        df = pd.read_csv(os.path.join(path_to_eval_folder, f))
                        train_val_eval_df_list.append(df)

                    elif "training" in f:
                        df = pd.read_csv(os.path.join(path_to_eval_folder, f))
                        train_eval_df_list.append(df)

                    elif "validation" in f:
                        df = pd.read_csv(os.path.join(path_to_eval_folder, f))
                        val_eval_df_list.append(df)


    # add 'training_' or 'validation_' to the column names of evaluation results coming from training and validation sets.
    # This is to be able to distinguish them in the final general csv file.

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


    concatenated_training_df = pd.concat(train_eval_df_list, sort=False)
    concatenated_training_df = concatenated_training_df.rename(columns=train_columns_dict)

    concatenated_validation_df = pd.concat(val_eval_df_list, sort=False)
    concatenated_validation_df = concatenated_validation_df.rename(columns=val_columns_dict)
    
    concatenated_train_and_validation_df = pd.concat(train_val_eval_df_list, sort=False)
    #concatenated_train_and_validation_df =  concatenated_train_and_validation_df.rename(columns=train_and_val_columns_dict)


    concatenated_training_df.to_csv(os.path.join(EMBEDDING_DEST,"training_all_evaluation_result_top_tri.csv"),index=None)
    concatenated_validation_df.to_csv(os.path.join(EMBEDDING_DEST,"validation_all_evaluation_result_top_tri.csv"),index=None)
    concatenated_train_and_validation_df.to_csv(os.path.join(EMBEDDING_DEST,"training_and_validation_all_evaluation_result_top_tri.csv"), index=None)

    # ---------
    # If you have columns on arguments, keep them in training but drop them in validation and train_and_val to prevent duplicates
    list_of_cols_in_validation_df = list(concatenated_validation_df)
    list_of_cols_in_train_val_df = list(concatenated_train_and_validation_df)
    args_cols = get_json_argument_list()

    args_cols_val = ["validation_"+item for item in args_cols]
    
    if len(list_of_cols_in_train_val_df) == len(list_of_cols_in_validation_df) and len(list_of_cols_in_train_val_df) > 7:
        concatenated_validation_df = concatenated_validation_df.drop(args_cols_val, axis=1, errors='ignore')
        concatenated_train_and_validation_df = concatenated_train_and_validation_df.drop(args_cols, axis=1, errors='ignore')


    # ---------

    all_three_df_list = [concatenated_training_df, concatenated_validation_df, concatenated_train_and_validation_df]
    concatenated_all_df = pd.concat(all_three_df_list, axis=1)
    concatenated_all_df.to_csv(os.path.join(EMBEDDING_DEST,"all_evaluation_result_top_tri.csv"), index=None)



def plot_curve(x, y, title, labels, set_type):
    """
    :param x: x axis values
    :param y: y axis values
    :param title: graph title
    :param labels: axis labels
    :param set_type: the set
    :return: None
    """

    print ("Here in plotting ...")
    import matplotlib.pyplot as plt

    plt.scatter(x, y)
    plt.title(title+" - " + set_type)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()


def main():
    #ts_list = ["1584753511", "1583770480", "1585521837", "1584025762", "1586831151", "1586740776",
              #"1587686591", "1587462051", "1589259198", "1589258734" , 
    #ts_list = ["1589222258","1591130418", "1591130635", "1591132845", "1591188766", "1591234815", "1591250445","1591297149", "1591329662", "1591342075", "1591395395", "1591423439", "1591434031","1591490025", "1591509560", "1591521386", "1591588276","1591600820", "1591615341", "1591684726", "1591695239"]

    #ts_list = ["1591712071", "1591783517", "1591793952", "1591813151", "1591881335", "1591897361", "1591914659", "1591986392" ,
               #"1591997885", "1592014294", "1592079079", "1592090557", "1592105924", "1592178919" ]

    #ts_list = ["1593023060", "1593023112", "1593023149", "1593132703", "1593133440", "1593134313", "1593242622",
               #"1593244389", "1593245325", "1593349242", "1593353302", "1593355864"]

    #ts_list = ["random"]
    #ts_list = ["1593458519", "1593462661", "1593470584", "1593570490", "1593581711", "1593585268", "1593683948",
               #"1593695731", "1593696278", "1593798768", "1593804603", "1593813177", "1593929477", "1593929501",
               #"1594019525", "1594033616", "1594113452", "1594118066", "1594132422", "1594165757", "1594192645",
               #"1594199191", "1594232848"]

    #ts_list = ["1594694428", "1594694844", "1594695178"]
    #ts_list = ["1594920479", "1594920854", "1594921222", "1594957148", "1594957337", "1594957873", "1594990440",
                   #   "1594991833", "1594992442", "1595027778", "1595029308", "1595029898", "1595035644", "1595061900",
                    #  "1595063681", "1595064319", "1595071590", "1595099038", "1595101976", "1595102546", "1595107729",
                    # "1595132851", "1595136249", "1595136799", "1595143205", "1595171169", "1595175053", "1595175523"]

   
    #ts_list= ["1595287279", "1595287977", "1595288363", "1595326272", "1595326978", "1595327354",
     #         "1595360634", "1595362328", "1595361718", "1595398605", "1595399328", "1595399723",
     #         "1595431794", "1595432150", "1595434064", "1595469825", "1595470197", "1595472034",
     #         "1595503244", "1595593323"]


    #ts_list = ["1595503323", "1595536453", "1595536980", "1595570417", "1595570961", "1595602850", "1595603756",
             #  "1595635727", "1595636690","1595668008", "1595669221"]


    #ts_list = ["1595669221", "1595883396", "1595904365", "1595904737", "1595919239", "1595941978",
               #"1595942353", "1595954945", "1595989172", "1596024687", "1596058492"]


    #ts_list = ["1596182551", "1596182973", "1596183379", "1596183933", "1596184224", "1596187834", "1596221527",
     #          "1596221771", "1596223288", "1596225537", "1596256485", "1596256723", "1596258245", "1596260525",
     #          "1596300288", "1596300554", "1596302071", "1596304056", "1596335566", "1596335814", "1596337331",
     #          "1596339342", "1596374295", "1596375453", "1596375695", "1596379176", "1596409725", "1596410763",
     #          "1596410988", "1596414379", "1596444832", "1596450560", "1596450802", "1596454143", "1596479945",
     #          "1596485467", "1596485699", "1596489082", "1596516521", "1596525946", "1596526192", "1596529501",
     #          "1596553484", "1596561093", "1596561322", "1596564704", "1596595541", "1596604431", "1596604622",
     #         "1596607509", "1596630544", "1596639464", "1596639649", "1596642538", "1596672248", "1596683659",
     #         "1596683840", "1596686401", "1596709811", "1596718871", "1596719042", "1596721616", "1596746123",
     #         "1596759993", "1596760102", "1596762700", "1596784123", "1596795103", "1596795150", "1596797763",
     #         "1596819094", "1596835082", "1596835093", "1596837656", "1596854477", "1596869949", "1596869960",
      #         "1596872548", "1596890418", "1596911384", "1596911583", "1596914043", "1596929673", "1596946541",
      #         "1596946785", "1596949246", "1596987783", "1596988070", "1596989989", "1597020996", "1597021394",
      #         "1597023204", "1597059046", "1597059501", "1597061142"]



    #ts_list = ["1598148899", "1598149283", "1598149994", "1598163229", "1598169861", "1598176752", "1598176969",
               #"1598189473", "1598190556", "1598202465", "1598208605", "1598225452"]

    ts_list = ["resnet50_50_patches"]

    not_found_list = []
    for ts in ts_list:
        print ("ts is: ", ts)
        not_found_list = evaluate(ts, not_found_list)


    print ("Not Found List:" , not_found_list)






if __name__ == '__main__':

    main()
    #concat_all_evaluation_results()
    #disease_embed_evaluate("schizophrenia")
    #concat_disease_evaluation_results("schizophrenia")




    





