from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from human_ISH_config import *
import pandas as pd
import numpy as np


def build_distance_matrix(path_to_embeddings):
    """
    Distance from one item to itself shows up as inf.
    :param filename: String. This is the name of the folder in the EMBEDDING_DEST folder which contains the embeddings csv file
    :return: pandas DataFrame. A distance matrix that has the euclidean distance between all the possible pairs of embedding vectors
    """

    embed_df = pd.read_csv(path_to_embeddings)
    print ("length is: ", len(embed_df))
    columns = list(embed_df)

    # ------- these line are to convert the data types from float64 to float32.
    # We need to create a dict. Keys will be column names and values will be types.
    # The first column is name and remains as string. The rest should be converted to float32.


    #type_dict = {}
    #type_dict[columns[0]] = 'string'
    #for column in columns[1:]:
        #type_dict[column] = 'float32'
 
    #embed_df = embed_df.astype(type_dict)
    # -------------------------------------------------------------------------
   
    distances = euclidean_distances(embed_df.iloc[:, 1:], embed_df.iloc[:, 1:])
    embed_df = embed_df.set_index([columns[0]])
    # format distance matrix
    distances_df = pd.DataFrame(distances)
    distances_df.columns = list(embed_df.index)
    distances_df.index = list(embed_df.index)
    distances_df.values[[np.arange(distances_df.shape[0])] * 2] = float("inf")

    print ("finished building the distance matrix ...")
    return distances_df


def build_distance_matrix_2(path_to_embeddings):
    """
    Distance from one item to itself shows up as 0. Could be misleading later when we are trying to find the closest image.
    :param filename: String. This is the name of the folder in the EMBEDDING_DEST folder which contains the embeddings csv file
    :return: pandas DataFrame. A distance matrix that has the euclidean distance between all the possible pairs of embedding vectors
    """

    embed_df = pd.read_csv(path_to_embeddings)
    image_id_list = embed_df['image_id']
    embed_df = embed_df.set_index(['image_id'])

    dist_df =pd.DataFrame(
        squareform(pdist(embed_df.loc[image_id_list])),
        columns=image_id_list,
        index=image_id_list
    )


    return dist_df



def find_closest_image(distances_df):
    """
    :param distances_df: pandas DataFrame. A distance matrix that has the euclidean distance between all the possible
    pairs of embedding vectors
    :return: pandas DataFrame. Has 2 columns. The first column is an image_id, the second column is the image_id of
    the corresponding closest image.
    """

    # find the closest image in each row

    min_indexes = distances_df.idxmin(axis=1, skipna=True)
    min_indexes_df = pd.DataFrame(min_indexes).reset_index()
    min_indexes_df.columns = ["id1", "id2"]
    min_indexes_df = min_indexes_df.applymap(str)

    print("finished finding the closest image ...")
    return min_indexes_df


def filter_dist_matrix_after_level_1(dist_matrix):

    info_csv_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
    info_csv = pd.read_csv(info_csv_path, index_col=None)
    patch_id_list = list(dist_matrix.index)


    # for each patch_id row, I need to set some of the cells to nan.
    for patch_id in patch_id_list:
        print (patch_id)
        this_image_id = info_csv[info_csv['patch_id'] == patch_id].image_id
        this_image_id = this_image_id.values[0]

        this_donor_id = info_csv[info_csv['patch_id'] == patch_id].donor_id
        this_donor_id = this_donor_id.values[0]


        not_same_donor_id = info_csv['donor_id'] != this_donor_id
        same_image_id = info_csv['image_id'] == this_image_id
        not_same_patch_id = info_csv['patch_id'] != patch_id
        res1 = list(info_csv[not_same_donor_id].patch_id.values)
        res2 = list(info_csv[same_image_id & not_same_patch_id].patch_id.values)
        res = res1 + res2

        res=[item for item in res if item in patch_id_list]


        dist_matrix.loc[patch_id,res] = np.nan



    dist_matrix.to_csv("/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation//dist_after_1.csv")
    return dist_matrix

def filter_dist_matrix_after_level_2(dist_matrix):


    info_csv_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
    info_csv_path = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/valid_patches_info.csv"
    info_csv = pd.read_csv(info_csv_path, index_col=None)
    patch_id_list = info_csv['patch_id'][:9]

    for patch_id in patch_id_list:
        this_image_id = info_csv[info_csv['patch_id'] == patch_id].image_id
        this_image_id = this_image_id.values[0]
        this_donor_id = info_csv[info_csv['patch_id'] == patch_id].donor_id
        this_donor_id = this_donor_id.values[0]

        same_image_id = info_csv['image_id'] == this_image_id
        same_donor_id = info_csv['donor_id'] == this_donor_id
        not_same_patch_id = info_csv['patch_id'] != patch_id

        res1 = list(info_csv[same_image_id | same_donor_id].patch_id.values)
        res2 = list(info_csv[not_same_patch_id].patch_id.values)

        res = res1 + res2
        res = [item for item in res if item in patch_id_list]

        dist_matrix.loc[patch_id, res] = np.nan


    dist_matrix.to_csv("/Users/pegah_abed/Documents/old_Human_ISH/cortex/dist_after_2.csv")
    return dist_matrix


def level_1_evaluation(min_indexes_df, level):
    """
    The level 1 evaluation checks to see for how many of the patches, the closest patch is from the same brain image.
    Which means the same gene, and the same donor, and the same brain tissue slice.
    :param min_indexes_df: pandas DataFrame. Has 2 columns. The first column is an image_id, the second column is the image_id of
    the corresponding closest image.
    :return: float. The proportion of matches.
    """


    if level == 'image':
        print ("skipping level 1 evaluation ...")
        return None

    elif level == 'patch':

        total_count = len(min_indexes_df)

        info_csv_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
        info_csv = pd.read_csv(info_csv_path, index_col=None)

        gene_donor_mapping = info_csv[['patch_id', 'image_id']]
        min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='id1', right_on='patch_id')
        min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='id2', right_on='patch_id')

        same_image = min_indexes_df.query('image_id_x == image_id_y')

        match_count = len(same_image)
        proportion = (match_count / total_count) * 100.0


        min_indexes_df['id1'] = [id.split("_")[0] for id in min_indexes_df['id1']]
        min_indexes_df['id2'] = [id.split("_")[0] for id in min_indexes_df['id2']]


        print (match_count)
        return  proportion



def level_2_evaluation(min_indexes_df, level):
    """
    The level 2  evaluation checks to see for how many of the patches, the closest patch is not from the same brain image,
    but is the same gene and comes from the same donor.
    :param min_indexes_df: pandas DataFrame. Has 2 columns. The first column is an image_id, the second column is the image_id of
    the corresponding closest image.
    :return: float. The proportion of matches.
    """


    if level == 'patch':
        total_count = len(min_indexes_df)

        info_csv_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
        info_csv = pd.read_csv(info_csv_path, index_col=None)

        gene_donor_mapping = info_csv[['patch_id', 'gene_symbol', 'donor_id', 'image_id']]
        min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='id1', right_on='patch_id')
        min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='id2', right_on='patch_id')

        not_the_same_image = min_indexes_df.query('image_id_x != image_id_y')
        same_gene = not_the_same_image.query('gene_symbol_x == gene_symbol_y')
        same_donor = same_gene.query('donor_id_x == donor_id_y')

        print (same_donor)
        match_count = len(same_donor)
        proportion = (match_count / total_count) * 100.0

        return proportion

    elif level == 'image':

        total_count = len(min_indexes_df)
        print ("total number of images: ", total_count)

        info_csv_path = os.path.join(DATA_DIR, STUDY, "human_ISH_info.csv")
        info_csv = pd.read_csv(info_csv_path, index_col=None)

        gene_donor_mapping = info_csv[['gene_symbol', 'donor_id', 'image_id']]
        gene_donor_mapping['image_id']=gene_donor_mapping['image_id'].astype(str)
        min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='id1', right_on='image_id')
        min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='id2', right_on='image_id')

        not_the_same_image = min_indexes_df.query('image_id_x != image_id_y')
        same_gene = not_the_same_image.query('gene_symbol_x == gene_symbol_y')
        same_donor = same_gene.query('donor_id_x == donor_id_y')

        print( same_donor)
        match_count = len(same_donor)
        print("number of matches with the same gene and the same donor: ", match_count)
        proportion = (match_count / total_count) * 100.0
     
        print ("proportion: ", proportion)
        print ("\n\n")

        return proportion



def level_3_evaluation(min_indexes_df, level):
    """
    The level 3 evaluation checks to see for how many of the patches, the closest patch is from the same gene
    but not from the same donor.
    :param min_indexes_df: pandas DataFrame. Has 2 columns. The first column is an image_id, the second column is the image_id of
    the corresponding closest image.
    :return: float. The proportion of matches.
    """


    if level == 'patch':

        total_count = len(min_indexes_df)

        info_csv_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
        info_csv = pd.read_csv(info_csv_path, index_col=None)

        gene_donor_mapping = info_csv[['patch_id', 'gene_symbol', 'donor_id', 'image_id']]
        min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='id1', right_on='patch_id')
        min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='id2', right_on='patch_id')

        not_the_same_image = min_indexes_df.query('image_id_x != image_id_y')
        not_the_same_donor = not_the_same_image.query('donor_id_x != donor_id_y')
        same_gene = not_the_same_donor.query('gene_symbol_x == gene_symbol_y')
        print (same_gene)

        match_count = len(same_gene)
        print (match_count)
        proportion = (match_count / total_count) * 100.0

        return proportion

    elif level == 'image':

        total_count = len(min_indexes_df)
        print ("total number of images: ", total_count)
        info_csv_path = os.path.join(DATA_DIR, STUDY, "human_ISH_info.csv")
        info_csv = pd.read_csv(info_csv_path, index_col=None)

        gene_donor_mapping = info_csv[['gene_symbol', 'donor_id', 'image_id']]
        gene_donor_mapping['image_id']=gene_donor_mapping['image_id'].astype(str)
        min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='id1', right_on='image_id')
        min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='id2', right_on='image_id')

        not_the_same_image = min_indexes_df.query('image_id_x != image_id_y')
        not_the_same_donor = not_the_same_image.query('donor_id_x != donor_id_y')
        same_gene = not_the_same_donor.query('gene_symbol_x == gene_symbol_y')
        print(same_gene)

        match_count = len(same_gene)
        print("number of matches with the same gene and not the same donor: ", match_count)
        proportion = (match_count / total_count) * 100.0
     
        print ("proportion: ", proportion)
        print ("\n\n")
        return proportion

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

        return proportion


def evaluate_sum_100(path_to_embeddings, level):
    """
    In this case, I compute the distance matrix once, and take the closest image to every image.
    Then, I check to see what percentage falls into condition of level 1,
                         what percentage falls into condition of level 2,
                     and what percentage falls into condition of level 3.


    """


    # embedding_file_name = EMBED_SET.split(".csv")[0] + "_embeddings.csv"
    # path_to_embeddings = os.path.join(EMBEDDING_DEST, filename, embedding_file_name)
    # path_to_embeddings = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/embed.csv"

    dist_df = build_distance_matrix(path_to_embeddings)
    min_indexes_df = find_closest_image(dist_df)


    print ("not the same gene")
    not_the_same_gene(min_indexes_df, level)

    print("level 1")
    level_1_proportion = level_1_evaluation(min_indexes_df, level)
    print(level_1_proportion)

    print("level 2")
    level_2_proportion = level_2_evaluation(min_indexes_df, level)
    print(level_2_proportion)

    print("level 3")
    level_3_proportion = level_3_evaluation(min_indexes_df, level)
    print(level_3_proportion)




def evaluate_with_filtering(path_to_embeddings):
    """
       In this case, I compute the distance matrix once, and take the closest image to every image to perform level 1.
       Then, before moving to the next level, I modify the distance matrix by assigning nan to those cells that
       met the conditions in the previous level, and calculate the closest images again using the new filtered distance matrix.


       """

    # embedding_file_name = EMBED_SET.split(".csv")[0] + "_embeddings.csv"
    # path_to_embeddings = os.path.join(EMBEDDING_DEST, filename, embedding_file_name)
    #path_to_embeddings = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/embed.csv"

    print("level 1")
    dist_df = build_distance_matrix(path_to_embeddings)
    min_indexes_df = find_closest_image(dist_df)
    print (min_indexes_df)
    level_1_proportion = level_1_evaluation(min_indexes_df)
    print(level_1_proportion)


    print ("level 2")
    new_dist_df = filter_dist_matrix_after_level_1(dist_df)
    min_indexes_df = find_closest_image(new_dist_df)
    print (min_indexes_df)
    level_2_proportion = level_2_evaluation(min_indexes_df)
    print(level_2_proportion)

    print ("level 3")
    new_dist_df = filter_dist_matrix_after_level_2(dist_df)
    min_indexes_df = find_closest_image(new_dist_df)
    print(min_indexes_df)
    level_2_proportion = level_3_evaluation(min_indexes_df)
    print(level_2_proportion)




def evaluate(ts, level):

    base_path = os.path.join(EMBEDDING_DEST, ts)
    contents = os.listdir(base_path)
    embeddings_files = []

    if level == 'patch':
        for item in contents:
            if item.endswith("embeddings.csv"):
                embeddings_files.append(item)


    elif level == 'image':
        for item in contents:
            if item.endswith("embeddings_image_level.csv"):
                embeddings_files.append(item)

    print ("list of embedding files is: ")
    print (embeddings_files)

    for item in embeddings_files:
        path_to_embeddings = os.path.join(EMBEDDING_DEST, ts, item)
        print (item)
        print ("sum 100 -----------------")
        evaluate_sum_100(path_to_embeddings, level)
 
        #print ("with filtering ----------------")
        #evaluate_with_filtering(path_to_embeddings, level)



def main():
    ts = "1583770480"
    evaluate(ts, 'image')
    #path_to_embeddings = "/Users/pegah_abed/Documents/old_Human_ISH/test_df.csv"
   # path_to_embeddings = os.path.join("/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation", ts, "mini_embeddings.csv")
    #dist = build_distance_matrix(path_to_embeddings)

    #find_closest_image(dist)

    #evaluate_sum_100(path_to_embeddings)


    """
    embed_file_name = "triplet_training_validation_embeddings.csv"
    embed_dir = os.path.join(DATA_DIR, STUDY, "segmentation_embeddings")
    ts_list = os.listdir(embed_dir) 

    for ts in ts_list:
        path_to_embeddings = os.path.join(embed_dir, ts, embed_file_name)
        print (path_to_embeddings)
        print ("{} --------------".format(ts))
        evaluate(path_to_embeddings)

        print ("________________________")
   
    #path_to_embeddings = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/embed.csv"
    #embed_df = pd.read_csv(path_to_embeddings)

    #dist_df = build_distance_matrix(path_to_embeddings)
    #new_dist_df = filter_dist_matrix_after_level_1(dist_df)
    """

main()
