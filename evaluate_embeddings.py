from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from human_ISH_config import *
import pandas as pd
import numpy as np


def build_distance_matrix(path_to_embeddings):
    """
    :param filename: String. This is the name of the folder in the EMBEDDING_DEST folder which containts the embeddings csv file
    :return: pandas DataFrame. A distance matrix that has the euclidean distance between all the possible pairs of embedding vectors
    """

    embed_df = pd.read_csv(path_to_embeddings)
    distances = euclidean_distances(embed_df.iloc[:, 1:], embed_df.iloc[:, 1:])
    embed_df = embed_df.set_index(['image_id'])
    # format distance matrix
    distances_df = pd.DataFrame(distances)
    distances_df.columns = list(embed_df.index)
    distances_df.index = list(embed_df.index)
    distances_df.values[[np.arange(distances_df.shape[0])] * 2] = float("inf")

    # dist_df_file_name = EMBED_SET.split(".csv")[0] + "_dist.csv"
    # dist_df.to_csv(os.path.join(EMBEDDING_DEST, filename, dist_df_file_name))
    distances_df.to_csv("/Users/pegah_abed/Documents/old_Human_ISH/cortex/dist.csv")
    return distances_df


def build_distance_matrix_2(path_to_embeddings):
    """
    :param filename: String. This is the name of the folder in the EMBEDDING_DEST folder which containts the embeddings csv file
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

    #dist_df_file_name = EMBED_SET.split(".csv")[0] + "_dist.csv"
    #dist_df.to_csv(os.path.join(EMBEDDING_DEST, filename, dist_df_file_name))

    return dist_df



def find_closest_image(distances_df):
    """
    :param distances_df: pandas DataFrame. A distance matrix that has the euclidean distance between all the possible
    pairs of embedding vectors
    :return: pandas DataFrame. Has 2 columns. The first column is an image_id, the second column is the image_id of
    the corresponding closest image.
    """

    # find the closest image
    min_indexes = distances_df.idxmin(axis=1, skipna=True)
    min_indexes_df = pd.DataFrame(min_indexes).reset_index()
    min_indexes_df.columns = ["id1", "id2"]
    min_indexes_df = min_indexes_df.applymap(str)

    return min_indexes_df


def filter_dist_matrix_after_level_1(dist_matrix):



    info_csv_path = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/valid_patches_info.csv"
    info_csv = pd.read_csv(info_csv_path, index_col=None)
    patch_id_list = list(dist_matrix.index)

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




    dist_matrix.to_csv("/Users/pegah_abed/Documents/old_Human_ISH/cortex/dist_after_1.csv")
    return dist_matrix

def filter_dist_matrix_after_level_2(dist_matrix):
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


def level_1_evaluation(min_indexes_df):
    """
    The level 1 evaluation checks to see for how many of the patches, the closest patch is from the same brain image.
    Which means the same gene, and the same donor, and the same brain tissue slice.
    :param min_indexes_df: pandas DataFrame. Has 2 columns. The first column is an image_id, the second column is the image_id of
    the corresponding closest image.
    :return: float. The proportion of matches.
    """

    total_count = len(min_indexes_df)

    # info_csv_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
    info_csv_path = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/valid_patches_info.csv"
    info_csv = pd.read_csv(info_csv_path, index_col=None)

    gene_donor_mapping = info_csv[['patch_id', 'image_id']]
    min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='id1', right_on='patch_id')
    min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='id2', right_on='patch_id')

    same_image = min_indexes_df.query('image_id_x == image_id_y')

    match_count = len(same_image)
    proportion = (match_count / total_count) * 100.0


    min_indexes_df['id1'] = [id.split("_")[0] for id in min_indexes_df['id1']]
    min_indexes_df['id2'] = [id.split("_")[0] for id in min_indexes_df['id2']]

    return  proportion



def level_2_evaluation(min_indexes_df):
    """
    The level 2  evaluation checks to see for how many of the patches, the closest patch is not from the same brain image,
    but is the same gene and comes from the same donor.
    :param min_indexes_df: pandas DataFrame. Has 2 columns. The first column is an image_id, the second column is the image_id of
    the corresponding closest image.
    :return: float. The proportion of matches.
    """

    total_count = len(min_indexes_df)

    # info_csv_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
    info_csv_path = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/valid_patches_info.csv"
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


    print (match_count)
    return proportion


def level_3_evaluation(min_indexes_df):
    """
    The level 3 evaluation checks to see for how many of the patches, the closest patch is from the same gene
    but not from the same donor.
    :param min_indexes_df: pandas DataFrame. Has 2 columns. The first column is an image_id, the second column is the image_id of
    the corresponding closest image.
    :return: float. The proportion of matches.
    """

    total_count = len(min_indexes_df)

    # info_csv_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
    info_csv_path = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/valid_patches_info.csv"
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



def evaluate_sum_100(filename):

    # embedding_file_name = EMBED_SET.split(".csv")[0] + "_embeddings.csv"
    # path_to_embeddings = os.path.join(EMBEDDING_DEST, filename, embedding_file_name)
    path_to_embeddings = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/embed.csv"

    dist_df = build_distance_matrix(path_to_embeddings)
    min_indexes_df = find_closest_image(dist_df)

    print("level 1")
    level_1_proportion = level_1_evaluation(min_indexes_df)
    print(level_1_proportion)

    print("level 2")
    level_2_proportion = level_2_evaluation(min_indexes_df)
    print(level_2_proportion)

    print("level 3")
    level_3_proportion = level_3_evaluation(min_indexes_df)
    print(level_3_proportion)


def evaluate_with_filtering():

    # embedding_file_name = EMBED_SET.split(".csv")[0] + "_embeddings.csv"
    # path_to_embeddings = os.path.join(EMBEDDING_DEST, filename, embedding_file_name)
    path_to_embeddings = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/embed.csv"

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



def main():
    path_to_embeddings = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/embed.csv"
    embed_df = pd.read_csv(path_to_embeddings)

    dist_df = build_distance_matrix(path_to_embeddings)
    new_dist_df = filter_dist_matrix_after_level_1(dist_df)


main()

