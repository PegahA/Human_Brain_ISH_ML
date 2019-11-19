from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from human_ISH_config import *
import pandas as pd
import numpy as np


def build_distance_matrix(filename):
    """
    :param filename: String. This is the name of the folder in the EMBEDDING_DEST folder which containts the embeddings csv file
    :return: pandas DataFrame. A distance matrix that has the euclidean distance between all the possible pairs of embedding vectors
    """

    # embedding_file_name = EMBED_SET.split(".csv")[0] + "_embeddings.csv"
    # embeddings_csv_file = os.path.join(EMBEDDING_DEST, filename, embedding_file_name)
    embeddings_csv_file = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/embed.csv"


    embed_df = pd.read_csv(embeddings_csv_file)
    distances = euclidean_distances(embed_df.iloc[:, 1:], embed_df.iloc[:, 1:])
    embed_df = embed_df.set_index(['image_id'])
    # format distance matrix
    distances_df = pd.DataFrame(distances)
    distances_df.columns = list(embed_df.index)
    distances_df.index = list(embed_df.index)
    distances_df.values[[np.arange(distances_df.shape[0])] * 2] = float("inf")

    # dist_df_file_name = EMBED_SET.split(".csv")[0] + "_dist.csv"
    # dist_df.to_csv(os.path.join(EMBEDDING_DEST, filename, dist_df_file_name))

    return distances_df


def build_distance_matrix_2(filename):
    """
    :param filename: String. This is the name of the folder in the EMBEDDING_DEST folder which containts the embeddings csv file
    :return: pandas DataFrame. A distance matrix that has the euclidean distance between all the possible pairs of embedding vectors
    """

    #embedding_file_name = EMBED_SET.split(".csv")[0] + "_embeddings.csv"
    #embeddings_csv_file = os.path.join(EMBEDDING_DEST, filename, embedding_file_name)
    embeddings_csv_file = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/embed.csv"

    embed_df = pd.read_csv(embeddings_csv_file)
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
    min_indexes_df.columns = ["image_id1", "image_id2"]
    min_indexes_df = min_indexes_df.applymap(str)

    return min_indexes_df



def level_1_evaluation(min_indexes_df):
    """
    The level 1 evaluation checks to see for how many of the patches, the closest patch is from the same brain image.
    Which means the same gene, and the same donor, and the same brain tissue slice.
    :param min_indexes_df: pandas DataFrame. Has 2 columns. The first column is an image_id, the second column is the image_id of
    the corresponding closest image.
    :return:
    """

    total_count = len(min_indexes_df)

    # info_csv_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
    info_csv_path = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/valid_patches_info.csv"
    info_csv = pd.read_csv(info_csv_path, index_col=None)

    gene_donor_mapping = info_csv[['patch_id', 'image_id']]
    min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='image_id1', right_on='patch_id')
    min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='image_id2', right_on='patch_id')

    same_image = min_indexes_df.query('image_id_x == image_id_y')
    match_count = len(same_image)
    proportion = (match_count / total_count) * 100.0


    min_indexes_df['image_id1'] = [id.split("_")[0] for id in min_indexes_df['image_id1']]
    min_indexes_df['image_id2'] = [id.split("_")[0] for id in min_indexes_df['image_id2']]


    # patches that come from a certain image are named with the following format:
    # the actual brain image id _ patch index
    # therefore, if two patches are coming from the same image, when we remove the patch index, the remaining which is the
    # image_id must be the same
    match_count = len(min_indexes_df[min_indexes_df['image_id1']==min_indexes_df['image_id2']])

    proportion = (match_count / total_count) * 100.0

    return  proportion



def level_2_evaluation(min_indexes_df):
    """
    The level 2  evaluation checks to see for how many of the patches, the closest patch is not from the same brain image,
    but is the same gene and comes from the same donor.
    :param min_indexes_df: pandas DataFrame. Has 2 columns. The first column is an image_id, the second column is the image_id of
    the corresponding closest image.
    :return:
    """

    total_count = len(min_indexes_df)

    # info_csv_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
    info_csv_path = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/valid_patches_info.csv"
    info_csv = pd.read_csv(info_csv_path, index_col=None)

    gene_donor_mapping = info_csv[['patch_id', 'gene_symbol', 'donor_id', 'image_id']]
    min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='image_id1', right_on='patch_id')
    min_indexes_df = pd.merge(min_indexes_df, gene_donor_mapping, left_on='image_id2', right_on='patch_id')

    not_the_same_image = min_indexes_df.query('image_id_x != image_id_y')
    same_gene = not_the_same_image.query('gene_symbol_x == gene_symbol_y')
    same_donor = same_gene.query('donor_id_x == donor_id_y')
    match_count = len(same_donor)
    proportion = (match_count / total_count) * 100.0

    return proportion



def evaluate(filename):
    dist_df = build_distance_matrix(filename)
    min_indexes_df = find_closest_image(dist_df)
    level_1_proportion = level_1_evaluation(min_indexes_df)
    level_2_proportion = level_2_evaluation(min_indexes_df)


