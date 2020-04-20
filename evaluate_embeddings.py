from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from human_ISH_config import *
import pandas as pd
import numpy as np


def generate_level_3_positive_pairs(valid_patches_info_path):
    """
    This function generates level 3 positive pairs from the patches.
    Positive means each pair has the same gene.
    Level 3 means that each of the items in the pair are from a different donor.

    :return: None. Stores a csv file.
    """

    valid_patches_info = pd.read_csv(os.path.join(valid_patches_info_path, "valid_patches_info.csv"))

    by_gene = valid_patches_info.groupby("gene_symbol")
    genes = by_gene.groups.keys()
    genes = list(genes)
    genes.sort()

    col_1 = []
    col_2 = []

    counter = 1
    for gene in genes:
        this_gene_dict = {}
        print(counter, gene)
        counter += 1
        this_gene_df = by_gene.get_group(gene)
        this_gene_patch_id_list = this_gene_df['patch_id']
        # print ("This gene has {} patches associated with it.".format(len(this_gene_patch_id_list)))

        for patch_id in this_gene_patch_id_list:
            this_patch_donor = this_gene_df[this_gene_df['patch_id'] == patch_id]['donor_id'].iloc[0]

            not_this_donor = this_gene_df[this_gene_df['donor_id'] != this_patch_donor]

            not_this_donor_patch_ids = not_this_donor['patch_id']
            this_gene_dict[patch_id] = list(not_this_donor_patch_ids)

        for item in this_gene_dict:
            this_item_values = this_gene_dict[item]
            for val in this_item_values:
                col_2.append(val)
                col_1.append(item)

    positive_pairs_level_3_df = pd.DataFrame(columns=['col_1', 'col_2'])
    positive_pairs_level_3_df['col_1'] = col_1
    positive_pairs_level_3_df['col_2'] = col_2

    positive_pairs_path = os.path.join(valid_patches_info_path, "positive_pairs_level_3.csv")
    positive_pairs_level_3_df.to_csv(positive_pairs_path, index=None)



def generate_level_2_positive_pairs(valid_patches_info_path):
    """
    This function generates level 2 positive pairs from the patches.
    Positive means each pair has the same gene.
    Level 2 means that each of the items in the pair are from the same donor but different image.

    :return: None. Stores a csv file.
    """

    valid_patches_info = pd.read_csv(os.path.join(valid_patches_info_path, "valid_patches_info.csv"))

    by_gene = valid_patches_info.groupby("gene_symbol")
    genes = by_gene.groups.keys()
    genes = list(genes)
    genes.sort()

    col_1 = []
    col_2 = []

    counter = 1
    for gene in genes:
        this_gene_dict = {}
        print(counter, gene)
        counter += 1
        this_gene_df = by_gene.get_group(gene)
        this_gene_patch_id_list = this_gene_df['patch_id']
        # print ("This gene has {} patches associated with it.".format(len(this_gene_patch_id_list)))

        for patch_id in this_gene_patch_id_list:
            this_patch_donor = this_gene_df[this_gene_df['patch_id'] == patch_id]['donor_id'].iloc[0]
            this_patch_image = this_gene_df[this_gene_df['patch_id'] == patch_id]['image_id'].iloc[0]

            same_donor = this_gene_df[this_gene_df['donor_id'] == this_patch_donor]
            same_donor_not_this_image = same_donor[same_donor['image_id']!= this_patch_image]
            same_donor_not_this_image_patch_ids = same_donor_not_this_image['patch_id']

            this_gene_dict[patch_id] = list(same_donor_not_this_image_patch_ids)

        for item in this_gene_dict:
            this_item_values = this_gene_dict[item]
            for val in this_item_values:
                col_2.append(val)
                col_1.append(item)

    positive_pairs_level_3_df = pd.DataFrame(columns=['col_1', 'col_2'])
    positive_pairs_level_3_df['col_1'] = col_1
    positive_pairs_level_3_df['col_2'] = col_2

    positive_pairs_path = os.path.join(valid_patches_info_path, "positive_pairs_level_2.csv")
    positive_pairs_level_3_df.to_csv(positive_pairs_path, index=None)


def generate_level_3_negative_pairs(valid_patches_info_path):
    valid_patches_info = pd.read_csv(os.path.join(valid_patches_info_path, "valid_patches_info.csv"))
    patch_id_list = list(valid_patches_info['patch_id'])

    negatives_for_each_patch_dict= {}
    col_1 = []
    col_2 = []

    counter = 1
    for patch_id in patch_id_list:
        print (counter, patch_id)
        counter += 1

        this_patch_gene = valid_patches_info[valid_patches_info['patch_id']== patch_id]['gene_symbol'].iloc[0]
        this_patch_donor = valid_patches_info[valid_patches_info['patch_id']== patch_id]['donor_id'].iloc[0]

        not_same_gene = valid_patches_info[valid_patches_info['gene_symbol']!= this_patch_gene]
        not_same_gene_not_same_donor = not_same_gene[not_same_gene['donor_id']!=this_patch_donor]
        not_same_gene_not_same_donor_patch_id_list = not_same_gene_not_same_donor['patch_id']

        negatives_for_each_patch_dict[patch_id] = list(not_same_gene_not_same_donor_patch_id_list)

    for item in negatives_for_each_patch_dict:
        this_item_values = negatives_for_each_patch_dict[item]
        for val in this_item_values:
            col_2.append(val)
            col_1.append(item)

    positive_pairs_level_3_df = pd.DataFrame(columns=['col_1', 'col_2'])
    positive_pairs_level_3_df['col_1'] = col_1
    positive_pairs_level_3_df['col_2'] = col_2

    positive_pairs_path = os.path.join(valid_patches_info_path, "negative_pairs_level_3.csv")
    positive_pairs_level_3_df.to_csv(positive_pairs_path, index=None)



def temp():

    df = pd.read_csv( "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_2/positive_pairs_level_2.csv")
    print (len(df))

        




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


def filter_dist_matrix_after_level_1(dist_matrix, level):


    if level == 'image':
        print ("skipping filtering after level 1 ...")
        return dist_matrix

    elif level == 'patch':
        info_csv_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
        info_csv = pd.read_csv(info_csv_path, index_col=None)
        patch_id_list = list(dist_matrix.index)


        # for each patch_id row, I need to set some of the cells to nan.
        for patch_id in patch_id_list:
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


        return dist_matrix

def filter_dist_matrix_after_level_2(dist_matrix, level):

    #print ("filtering the distance matrix after level 2 ...")
    if level == 'patch':
        info_csv_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
        info_csv = pd.read_csv(info_csv_path, index_col=None)
        patch_id_list = list(dist_matrix.index)

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



        return dist_matrix

    elif level == 'image':
        info_csv_path = os.path.join(DATA_DIR, STUDY, "human_ISH_info.csv")
        info_csv = pd.read_csv(info_csv_path, index_col=None)
        image_id_list = list(dist_matrix.index)
        index = 0

        for image_id in image_id_list:
            #print (image_id)
            #print (index)
            #index = index +1 
            #this_image_id = image_id
            this_donor_id = info_csv[info_csv['image_id'] == image_id].donor_id
            this_donor_id = this_donor_id.values[0]

            #same_image_id = info_csv['image_id'] == this_image_id
            same_donor_id = info_csv['donor_id'] == this_donor_id

            #res1 = list(info_csv[same_image_id | same_donor_id].image_id.values)
            res = list(info_csv[same_donor_id].image_id.values)   
            #res = res1
            res = [item for item in res if item in image_id_list]

            dist_matrix.loc[image_id, res] = np.nan
        
        print ("finished")
        return dist_matrix


def level_1_evaluation(min_indexes_df, level):
    """
    The level 1 evaluation checks to see for how many of the patches, the closest patch is from the same brain image.
    Which means the same gene, and the same donor, and the same brain tissue slice.
    :param min_indexes_df: pandas DataFrame. Has 2 columns. The first column is an image_id, the second column is the image_id of
    the corresponding closest image.
    :return: float. The proportion of matches.
    """

    print ("in level 1 ...")
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

    print ("in level 2 ... ")
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

        return proportion



def level_3_evaluation(min_indexes_df, level):
    """
    The level 3 evaluation checks to see for how many of the patches, the closest patch is from the same gene
    but not from the same donor.
    :param min_indexes_df: pandas DataFrame. Has 2 columns. The first column is an image_id, the second column is the image_id of
    the corresponding closest image.
    :return: float. The proportion of matches.
    """

    print ("in level 3 ...")

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
        
        print ("proportion is: ", proportion)
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

    return level_1_proportion, level_2_proportion, level_3_proportion


def evaluate_with_filtering(path_to_embeddings, level):
    """
       In this case, I compute the distance matrix once, and take the closest image to every image to perform level 1.
       Then, before moving to the next level, I modify the distance matrix by assigning nan to those cells that
       met the conditions in the previous level, and calculate the closest images again using the new filtered distance matrix.


       """

    # embedding_file_name = EMBED_SET.split(".csv")[0] + "_embeddings.csv"
    # path_to_embeddings = os.path.join(EMBEDDING_DEST, filename, embedding_file_name)
    #path_to_embeddings = "/Users/pegah_abed/Documents/old_Human_ISH/cortex/embed.csv"


    #print ("not the same gene")
    #not_the_same_gene(min_indexes_df, level)

    print("level 1")
    dist_df = build_distance_matrix(path_to_embeddings)
    min_indexes_df = find_closest_image(dist_df)
    print (min_indexes_df)
    level_1_proportion = level_1_evaluation(min_indexes_df, level)
    print(level_1_proportion)


    print ("level 2")
    new_dist_df = filter_dist_matrix_after_level_1(dist_df, level)
    min_indexes_df = find_closest_image(new_dist_df)
    print (min_indexes_df)
    level_2_proportion = level_2_evaluation(min_indexes_df, level)
    print(level_2_proportion)

    print ("-level 3-")
    new_dist_df = filter_dist_matrix_after_level_2(dist_df, level)
    min_indexes_df = find_closest_image(new_dist_df)
    print(min_indexes_df)
    level_3_proportion = level_3_evaluation(min_indexes_df, level)
    print(level_3_proportion)


    return level_1_proportion, level_2_proportion, level_3_proportion



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
        #print ("sum 100 -----------------")
        #l_1, l_2, l_3 = evaluate_sum_100(path_to_embeddings, level)

        print ("with filtering ----------------")
        l_1, l_2, l_3 = evaluate_with_filtering(path_to_embeddings, level)

    return l_1, l_2, l_3



def main():
    #ts_list = ["1584753511"]
    ts_list =  ["resnet50_10_patches_standardized"]

    for ts in ts_list:

        columns = ["ts", "level_1", "level_2", "level_3"]
        eval_results_df = pd.DataFrame(columns=columns)

        print ("ts is: ", ts)
        l_1, l_2, l_3 = evaluate(ts, 'image')

        print (l_1, l_2, l_3)
         
        eval_results_df.loc[0] = [ts, l_1, l_2, l_3]
        #eval_results_df["ts"] = ts
        #eval_results_df["level_1"] = l_1
        #eval_results_df["level_2"] = l_2
        #eval_results_df["level_3"] = l_3
       
        print (eval_results_df)
        eval_path = os.path.join(EMBEDDING_DEST, ts)
        eval_results_df.to_csv(os.path.join(eval_path, "evaluation_result.csv"), index=None)
        
        print (ts)
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


if __name__ == '__main__':
    valid_patches_info_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_2"
    generate_level_3_negative_pairs(valid_patches_info_path)

