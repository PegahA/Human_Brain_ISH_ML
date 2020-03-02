import os
import pandas as pd
import numpy as np
import random
from human_ISH_config import *
#import h5py
import time
from shutil import copyfile
import operator
import matplotlib.pyplot as plt

random.seed(1)

#if (not os.path.exists(os.path.join(DATA_DIR, STUDY, "sets"))):
    #os.mkdir(os.path.join(DATA_DIR, STUDY, "sets"))



def get_stats(images_info_df):
    """
    Uses the images_info_df and calculates some stats.
    :param images_info_df: pandas dataframe that has the information of all image
    :return: a dictionary containing stats.
    """

    stats_dict = {'image_count':None, 'donor_count':None, 'female_donor_count':None, 'male_donor_count':None,
                  'unique_genes_count': None, 'unique_entrez_id_count' : None}

    image_id_list = images_info_df['image_id']
    gene_symbol_list = images_info_df['gene_symbol']
    entrez_id_list = images_info_df['entrez_id']
    experiment_id_list = images_info_df['experiment_id']
    specimen_id_list = images_info_df['specimen_id']
    donor_id_list = images_info_df['donor_id']
    donor_sex_list = images_info_df['donor_sex']
    female_donors = images_info_df[images_info_df['donor_sex'] == 'F']
    male_donors = images_info_df[images_info_df['donor_sex'] == 'M']

    # -----------


    # How many donors does this study have? How many are female and how many are male?
    donors_count = len(set(images_info_df['donor_id']))
    print ("Total number of donors: {}".format(donors_count))

    female_donors_count = len(set(female_donors['donor_id']))
    print("Number of female donors: {}".format(female_donors_count))

    male_donors_count = len(set(male_donors['donor_id']))
    print("Number of male donors: {}".format(male_donors_count))


    if female_donors_count + male_donors_count != donors_count:
        print ("something is not right about the number of female and male donors ...")

    # -----------

    # How many unique genes does this study include?
    gene_count = len(set(gene_symbol_list))
    print ("Number of unique genes: {}".format(gene_count))
    entrez_id_count = len(set(entrez_id_list))
    print("Number of unique entrez IDs: {}".format(entrez_id_count))

    if entrez_id_count != gene_count:
        print ("something is not right. The number of unique genes should be equal to the number of unique entrez IDs")

    # -----------

    # How many genes have been tested from each donor.
    # How many images do we have from each donor.
    group_by_donor = images_info_df.groupby('donor_id')
    unique_gene_count_per_donor_list = []
    unique_image_count_per_donor_list = []
    for key, item in group_by_donor:
        this_group_genes = group_by_donor.get_group(key)['gene_symbol']
        this_group_images = group_by_donor.get_group(key)['image_id']
        unique_gene_count_per_donor_list.append(len(set(this_group_genes)))
        unique_image_count_per_donor_list.append(len(set(this_group_images)))

    print("Minimum number of unique genes from a donor: {}".format(min(unique_gene_count_per_donor_list)))
    print("Maximum number of unique genes from a donor: {}".format(max(unique_gene_count_per_donor_list)))
    print("Average number of unique genes from a donor: {}".format(np.mean(unique_gene_count_per_donor_list)))

    print("Minimum number of images from a donor: {}".format(min(unique_image_count_per_donor_list)))
    print("Maximum number of images from a donor: {}".format(max(unique_image_count_per_donor_list)))
    print("Average number of images from a donor: {}".format(np.mean(unique_image_count_per_donor_list)))

    # -----------

    # How many images do we have from each gene.
    # How many donors do we have from each gene.
    group_by_gene = images_info_df.groupby('gene_symbol')
    unique_donor_count_per_gene_list = []
    unique_image_count_per_gene_list = []
    for key, item in group_by_gene:
        this_group_donors = group_by_gene.get_group(key)['donor_id']
        this_group_images = group_by_gene.get_group(key)['image_id']
        unique_donor_count_per_gene_list.append(len(set(this_group_donors)))
        unique_image_count_per_gene_list.append(len(set(this_group_images)))

    print("Minimum number of unique donors from a gene: {}".format(min(unique_donor_count_per_gene_list)))
    print("Maximum number of unique donors from a gene: {}".format(max(unique_donor_count_per_gene_list)))
    print("Average number of unique donors from a gene: {}".format(np.mean(unique_donor_count_per_gene_list)))

    print("Minimum number of images from a gene: {}".format(min(unique_image_count_per_gene_list)))
    print("Maximum number of images from a gene: {}".format(max(unique_image_count_per_gene_list)))
    print("Average number of images from a gene: {}".format(np.mean(unique_image_count_per_gene_list)))


    gene_on_all_donors_count = 0
    gene_on_only_one_donor_count = 0

    for item in unique_donor_count_per_gene_list:
        if item == donors_count:
            gene_on_all_donors_count +=1
        if item == 1:
            gene_on_only_one_donor_count += 1


    print ("There are {} genes that have been sampled from all the {} donors.".format(gene_on_all_donors_count, donors_count))
    print ("There are {} genes that have been sampled from only 1 donor.".format(gene_on_only_one_donor_count))
    # -----------

    stats_dict['image_count'] = len(image_id_list)
    stats_dict['donor_count'] = donors_count
    stats_dict['female_donor_count'] = female_donors_count
    stats_dict['male_donor_count'] = male_donors_count
    stats_dict['unique_genes_count'] = gene_count
    stats_dict['unique_entrez_id_count'] = entrez_id_count

    return stats_dict


def define_sets_with_no_shared_genes(images_info_df):
    """
    We want to create training, validation, and test set.
    The condition is that the sets should not have any genes in common.
    :param images_info_df:  pandas dataframe that has the information of all image
    :return: 3 pandas dataframes: training, validation, test
    """

    unique_genes = list(np.unique(images_info_df['gene_symbol']))
    total_unique_gene_count = len(unique_genes)
    print(total_unique_gene_count)


    test_genes_count = int((TEST_SPLIT / 100.0) * total_unique_gene_count)
    validation_gene_count = int((VALIDATION_SPLIT / 100.0) * total_unique_gene_count)

    test_genes = random.sample(unique_genes, test_genes_count)

    remaining_genes = [x for x in unique_genes if x not in test_genes]
    validation_genes = random.sample(remaining_genes, validation_gene_count)


    training_genes = [x for x in remaining_genes if x not in validation_genes]

    training_df = images_info_df[images_info_df['gene_symbol'].isin(training_genes)]
    validation_df = images_info_df[images_info_df['gene_symbol'].isin(validation_genes)]
    test_df = images_info_df[images_info_df['gene_symbol'].isin(test_genes)]

    training_df = training_df.sort_values(by=['image_id'])
    validation_df = validation_df.sort_values(by=['image_id'])
    test_df = test_df.sort_values(by=['image_id'])

    train_val_df = pd.concat([training_df, validation_df], ignore_index=True)
    train_val_df = train_val_df.sort_values(by=['image_id'])

    training_df.to_csv(os.path.join(DATA_DIR, STUDY, "sets", "training.csv"), index=None)
    validation_df.to_csv(os.path.join(DATA_DIR, STUDY, "sets", "validation.csv"), index=None)
    test_df.to_csv(os.path.join(DATA_DIR, STUDY, "sets", "test.csv"), index=None)
    train_val_df.to_csv(os.path.join(DATA_DIR, STUDY, "sets", "training_validation.csv"), index=None)

    return training_df, validation_df, test_df, train_val_df


def define_sets_with_no_shared_donors(images_info_df):
    """
    We want to create training, validation, and test set.
    The condition is that the sets should not have any donors in common.
    :param images_info_df:  pandas dataframe that has the information of all image
    :return: 3 pandas dataframes: training, validation, test
    """


    gene_count_threshold_for_test_set  = 60 #70
    gene_count_threshold_for_validation_set = 85 #90
    group_by_donor = images_info_df.groupby('donor_id')

    test_set_donor_list = []
    validation_set_donor_list = []
    training_set_donor_list = []

    for key, item in group_by_donor:
        this_group_genes = group_by_donor.get_group(key)['gene_symbol']

        if len(set(this_group_genes)) < gene_count_threshold_for_test_set:
            test_set_donor_list.append(key)

        elif len(set(this_group_genes)) < gene_count_threshold_for_validation_set:
            validation_set_donor_list.append(key)
        else:
            training_set_donor_list.append(key)


    print ("\n---- test set ----")
    #print (test_set_info_list)

    test_set_image_count = 0
    test_set_gene_list = []
    test_set_donor_count = len(test_set_donor_list)

    for item in test_set_donor_list:
        this_group_images = group_by_donor.get_group(item)['image_id']
        this_group_genes = group_by_donor.get_group(item)['gene_symbol']
        test_set_image_count += len(set(this_group_images))
        test_set_gene_list.extend(set(this_group_genes))

    print ("number of donors in test set: ",test_set_donor_count)
    print ("test set image count" , test_set_image_count)
    print("test set unique gene count", len(set(test_set_gene_list)))



    print("\n---- validation set ----")
    #print(validation_set_info_list)

    validation_set_image_count = 0
    validation_set_gene_list= []
    validation_set_donor_count = len(validation_set_donor_list)

    for item in validation_set_donor_list:
        this_group_images = group_by_donor.get_group(item)['image_id']
        this_group_genes = group_by_donor.get_group(item)['gene_symbol']
        validation_set_image_count += len(set(this_group_images))
        validation_set_gene_list.extend(set(this_group_genes))

    print("number of donors in validation set: ",validation_set_donor_count)
    print("validation set image count", validation_set_image_count)
    print("validation set unique gene count", len(set(validation_set_gene_list)))


    print("\n---- training set ----")
    #print(training_set_info_list)

    training_set_image_count = 0
    training_set_gene_list = []
    training_set_donor_count = len(training_set_donor_list)

    for item in training_set_donor_list:
        this_group_images = group_by_donor.get_group(item)['image_id']
        this_group_genes = group_by_donor.get_group(item)['gene_symbol']
        training_set_image_count += len(set(this_group_images))
        training_set_gene_list.extend(set(this_group_genes))

    print("number of donors in training set: ",training_set_donor_count)
    print("training set image count", training_set_image_count)
    print("training set unique gene count", len(set(training_set_gene_list)))

    print ("\n")


    #----------
    training_df = images_info_df[images_info_df['donor_id'].isin(training_set_donor_list)]
    validation_df = images_info_df[images_info_df['donor_id'].isin(validation_set_donor_list)]
    test_df = images_info_df[images_info_df['donor_id'].isin(test_set_donor_list)]

    training_df = training_df.sort_values(by=['image_id'])
    validation_df = validation_df.sort_values(by=['image_id'])
    test_df = test_df.sort_values(by=['image_id'])

    training_df.to_csv(os.path.join(DATA_DIR, STUDY, "sets", "training.csv"), index=None)
    validation_df.to_csv(os.path.join(DATA_DIR, STUDY, "sets", "validation.csv"), index=None)
    test_df.to_csv(os.path.join(DATA_DIR, STUDY, "sets", "test.csv"), index=None)


    return training_df, validation_df, test_df



def compare_set_genes_list(training_df, validation_df, test_df):
    """
    Compare the 3 sets to see how many genes they have in common.
    :param training_df: pandas dataframe containing the training data
    :param validation_df: pandas dataframe containing the validation data
    :param test_df: pandas dataframe containing the test data
    :return: 4 lists. Each list has the shared genes between different sets:
    genes shared between train and validation
    genes shared between train and test
    genes shared between validation and test
    genes shared between all 3 sets

    """
    train_set_genes = set(training_df['gene_symbol'])
    validation_set_genes = set(validation_df['gene_symbol'])
    test_set_genes = set(test_df['gene_symbol'])

    train_validation_shared_genes_list = list(set(train_set_genes) & set(validation_set_genes))
    train_test_shared_genes_list = list(set(train_set_genes) & set(test_set_genes))
    validation_test_shared_genes_list = list(set(test_set_genes) & set(validation_set_genes))
    all_shared_genes_list = list(set(train_set_genes) & set(validation_set_genes) & set(test_set_genes))

    print("Number of shared genes between train and validation: ", len(train_validation_shared_genes_list))
    print("Number of shared genes between train and test: ", len(train_test_shared_genes_list))
    print("Number of shared genes between validation and test: ", len(validation_test_shared_genes_list))
    print("Number of shared genes between all 3 sets: ", len(all_shared_genes_list))
    print ("\n")


    return train_validation_shared_genes_list, train_test_shared_genes_list, validation_test_shared_genes_list, all_shared_genes_list


def create_new_sets_by_removing_shared_genes(images_info_df, training_df, validation_df, test_df,  train_validation_shared_genes_list,
               train_test_shared_genes_list, validation_test_shared_genes_list, all_shared_genes_list):
    """
    This function gets the set dataframes and the list of genes that are shared between them.
    It then modifies validation set so that it doesn't have any genes shared with test set.
    And modifies training set so that it doesn't have any genes shared with the new validation set and the test set.
    :param images_info_df:  pandas dataframe that has the information of all image
    :param training_df: pandas dataframe containing the training data
    :param validation_df: pandas dataframe containing the validation data
    :param test_df: pandas dataframe containing the test data
    :param train_validation_shared_genes_list: list of genes shared between train and validation
    :param train_test_shared_genes_list: list of genes shared between train and test
    :param validation_test_shared_genes_list: list of genes shared between validation and test
    :param all_shared_genes_list: list of genes shared between all 3 sets
    :return: 3 dataframes: training, validation, test
    """

    print ("Modifying the sets...")

    # -----------

    #print ("---- Handling validation")
    validation_set_genes = set(validation_df['gene_symbol'])
    genes_not_shared_between_val_test = set(validation_set_genes) - set(validation_test_shared_genes_list)
    new_validation_df = validation_df[validation_df['gene_symbol'].isin(genes_not_shared_between_val_test)]


    new_validation_images = set(new_validation_df['image_id'])
    new_validation_genes = set(new_validation_df['gene_symbol'])
    #print ("new_validation_set_image_count: ", len(new_validation_images))
    #print("new_validation_set_gene_count: ",len(new_validation_genes))

    #print ("\n")


    # ----------

    #print ("---- Handling training")
    training_set_genes = set(training_df['gene_symbol'])
    genes_not_shared_between_train_validation_test = set(training_set_genes) - set(train_test_shared_genes_list)  - set(new_validation_genes)
    new_training_df = training_df[training_df['gene_symbol'].isin(genes_not_shared_between_train_validation_test)]

    new_training_genes = set(new_training_df['gene_symbol'])
    new_training_images = set(new_training_df['image_id'])
    #print("new_training_set_image_count: ", len(new_training_images))
    #print("new_training_set_gene_count: ", len(new_training_genes))

    #print("\n")

    return new_training_df, new_validation_df, test_df



def get_stats_on_sets(stats_dict, training_df, validation_df, test_df):
    """
    Calculates some stats on the sets.
    :param images_info_df: pandas dataframe that has the information of all image
    :param training_df: pandas dataframe containing the training data
    :param validation_df: pandas dataframe containing the validation data
    :param test_df: pandas dataframe containing the test data
    :return: None
    """

    original_image_count = stats_dict['image_count']
    original_gene_count = stats_dict['unique_genes_count']

    # ----- training info ------
    training_genes_count = len(set(training_df['gene_symbol']))
    training_images_count = len(set(training_df['image_id']))
    training_donor_count = len(set(training_df['donor_id']))
    print ("\n---- Train ----")
    print ("image count: ", training_images_count)
    print ("gene count: ", training_genes_count)
    print ("donor count: ", training_donor_count)



    # ----- validation info -----
    validation_images_count = len(set(validation_df['image_id']))
    validation_genes_count = len(set(validation_df['gene_symbol']))
    validation_donor_count = len(set(validation_df['donor_id']))
    print("\n---- Validation ----")
    print("image count: ", validation_images_count)
    print("gene count: ", validation_genes_count)
    print("donor count: ", validation_donor_count)

    # ----- test info ------
    test_images_count = len(set(test_df['image_id']))
    test_genes_count = len(set(test_df['gene_symbol']))
    test_donor_count = len(set(test_df['donor_id']))
    print("\n---- Test ----")
    print("image count: ", test_images_count)
    print("gene count: ", test_genes_count)
    print("donor count: ", test_donor_count)


    current_image_count = training_images_count + validation_images_count + test_images_count
    current_gene_count = training_genes_count + validation_genes_count + test_genes_count

    print ("original image count: ", original_image_count)
    print ("original gene count: ", original_gene_count)
    print("\n")
    print("current image count: ", current_image_count)
    print("current gene count: ", current_gene_count)
    print("\n")
    print (original_image_count - current_image_count , " images thrown away")
    print (original_gene_count - current_gene_count, " genes thrown away")
    print ("\n")

    print ("Train image percentage: ", (training_images_count/current_image_count)*100)
    print("Validation image percentage: ", (validation_images_count / current_image_count) * 100)
    print("Test image percentage: ", (test_images_count / current_image_count) * 100)



def donor_info(my_set):

    group_by_donor = my_set.groupby('donor_id')

    unique_gene_count_per_donor_list = []
    unique_image_count_per_donor_list = []
    for key, item in group_by_donor:
        this_group_genes = group_by_donor.get_group(key)['gene_symbol']
        this_group_images = group_by_donor.get_group(key)['image_id']
        unique_gene_count_per_donor_list.append((key,len(set(this_group_genes))))
        unique_image_count_per_donor_list.append((key,len(set(this_group_images))))


    print("\ngene count per donor: ")
    print (unique_gene_count_per_donor_list)
    print ("\nimage count per donor: ")
    print (unique_image_count_per_donor_list)


def make_triplet_csv_no_segmentation(df, out_file):
    """
    Use this function to create input suited for the triplet-reid training scripts
    """

    temp_df = df.assign(image=lambda df: df.image_id.apply(lambda row: "{}.jpg".format(row)))[['gene_symbol', 'image']]
    new_image_info= []


    total_number_of_circles = NUMBER_OF_CIRCLES_IN_HEIGHT * NUMBER_OF_CIRCLES_IN_WIDTH
    for patch_index in range(1, total_number_of_circles+1):
        patch_image_list = [(id.split(".")[0]+"_"+str(patch_index)+".jpg",gene) for id, gene in zip(temp_df['image'],temp_df['gene_symbol'])]
        new_image_info += patch_image_list
    new_df = pd.DataFrame(columns=['gene_symbol','image'])
    new_df['image'] = [item[0] for item in new_image_info]
    new_df['gene_symbol'] = [item[1] for item in new_image_info]

    new_df = new_df.sort_values(by=['image'])

    return (new_df.to_csv(out_file, index=False, header=False))


def make_triplet_csv_with_segmentation(df, out_file):


    csv_file_name = "less_than_" + str(PATCH_COUNT_PER_IMAGE) + ".csv"
    not_enough_patches_df = pd.read_csv(os.path.join(DATA_DIR, STUDY, "segmentation_data", "outlier_images", csv_file_name))

    not_enough_patches_dict = dict(zip(not_enough_patches_df["image_id"], not_enough_patches_df["count"]))


    temp_df = df.assign(image=lambda df: df.image_id.apply(lambda row: "{}.jpg".format(row)))[['gene_symbol', 'image']]
    new_image_info = []

    for id, gene in zip(temp_df['image'],temp_df['gene_symbol']):

        id_temp = int(id.split(".")[0])
        if id_temp in not_enough_patches_dict:

            count = not_enough_patches_dict[id_temp]
            for patch_index in range(0, count):
                patch_image_list = [(id.split(".")[0] + "_" + str(patch_index) + ".jpg", gene)]
                new_image_info += patch_image_list


        else:

            for patch_index in range(0, PATCH_COUNT_PER_IMAGE):
                patch_image_list = [(id.split(".")[0] + "_" + str(patch_index) + ".jpg", gene)]
                new_image_info += patch_image_list

    new_df = pd.DataFrame(columns=['gene_symbol', 'image'])
    new_df['image'] = [item[0] for item in new_image_info]
    new_df['gene_symbol'] = [item[1] for item in new_image_info]

    new_df = new_df.sort_values(by=['image'])

    return (new_df.to_csv(out_file, index=False, header=False))





def make_triplet_csvs(dfs):

    out_base = os.path.join(DATA_DIR, STUDY, "sets") + "/triplet"

    if PATCH_TYPE=="segmentation":
        return tuple((make_triplet_csv_with_segmentation(df, "{}_{}.csv".format(out_base, ext)) and "{}_{}.csv".format(
            out_base, ext))
                     for df, ext in zip(dfs, ("training", "validation", "test", "training_validation")))

    else:
        return tuple((make_triplet_csv_no_segmentation(df, "{}_{}.csv".format(out_base,ext)) and "{}_{}.csv".format(out_base, ext))
                     for df, ext in zip(dfs, ("training", "validation", "test", "training_validation")))



def convert_h5_to_csv():

    exp_root_contents = os.listdir(EXPERIMENT_ROOT)
    for item in exp_root_contents:
        if item.endswith(".h5"):
            embedding_csv_name = item.split(".")[0] + ".csv"
            set_csv_file_name = embedding_csv_name.replace("_embeddings", "")
            print ("set csv file name is: ", set_csv_file_name)
            print ("item is: ", item)

            set_csv_file = os.path.join(DATA_DIR, STUDY, "sets", set_csv_file_name)
            df = pd.read_csv(set_csv_file, names=['gene', 'image_id'])
            f = h5py.File(os.path.join(EXPERIMENT_ROOT, item), 'r')['emb']
            df['image_id']= df.apply(lambda x: x['image_id'].split('.')[0], axis =  1)
            pd.DataFrame(np.array(f), index=df.image_id).to_csv(os.path.join(EXPERIMENT_ROOT, embedding_csv_name))




def save_embedding_info_into_file(filename):


    if (not os.path.exists(EMBEDDING_DEST)):
        os.mkdir(EMBEDDING_DEST)

    os.mkdir(os.path.join(EMBEDDING_DEST, filename))
    embed_info_dir = os.path.join(EMBEDDING_DEST, filename)

    exp_root_contents = os.listdir(EXPERIMENT_ROOT)
    for item in exp_root_contents:
        if item.endswith(".csv"):
            copyfile(os.path.join(EXPERIMENT_ROOT, item), os.path.join(embed_info_dir, item))
        elif item.endswith(".json"):
            copyfile(os.path.join(EXPERIMENT_ROOT, item), os.path.join(embed_info_dir, item))
        elif item.endswith(".log"):
            copyfile(os.path.join(EXPERIMENT_ROOT, item), os.path.join(embed_info_dir, item))
        elif item.startswith("events."):
            copyfile(os.path.join(EXPERIMENT_ROOT, item), os.path.join(embed_info_dir, item))

    return filename



def merge_embeddings_to_gene_level(filename):
    """
    We have an embedding for every image in the dataset. However, each gene may have more than one image associated to it.
    This function will take all the images that correspond to an image, and average over the values of the embedding vector to generate a final embedding for that gene.
    """

    embed_file_contents = os.listdir(os.path.join(EMBEDDING_DEST, filename))
    for item in embed_file_contents:
        if item.endswith(".csv"):
            embeddings_file = pd.read_csv(os.path.join(EMBEDDING_DEST, filename, item))
            patches_info = pd.read_csv(os.path.join(IMAGE_ROOT, "valid_patches_info.csv"))
            
            embeddings_file = embeddings_file.rename(columns={'image_id': 'patch_id'})
            # perform left merge on the two dataframes to add gene_symbol to the embeddings.csv
            merged_df = embeddings_file.merge(patches_info[["patch_id", "gene_symbol"]], how = "left" , on = "patch_id")

            # reorder the dataframe columns
            merged_columns = list(merged_df)
            merged_columns = [merged_columns[0]] + [merged_columns [-1]] + merged_columns[1:-1]
            merged_df = merged_df[merged_columns]

            # drop the patch_id column
            merged_df = merged_df.drop(columns=["patch_id"])

            # group by gene_symbol and average over the embedding values
            grouped_df = merged_df.groupby(['gene_symbol']).mean()

            print (grouped_df.head())

            print ("the number of genes is: {}".format(len(grouped_df)))

            # and then I want to save this file as gene_embddings in the same folder.
            item_name = item.split(".")[0]
            save_to_path = os.path.join(EMBEDDING_DEST, filename, item_name+"_gene_level.csv")
            grouped_df.to_csv(save_to_path)


def filter_out_common_genes(df_file_name,threshold = 3):

    df = pd.read_csv(os.path.join(DATA_DIR, STUDY, "sets", df_file_name))
    print(len(df))

    genes = df.iloc[:, 0]
    unique_gene_count_dict = {}

    genes_unique, counts = np.unique(genes, return_counts=True)

    for i in range(len(genes_unique)):
        unique_gene_count_dict[genes_unique[i]] = counts[i]

    sorted_dict = sorted(unique_gene_count_dict.items(), key=operator.itemgetter(1))

    print (sorted_dict)

    most_common = []
    for i in range(threshold):
        most_common.append(sorted_dict[-1-i][0])

    # ----------

    new_df = df[~df.iloc[:,0].isin(most_common)]
    print(len(new_df))


    genes = new_df.iloc[:, 0]
    unique_gene_count_dict = {}

    genes_unique, counts = np.unique(genes, return_counts=True)

    for i in range(len(genes_unique)):
        unique_gene_count_dict[genes_unique[i]] = counts[i]

    sorted_dict = sorted(unique_gene_count_dict.items(), key=operator.itemgetter(1))

    print(sorted_dict)

    new_df_file_name = df_file_name.split(".")[0] + "_filtered.csv"
    new_df.to_csv(os.path.join(DATA_DIR, STUDY, "sets", new_df_file_name), index=None)


def filter_out_genes_out_of_mean_and_std(df_file_name):

    in_range = []
    df = pd.read_csv(os.path.join(DATA_DIR, STUDY, "sets", df_file_name))
    print(len(df))

    genes = df.iloc[:, 0]
    unique_gene_count_dict = {}

    genes_unique, counts = np.unique(genes, return_counts=True)

    for i in range(len(genes_unique)):
        unique_gene_count_dict[genes_unique[i]] = counts[i]

    sorted_dict = sorted(unique_gene_count_dict.items(), key=operator.itemgetter(1))

    print (sorted_dict)
    ordered_genes = [item[0] for item in sorted_dict]
    ordered_unique_gene_count = [item[1] for item in sorted_dict]

    avg =np.mean(ordered_unique_gene_count)
    sd = np.std(ordered_unique_gene_count)

    max_lim = int(avg) + int(sd)
    min_lim = int(avg) - int(sd)

    print ("avg is: ", avg)
    print ("sd is: ", sd)
    print ("max lim is: ", max_lim)
    print ("min lim is: ",min_lim)

    num_of_out_of_range_genes = 0
    num_of_out_of_range_images = 0
    for item in sorted_dict:
        if item[1]> min_lim and item[1] < max_lim:
            in_range.append(item[0])
        else:
            num_of_out_of_range_genes +=1
            num_of_out_of_range_images += item[1]

    print ("num of out of range genes: ", num_of_out_of_range_genes)
    print ("num of out of range images: ", num_of_out_of_range_images)


    # ----------

    new_df = df[df.iloc[:, 0].isin(in_range)]
    print(len(new_df))

    genes = new_df.iloc[:, 0]
    unique_gene_count_dict = {}

    genes_unique, counts = np.unique(genes, return_counts=True)

    for i in range(len(genes_unique)):
        unique_gene_count_dict[genes_unique[i]] = counts[i]

    sorted_dict = sorted(unique_gene_count_dict.items(), key=operator.itemgetter(1))

    print(sorted_dict)

    new_df_file_name = df_file_name.split(".")[0] + "_in_range.csv"
    new_df.to_csv(os.path.join(DATA_DIR, STUDY, "sets", new_df_file_name), index=None)





def draw_hist(df_file_name):
    df = pd.read_csv(os.path.join(DATA_DIR, STUDY, "sets", df_file_name))
    print(len(df))

    genes = df.iloc[:, 0]
    unique_gene_count_dict = {}

    genes_unique, counts = np.unique(genes, return_counts=True)

    for i in range(len(genes_unique)):
        unique_gene_count_dict[genes_unique[i]] = counts[i]

    sorted_dict = sorted(unique_gene_count_dict.items(), key=operator.itemgetter(1))

    ordered_genes = [item[0] for item in sorted_dict]
    ordered_unique_gene_count = [item[1] for item in sorted_dict]

    print(ordered_genes)
    print(ordered_unique_gene_count)
    print(np.mean(ordered_unique_gene_count))
    print(np.std(ordered_unique_gene_count))

    plt.hist(ordered_unique_gene_count, normed=False, bins=100)
    plt.ylabel('unique gene count')
    plt.show()

def images_wiht_no_valid_patches():
    path_to_outliers = os.path.join(DATA_DIR,STUDY,"segmentation_data","outlier_images")
    content = os.listdir(path_to_outliers)
    no_valid_patch_list = []

    for item in content:
        if item.endswith(".jpg"):
            no_valid_patch_list.append(item.split(".")[0])
    return no_valid_patch_list

def make_sets():

    images_info_df = pd.read_csv(os.path.join(DATA_DIR, STUDY, "human_ISH_info.csv"))
    no_valid_patch_list = images_wiht_no_valid_patches()
    images_info_df = images_info_df[~images_info_df["image_id"].isin(no_valid_patch_list)]

    stats_dict = get_stats(images_info_df)

    training_df, validation_df, test_df, train_val_df = define_sets_with_no_shared_genes(images_info_df)
    get_stats_on_sets(stats_dict, training_df, validation_df, test_df)

    make_triplet_csvs((training_df, validation_df, test_df, train_val_df))

    filter_out_common_genes("triplet_training.csv")
    filter_out_genes_out_of_mean_and_std("triplet_training.csv")

    """
    training_df, validation_df, test_df = define_sets_with_no_shared_donors(images_info_df)

    train_validation_shared_genes_list, train_test_shared_genes_list, validation_test_shared_genes_list, all_shared_genes_list = \
        compare_set_genes_list(training_df, validation_df, test_df)

    new_training_df, new_validation_df, test_df = create_new_sets_by_removing_shared_genes(images_info_df, training_df, validation_df, test_df,
                                             train_validation_shared_genes_list, train_test_shared_genes_list,
                                             validation_test_shared_genes_list, all_shared_genes_list)

    get_stats_on_sets(stats_dict, new_training_df, new_validation_df, test_df)
    """


def run():
    pass

if __name__ == '__main__':

    run()
    









