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
random.seed(1)






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



    # -------------------
    # I want to group by donor, in each donor, see on average, how many images there are per gene
    # and then average over all the donors
    group_by_donor = images_info_df.groupby('donor_id')
    avg_num_of_imaes_per_gene_list = []
    for key, item in group_by_donor:
        # for each donor
        this_group_genes = list(group_by_donor.get_group(key)['gene_symbol'])  # get a list of its genes (has duplicates)

        # for each unique genes, see how many times it appears in the list (== how many images we have of it in this donor)
        this_group_genes_count_list = [[x,this_group_genes.count(x)] for x in set(this_group_genes)]
        sum = 0
        for item in this_group_genes_count_list:
            sum += item[1]

        # in this donor, on average, we have 'avg' number of images per each gene.
        avg = sum / len(this_group_genes_count_list)

        # append it to the list
        avg_num_of_imaes_per_gene_list.append(avg)

    avg_num_of_images_per_gene_in_each_donor_over_all = np.mean(avg_num_of_imaes_per_gene_list)
    print ("Average number of images per each gene in each donor, Over all donors: ",avg_num_of_images_per_gene_in_each_donor_over_all)


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


    sets_path = os.path.join(DATA_DIR, STUDY, "sets_"+str(PATCH_COUNT_PER_IMAGE)+"_patches_"+str(SEGMENTATION_TRAINING_SAMPLES)+"_seg")
    if (not os.path.exists(sets_path)):
        os.mkdir(sets_path)

    training_df.to_csv(os.path.join(sets_path, "training.csv"), index=None)
    validation_df.to_csv(os.path.join(sets_path, "validation.csv"), index=None)
    test_df.to_csv(os.path.join(sets_path, "test.csv"), index=None)
    train_val_df.to_csv(os.path.join(sets_path, "training_validation.csv"), index=None)

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

    sets_path = os.path.join(DATA_DIR, STUDY, "sets_" + str(PATCH_COUNT_PER_IMAGE) + "_patches_"+str(SEGMENTATION_TRAINING_SAMPLES)+"_seg")
    if (not os.path.exists(sets_path)):
        os.mkdir(sets_path)


    training_df.to_csv(os.path.join(sets_path, "training.csv"), index=None)
    validation_df.to_csv(os.path.join(sets_path, "validation.csv"), index=None)
    test_df.to_csv(os.path.join(sets_path, "test.csv"), index=None)


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
    not_enough_patches_df = pd.read_csv(os.path.join(DATA_DIR, STUDY, "segmentation_data","trained_on_"+str(SEGMENTATION_TRAINING_SAMPLES) ,"outlier_images", csv_file_name))

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

    sets_path = os.path.join(DATA_DIR, STUDY, "sets_" + str(PATCH_COUNT_PER_IMAGE) + "_patches_"+str(SEGMENTATION_TRAINING_SAMPLES)+"_seg")
    out_base = sets_path + "/triplet"

    if PATCH_TYPE=="segmentation":
        return tuple((make_triplet_csv_with_segmentation(df, "{}_{}.csv".format(out_base, ext)) and "{}_{}.csv".format(
            out_base, ext))
                     for df, ext in zip(dfs, ("training", "validation", "test", "training_validation")))

    else:
        return tuple((make_triplet_csv_no_segmentation(df, "{}_{}.csv".format(out_base,ext)) and "{}_{}.csv".format(out_base, ext))
                     for df, ext in zip(dfs, ("training", "validation", "test", "training_validation")))



def convert_h5_to_csv():

    sets_path = os.path.join(DATA_DIR, STUDY, "sets_" + str(PATCH_COUNT_PER_IMAGE) + "_patches_"+str(SEGMENTATION_TRAINING_SAMPLES)+"_seg")

    exp_root_contents = os.listdir(EXPERIMENT_ROOT)
    for item in exp_root_contents:
        if item.endswith(".h5"):
            embedding_csv_name = item.split(".")[0] + ".csv"
            set_csv_file_name = embedding_csv_name.replace("_embeddings", "")
            print ("set csv file name is: ", set_csv_file_name)
            print ("item is: ", item)

            set_csv_file = os.path.join(sets_path, set_csv_file_name)
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
    We have an embedding for every patch in the dataset. However, each gene may have more than one image associated to it.
    This function will take all the images that correspond to an image, and average over the values of the embedding vector to generate a final embedding for that gene.
    """

    embed_file_contents = os.listdir(os.path.join(EMBEDDING_DEST, filename))
    for item in embed_file_contents:
        if item.endswith("embeddings.csv"):
            #if item.endswith("_gene_level.csv") or item.endswith("_image_level.csv"):
                #pass
            #else:

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





def merge_embeddings_to_image_level(filename):
    """
        We have an embedding for every patch in the dataset.
        This function will take all the patches that correspond to an image, and average over the values of the
        embedding vector to generate a final embedding for that image.
        """
    
    embed_file_contents = os.listdir(os.path.join(EMBEDDING_DEST, filename))
    for item in embed_file_contents:
        if item.endswith("embeddings.csv"):
            #if item.endswith("_gene_level.csv") or item.endswith("_image_level.csv"):
                #pass
            #else:
            print ("staaaaaaart: ", item)
            embeddings_file = pd.read_csv(os.path.join(EMBEDDING_DEST, filename, item))
            patches_info = pd.read_csv(os.path.join(IMAGE_ROOT, "valid_patches_info.csv"))

            print (embeddings_file.head())
            print ("---")
            print (patches_info.head())
            im_id_list = patches_info['image_id']
            im_id_ex = im_id_list[10]
            print (im_id_ex)
            print (type(im_id_ex))

            if filename == "random":
                embeddings_file = embeddings_file.rename(columns={'id': 'patch_id'})
            else:
                embeddings_file = embeddings_file.rename(columns={'image_id': 'patch_id'})


            p_id_list =embeddings_file['patch_id']
            p_id_ex = p_id_list[10]
            print (p_id_ex)
            print (type(p_id_ex))
            print ("~~~~~")
            # perform left merge on the two dataframes to add gene_symbol to the embeddings.csv
            merged_df = embeddings_file.merge(patches_info[["patch_id", "image_id"]], how="left", on="patch_id")

            print ("_---")
            print (merged_df.head())
            # reorder the dataframe columns
            merged_columns = list(merged_df)
            merged_columns = [merged_columns[0]] + [merged_columns[-1]] + merged_columns[1:-1]
            merged_df = merged_df[merged_columns]

            print (merged_df.head())
            print ("///")
            im_id_list = merged_df['image_id']
            im_id_ex = im_id_list[10]
            print (im_id_ex)
            print (type(im_id_ex))
            # drop the patch_id column
            merged_df = merged_df.drop(columns=["patch_id"])
            merged_df = merged_df.astype({'image_id': 'int'})
            print ("_____")
            print (merged_df.head())

            # group by gene_symbol and average over the embedding values
            grouped_df = merged_df.groupby(['image_id']).mean()

            print ("[[[[")
            print(grouped_df.head())

            print("the number of images is: {}".format(len(grouped_df)))

            # and then I want to save this file as gene_embddings in the same folder.
            item_name = item.split(".")[0]
            save_to_path = os.path.join(EMBEDDING_DEST, filename, item_name + "_image_level.csv")
            grouped_df.to_csv(save_to_path)


def filter_out_common_genes(df_file_name,threshold = 3):
    sets_path = os.path.join(DATA_DIR, STUDY, "sets_" + str(PATCH_COUNT_PER_IMAGE) + "_patches_"+str(SEGMENTATION_TRAINING_SAMPLES)+"_seg")

    df = pd.read_csv(os.path.join(sets_path, df_file_name))
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
    new_df.to_csv(os.path.join(sets_path, new_df_file_name), index=None)


def filter_out_genes_out_of_mean_and_std(df_file_name):

    sets_path = os.path.join(DATA_DIR, STUDY, "sets_" + str(PATCH_COUNT_PER_IMAGE) + "_patches_"+str(SEGMENTATION_TRAINING_SAMPLES)+"_seg")

    in_range = []
    df = pd.read_csv(os.path.join(sets_path, df_file_name))
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
    new_df.to_csv(os.path.join(sets_path, new_df_file_name), index=None)





def draw_hist(df_file_name):
    sets_path = os.path.join(DATA_DIR, STUDY, "sets_" + str(PATCH_COUNT_PER_IMAGE) + "_patches_"+str(SEGMENTATION_TRAINING_SAMPLES)+"_seg")
    df = pd.read_csv(os.path.join(sets_path, df_file_name))
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
    path_to_outliers = os.path.join(DATA_DIR,STUDY,"segmentation_data","trained_on_"+str(SEGMENTATION_TRAINING_SAMPLES),"outlier_images")
    less_than_thresh_df = pd.read_csv(os.path.join(path_to_outliers, "less_than_" + str(PATCH_COUNT_PER_IMAGE) + ".csv"))
    no_valid_patch_list = list(less_than_thresh_df[less_than_thresh_df["count"] == 0]["image_id"])
    no_valid_patch_list = [str(item) for item in no_valid_patch_list]
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


def generate_random_embeddings(info_csv_file, embeddings_length):
    """
    this function generates random embeddings for the images. The result will be a csv files that has the embedding vector of every image.

    :param embeddings_length: the length of the embedding vector which also determines the number of columns in the final csv file.
    :param range_min: the minimum possible value in the embeddings.
    :param range_max: the maximum possible value in the embeddings.

    :return: None
    """


    path_to_info_csv = os.path.join(DATA_DIR,STUDY, "sets_20_patches_20_seg/validation.csv")
    #path_to_info_csv = os.path.join(IMAGE_ROOT,info_csv_file)
    info_csv = pd.read_csv(path_to_info_csv,)

    columns = list(info_csv)
    id_column = info_csv[columns[0]]

    n_images = len(info_csv)

    cols = np.arange(0, embeddings_length)
    cols = list(map(str, cols))
    cols = ['id'] + cols

    random_embed_file = pd.DataFrame(columns=cols)
    random_embed_file['id'] = id_column

    for i in range(embeddings_length):
        sample = np.random.uniform(size=(n_images,))
        random_embed_file[str(i)] = sample


    path_to_random = os.path.join(EMBEDDING_DEST, "random")
    if (not os.path.exists(path_to_random)):
        os.mkdir(path_to_random)

    random_embed_file.to_csv(os.path.join(path_to_random, "random_validation_embeddings_image_level.csv"),index=None)

    print ("finished generating random embeddings...")


def get_embeddings_from_pre_trained_model(model_name="resnet50", trained_on="imagenet", dim=128, standardize=False,
                                          chunk_range=None, chunk_ID=None):
    """
    Generates embeddings from a pre-trained model without further training.
    The function uses 'valid_patches_info.csv' to take the list of images to perform on.
    But, if the number of images is too high, there might be OOM errors. In that case, the function will perform on chunks
    of images at each time. There is another function,  'get_embeddings_from_pre_trained_model_in_chunks', which will be
    called first and will generate the chunk. That function will then call this function.

    :param model_name: string, the pre-trained model to be used
    :param trained_on: string, the dataset on which the pre-trained model has been trained
    :param dim: int, the dimensionality of the output embeddings
    :param standardize: bool, flag to linearly scale each image to have mean 0 and variance 1
    :param chunk_range: tuple, if the function is supposed to perform on chunks, this tuple indicates the start and end index
    :param chunk_ID: int, if the function is supposed to perform on chunks, this value indicates the chunk ID
    :return: None. It creates the embeddings and stores them in a csv file.
    """

    print("Generating embeddings from a plain ", model_name)

    # ---- imports libraries ---------
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras import Model
    from tensorflow.keras.preprocessing import image
    # --------------------------------

    if standardize:

        embed_folder_name = model_name + "_" + str(PATCH_COUNT_PER_IMAGE) + "_patches_standardized_2"
        if chunk_ID:
            embeddings_csv_file_name = model_name + "_standardized_embeddings_" + str(chunk_ID) +".csv"
        else:
            embeddings_csv_file_name = model_name + "_standardized_embeddings.csv"

    else:
        embed_folder_name = model_name + "_" + str(PATCH_COUNT_PER_IMAGE) + "_patches"
        if chunk_ID:
            embeddings_csv_file_name = model_name + "_embeddings_" + str(chunk_ID)+".csv"
        else:
            embeddings_csv_file_name = model_name + "_embeddings.csv"


    if (not os.path.exists(os.path.join(EMBEDDING_DEST, embed_folder_name))):
        os.mkdir(os.path.join(EMBEDDING_DEST, embed_folder_name))


    valid_patches_info_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
    valid_patches_info = pd.read_csv(valid_patches_info_path)
    patch_id_list = valid_patches_info['patch_id']

    image_dir = IMAGE_ROOT
    print("image dir is: ", image_dir)

    height = PATCH_HEIGHT
    width = PATCH_WIDTH

    image_list = [item + ".jpg" for item in patch_id_list]
    image_list.sort()

    if chunk_range:
        chunk_start = chunk_range[0]
        chunk_end = chunk_range[1]
        image_list = image_list[chunk_start:chunk_end]

    
    loaded_images = []
    print("started loading images ...")
    for i in range(len(image_list)):
        print (i , " loading", " chunk_ID: ", chunk_ID)
        image_to_embed = image_list[i]
        image_id = image_to_embed.split(".")[0]

        img_path = os.path.join(image_dir, image_to_embed)
        img = image.load_img(img_path, target_size=(height, width))
        loaded_images.append(img)

    print("finished loading images ...\n")

    if standardize:
        print("started standardizing images ...")

        for i in range(len(loaded_images)):
            print(i, " standardizing", " chunk_ID: ", chunk_ID)
            img = loaded_images[i] 
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = tf.image.per_image_standardization(img)
            img_data = tf.keras.backend.eval(img)
            tf.keras.backend.clear_session()
            loaded_images[i] = img_data

        print("finished standardizing images ...")

    embeddings_list = []
    if model_name == "resnet50":
        from tensorflow.keras.applications.resnet50 import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input
        pre_trained_model = ResNet50(input_shape=(height, width, 3),
                                     include_top=False,
                                     pooling=max,
                                     weights=trained_on)

        # freeze all the layers
        for layer in pre_trained_model.layers:
            layer.trainable = False

        # print (pre_trained_model.summary())

        last_layer = pre_trained_model.get_layer(index=-1)
        print("last layer output shape is: ", last_layer.output_shape)
        last_output = last_layer.output

        # x = layers.AveragePooling2D((8,8))(last_output)
        # x = layers.Flatten()(x)

        x = layers.Flatten()(last_output)
        x = layers.Dense(dim, activation='relu')(x)

        model = Model(pre_trained_model.input, x)
        # model = Model(pre_trained_model.input, last_output)

        print("total number of images: ", len(image_list))

        print("\n started passing images through the model ...")
        for i in range(len(loaded_images)):
            print(i, " passing through", " chunk_ID: ", chunk_ID)
            image_id = image_list[i].split(".")[0]
            img = loaded_images[i]
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            # img_data = np.vstack([x])
            resnet50_feature = model.predict(img_data)
            resnet50_feature = resnet50_feature.squeeze().tolist()
            resnet50_feature = [image_id] + resnet50_feature

            embeddings_list.append(resnet50_feature)

        tf.keras.backend.clear_session()

        column_names = np.arange(0, dim)
        column_names = [str(name) for name in column_names]
        column_names = ['image_id'] + column_names

        embedding_df = pd.DataFrame(embeddings_list, columns=column_names)

        embeddings_path = os.path.join(EMBEDDING_DEST, embed_folder_name, embeddings_csv_file_name)
        embedding_df.to_csv(embeddings_path, index=None)
    


def  get_embeddings_from_pre_trained_model_in_chunks(number_of_chunks=10, model_name="resnet50", trained_on="imagenet", dim=128, standardize=True,):

    """
    This function is used when the number of images is too high for the system to handle them all at once and it could cause
    OOM problems.
    This function will group them in chunks and call the 'get_embeddings_from_pre_trained_model' function on each chunk.

    :param number_of_chunks: int
    :param model_name:  string, the pre-trained model to be used
    :param trained_on: string, the dataset on which the pre-trained model has been trained
    :param dim: int, the dimensionality of the output embeddings
    :param standardize: bool, flag to linearly scale each image to have mean 0 and variance 1
    :return: None
    """

    valid_patches_info_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
    valid_patches_info = pd.read_csv(valid_patches_info_path)
    patch_id_list = valid_patches_info['patch_id']

    image_dir = IMAGE_ROOT
    print("image dir is: ", image_dir)

    image_list = [item + ".jpg" for item in patch_id_list]
    image_list.sort()
    number_of_images = len(image_list)
    print ("there are {} images in this directory".format(number_of_images))


    # minimum number of images in each chunk. There might be more in the last chunk.
    min_in_each_chunk = number_of_images // number_of_chunks

    start = 0
    for i in range(1, number_of_chunks):
        this_chunk_start_ind = start
        this_chunk_end_ind = start+min_in_each_chunk

        print ("this chunk start and end indices are {} , {}".format(this_chunk_start_ind, this_chunk_end_ind))
        chunk_ID = i
        print ("chunk ID: {}".format(chunk_ID))
        start = this_chunk_end_ind

        print ('calling get embed function ...')
        get_embeddings_from_pre_trained_model(model_name=model_name, trained_on=trained_on, dim=dim, standardize=standardize,
                                              chunk_range=(this_chunk_start_ind, this_chunk_end_ind), chunk_ID=chunk_ID)



    # handling the last chunk
    this_chunk_start_ind = start
    this_chunk_end_ind= number_of_images
    print ("last chunk's start and end indices are {} , {}".format(this_chunk_start_ind, this_chunk_end_ind))
    chunk_ID = number_of_chunks
    print("chunk ID: {}".format(chunk_ID))
    print('calling get embed function ...')
    get_embeddings_from_pre_trained_model(model_name="resnet50", trained_on="imagenet", dim=128,
                                          standardize=standardize,
                                          chunk_range=(this_chunk_start_ind, this_chunk_end_ind), chunk_ID=chunk_ID)



def concatenate_embedding_chunks(embed_folder_name, number_of_chunks =10):
    

    print ("started concatenating embedding chunks ...")
    embed_folder_path = os.path.join(EMBEDDING_DEST, embed_folder_name)

    embed_folder_content = os.listdir(embed_folder_path)
    general_csv_name = ""
    embed_csv_files = []
    for i in range(1, number_of_chunks+1):
        embed_csv_name = ""
        for item in embed_folder_content:
            if item.endswith("_"+str(i)+".csv"):
                general_csv_name = item.split("_"+str(i)+".csv")[0]
                embed_csv_name = item
                break

        print ("embedding csv file name: {}".format(embed_csv_name))
        embed_csv_file = pd.read_csv(os.path.join(embed_folder_path, embed_csv_name))

        embed_csv_files.append(embed_csv_file)

    print ("finished reading all the embedding files ... ")

    print ("general csv name: {}".format(general_csv_name))

    general_csv_name = general_csv_name +".csv"

    final_embed_csv = pd.concat(embed_csv_files, ignore_index=True)
    final_embed_csv.to_csv(os.path.join(embed_folder_path, general_csv_name),index=None)


    # ------
    check_concatenated_embeddings(embed_folder_name, general_csv_name)



def check_each_chunk(embed_folder_name, first_file_index, second_file_index, number_of_chunks =10):
    embed_folder_path = os.path.join(EMBEDDING_DEST, embed_folder_name)
    embed_folder_content = os.listdir(embed_folder_path)




def check_concatenated_embeddings(embed_folder_name, general_csv_name, number_of_chunks =10):



    valid_patches_info_path = os.path.join(IMAGE_ROOT, "valid_patches_info.csv")
    valid_patches_info = pd.read_csv(valid_patches_info_path)
    patch_id_list = valid_patches_info['patch_id']  # the type is actually pandas series. Not python list

    concat_embed_path = os.path.join(EMBEDDING_DEST, embed_folder_name, general_csv_name)
    concat_embed_file = pd.read_csv(concat_embed_path)
    image_id_list = concat_embed_file['image_id']  # the type is actually pandas series. Not python list


    print ("patch count in valid patch info: {} ".format(len(patch_id_list)))
    print ("patch count in concatenated embed file: {}".format(len(image_id_list)))


    dif_count = 0

    # I have to use .values because it is pandas series. Otherwise it will check the indices and return False.
    for item in patch_id_list[:10].values:
        if item not in image_id_list[:10].values:
            dif_count +=1
   
    print ("difference is: {} ".format(dif_count))
  
def run():
    #pass
    embed_file_name = "triplet_training_validation_embeddings.csv"
    embed_dir = os.path.join(DATA_DIR, STUDY, "segmentation_embeddings")
    #ts_list = os.listdir(embed_dir)
    ts_list =["1584025762"]
    for ts in ts_list:
        print ("ts:", ts)
        filename = os.path.join(embed_dir, ts, embed_file_name)
        merge_embeddings_to_gene_level(ts)
        merge_embeddings_to_image_level(ts)


def specific_donor_embeddings(donor_id, embed_folder_name):


    images_info_df = pd.read_csv(os.path.join(DATA_DIR, STUDY, "human_ISH_info.csv"))
    this_donor = images_info_df[images_info_df['donor_id']==donor_id]
    this_donor_image_id_gene = this_donor[['image_id', 'gene_symbol']]

    embed_dir = os.path.join(EMBEDDING_DEST, embed_folder_name)
    
    embed_file_name  = ""
    for item in os.listdir(embed_dir):
        if item.endswith("embeddings_image_level.csv"):
            embed_file_name = item
    
    embed_df = pd.read_csv(os.path.join(EMBEDDING_DEST, embed_folder_name, embed_file_name))



    merged_df = pd.merge(this_donor_image_id_gene, embed_df, on='image_id')
    merged_df = merged_df.drop(columns=['image_id'])
    
    merged_df_no_meta = merged_df.drop(columns=['gene_symbol'])

    merged_df.to_csv(os.path.join(EMBEDDING_DEST, embed_folder_name,donor_id+".csv"), index=None)
    convert_to_tsv(os.path.join(EMBEDDING_DEST, embed_folder_name,donor_id+".csv"))

    merged_df_no_meta.to_csv(os.path.join(EMBEDDING_DEST, embed_folder_name, donor_id+"_no_meta.csv"), header=False, index=None)
    convert_to_tsv(os.path.join(EMBEDDING_DEST, embed_folder_name, donor_id+"_no_meta.csv"))



def convert_to_tsv(path_to_csv):


    # With meta

    path_to_tsv = path_to_csv.split(".")[0] + ".tsv"
    csv_read = pd.read_csv(path_to_csv)

    with open(path_to_tsv, 'w') as write_tsv:
        write_tsv.write(csv_read.to_csv(sep='\t', index=False))


    # Without meta data

    path_to_tsv = path_to_csv.split(".")[0] + "_no_meta.tsv"
    csv_read = pd.read_csv(path_to_csv)

    csv_read = csv_read.drop(columns=['gene_symbol', 'Cortical.marker..human.', 'Expression.level'])

    with open(path_to_tsv, 'w') as write_tsv:
        write_tsv.write(csv_read.to_csv(sep='\t', index=False, header=False))




def get_image_level_embeddings_of_a_target_set(path_to_sets, ts, target_sets=["training", "validation"]):

    embeddings_path = os.path.join(EMBEDDING_DEST, ts)
    contents = os.listdir(embeddings_path)

    image_level_embeddings_file_name = ""

    for item in contents:
        if item.endswith("training_validation_embeddings_image_level.csv"):
            image_level_embeddings_file_name = item
            break

    image_level_embeddings = pd.read_csv(os.path.join(embeddings_path, image_level_embeddings_file_name))

    for target in target_sets:

        print ("Getting embeddings of the " + target + " set...")
        path_to_target_set = os.path.join(path_to_sets, target +".csv")
        target_df = pd.read_csv(path_to_target_set)
        target_image_id = list(target_df['image_id'])

        target_embeddings = image_level_embeddings[image_level_embeddings['image_id'].isin(target_image_id)]

        target_embeddings_file_name = target +"_embeddings_image_level.csv"
        target_embeddings.to_csv(os.path.join(embeddings_path, target_embeddings_file_name), index=None)




    print ("Finished getting embeddings of target sets.")



def helper_function_to_get_embeddings_of_target_sets():

    ts_set_list = [("1584753511", "sets_10_patches_20_seg"),
                   ("1583770480", "sets_10_patches_20_seg"),
                   ("1585521837", "sets_10_patches_20_seg"),
                   ("1584025762", "sets_10_patches_20_seg"),
                   ("1586831151", "sets_5_patches_20_seg"),
                   ("1586740776", "sets_5_patches_20_seg"),
                   ("1587686591", "sets_20_patches_20_seg"),
                   ("1587462051", "sets_20_patches_20_seg"),
                   ("1589259198", "sets_10_patches_40_seg"),
                   ("1589258734", "sets_20_patches_40_seg"),
                   ("1589222258", "sets_20_patches_40_seg")]


    for item in ts_set_list:
        ts = item [0]
        set = item[1]

        print ("ts is: ", ts)
        path_to_set = os.path.join(DATA_DIR, STUDY, set)
        get_image_level_embeddings_of_a_target_set(path_to_set, ts)




def preprocess_zeng_layer_marker_and_expression(path_to_zeng):

    acceptable_layer_names = {"layer 1", "layer 1", "layer 3", "layer 4", "layer 5", "layer 6"}
    zeng_df = pd.read_csv(path_to_zeng)
    layer_marker_list = list(zeng_df['Cortical.marker..human.'])

    for i in range(len(layer_marker_list)):
        if layer_marker_list[i] not in acceptable_layer_names:
            layer_marker_list[i] = float("NaN")

    na_count = 0

    for item in layer_marker_list:
        if item not in acceptable_layer_names:
            na_count +=1

    #print ("There are {} NA values and {} layer markers from a total of {}.".format(na_count, len(layer_marker_list)-na_count,len(layer_marker_list)))

    zeng_df['Cortical.marker..human.'] = layer_marker_list

    new_zeng_path = path_to_zeng.split(".")[0] + "_processed.csv"
    #zeng_df.to_csv(new_zeng_path, index= None, na_rep='NA')



def merge_with_zeng_layer_marker_and_expression(path_to_zeng, path_to_gene_level_embeddings):


    zeng_df = pd.read_csv(path_to_zeng)
    gene_level_embed_df = pd.read_csv(path_to_gene_level_embeddings)

    zeng_gene_symbol_list = list(zeng_df['gene_symbol'])
    embed_gene_symbol_list = list(gene_level_embed_df['gene_symbol'])

    zeng_gene_symbol_list_unique =  set(zeng_gene_symbol_list)
    embed_gene_symbol_list_unique = set(embed_gene_symbol_list)


    in_zeng_not_in_embed = []
    in_embed_not_in_zeng = []


    for item in zeng_gene_symbol_list_unique:
        if item not in embed_gene_symbol_list_unique:
            in_zeng_not_in_embed.append(item)

    for item in embed_gene_symbol_list_unique:
        if item not in zeng_gene_symbol_list_unique:
            in_embed_not_in_zeng.append(item)

    print ("There are {} genes in Zeng that are not in the embeddings file.".format(len(in_zeng_not_in_embed)))
    print (in_zeng_not_in_embed)

    print ("----")
    print ("There are {} genes in the embeddings file that are not in Zeng.".format(len(in_embed_not_in_zeng)))
    print (in_embed_not_in_zeng)


    merged_with_markers_df = gene_level_embed_df.merge(zeng_df, how='left', on='gene_symbol')
    columns = list(merged_with_markers_df)
    columns = [columns[0]] + [columns[-2]] + [columns[-1] ] + columns[1:-2]
    merged_with_markers_df = merged_with_markers_df[columns]
    new_path = path_to_gene_level_embeddings.split(".")[0] + "_with_marker.csv"
    merged_with_markers_df.to_csv(new_path, index=None, na_rep='NA')

    acceptable_layer_names = {"layer 1", "layer 1", "layer 3", "layer 4", "layer 5", "layer 6"}
    na_count = 0

    layer_marker_col = list(merged_with_markers_df['Cortical.marker..human.'])
    for item in layer_marker_col:
        if item not in acceptable_layer_names:
            na_count += 1

    print("----")
    print("There are {} NA values and {} layer markers from a total of {}.".format(na_count,
                                                                                   len(layer_marker_col) - na_count,
                                                                                   len(layer_marker_col)))

    return new_path








if __name__ == '__main__':
    #generate_random_embeddings("", 128)
    #merge_embeddings_to_image_level("resnet50")
    #get_embeddings_from_pre_trained_model(standardize=True)
    #get_embeddings_from_pre_trained_model_in_chunks()
  
    #concatenate_embedding_chunks("resnet50_10_patches_standardized", number_of_chunks=10)
    #merge_embeddings_to_gene_level("1589259198")
    #merge_embeddings_to_image_level("1589259198")

    #helper_function_to_get_embeddings_of_target_sets()

    #get_stats("/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_2/human_ISH_info.csv")
    #path_to_csv = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_2/test_image_level_copy_2.csv"
    #convert_to_tsv(path_to_csv)


    #specific_donor_embeddings('H08-0025', '1587462051')



    path_to_zeng = "/Users/pegah_abed/Downloads/Cleaned_Zeng_dataset.csv"
    preprocess_zeng_layer_marker_and_expression(path_to_zeng)

    new_path_to_zeng = "/Users/pegah_abed/Downloads/Cleaned_Zeng_dataset_processed.csv"
    path_to_gene_level_embed = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_2/1591329662/triplet_training_validation_embeddings_gene_level.csv"
    new_path = merge_with_zeng_layer_marker_and_expression(new_path_to_zeng, path_to_gene_level_embed)
    #convert_to_tsv(new_path)
    








