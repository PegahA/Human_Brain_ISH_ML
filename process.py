import os
import pandas as pd
import numpy as np
import random
from human_ISH_config import *
import h5py

random.seed(1)

if (not os.path.exists(os.path.join(DATA_DIR, STUDY, "sets"))):
    os.mkdir(os.path.join(DATA_DIR, STUDY, "sets"))


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

    training_df.to_csv(os.path.join(DATA_DIR, STUDY, "sets", "training.csv"), index=None)
    validation_df.to_csv(os.path.join(DATA_DIR, STUDY, "sets", "validation.csv"), index=None)
    test_df.to_csv(os.path.join(DATA_DIR, STUDY, "sets", "test.csv"), index=None)

    return training_df, validation_df, test_df


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


def make_triplet_csv(df, out_file):
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



def make_triplet_csvs(dfs):

    out_base = os.path.join(DATA_DIR, STUDY, "sets") + "/triplet"
    return tuple((make_triplet_csv(df, "{}_{}.csv".format(out_base,ext)) and "{}_{}.csv".format(out_base, ext))
                 for df, ext in zip(dfs, ("training", "validation", "test")))



def convert_h5_to_csv():

    exp_root_contents = os.listdir(EXPERIMENT_ROOT)
    for item in exp_root_contents:
        if item.endswith(".h5"):
            embedding_csv_name = item.split(".")[0] + ".csv"
            set_csv_file_name = item.replace("_embedding", "")
            print ("set csv file name is: ", set_csv_file_name)

            set_csv_file = os.path.join(DATA_DIR, STUDY, "sets", set_csv_file_name)
            df = pd.read_csv(set_csv_file, names=['gene', 'image_id'])
            f = h5py.File(item, 'r')['emb']
            df['image_id']= df.apply(lambda x: x['image_id'].split('.')[0], axis =  1)
            pd.DataFrame(np.array(f), index=df.image_id).to_csv(os.path.join(EXPERIMENT_ROOT, embedding_csv_name))


def run():

    images_info_df = pd.read_csv(os.path.join(DATA_DIR, STUDY, "human_ISH_info.csv"))

    stats_dict = get_stats(images_info_df)

    training_df, validation_df, test_df = define_sets_with_no_shared_genes(images_info_df)
    get_stats_on_sets(stats_dict, training_df, validation_df, test_df)

    make_triplet_csvs((training_df, validation_df, test_df))

    """
    training_df, validation_df, test_df = define_sets_with_no_shared_donors(images_info_df)

    train_validation_shared_genes_list, train_test_shared_genes_list, validation_test_shared_genes_list, all_shared_genes_list = \
        compare_set_genes_list(training_df, validation_df, test_df)

    new_training_df, new_validation_df, test_df = create_new_sets_by_removing_shared_genes(images_info_df, training_df, validation_df, test_df,
                                             train_validation_shared_genes_list, train_test_shared_genes_list,
                                             validation_test_shared_genes_list, all_shared_genes_list)

    get_stats_on_sets(stats_dict, new_training_df, new_validation_df, test_df)
    """

if __name__ == '__main__':

    run()








