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
import json





def get_sz_labels_image_and_donor_level():

    #path_to_sz_info = os.path.join(DATA_DIR, STUDY, "human_ISH_info.csv")
    general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/"
    path_to_sz_info = os.path.join(general_path, "sz", "human_ISH_info.csv")
    sz_info_df = pd.read_csv(path_to_sz_info)

    new_df = pd.DataFrame(columns=['ID', 'disease_diagnosis'])

    # --------------- image level
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

    new_df['disease_diagnosis'] = diagnosis
    file_name = "sz_diagnosis_image_level.csv"
    new_df.to_csv(os.path.join(general_path, file_name), index=None)

    print ("image sz count: ", image_sz_count)
    print ("image no sz count: ", image_no_sz_count)
    print ("total: ", image_sz_count + image_no_sz_count)


    # --------------- donor level
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

    new_df = pd.DataFrame(columns=['ID', 'disease_diagnosis'])
    new_df['ID']= donor_list
    new_df['disease_diagnosis'] = diagnosis_list

    file_name = "sz_diagnosis_donor_level.csv"
    new_df.to_csv(os.path.join(general_path, file_name), index=None)

    print ("donor sz count: ", donor_sz_count)
    print ("donor no sz count: ", donor_no_sz_count)
    print ("total: ", donor_sz_count + donor_no_sz_count)



def get_sz_labels_gene_level():
    general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/"
    path_to_sz_info = os.path.join(general_path, "sz", "human_ISH_info.csv")
    sz_info_df = pd.read_csv(path_to_sz_info)
    sz_genes = sz_info_df['gene_symbol'].unique()
    sz_genes_dict ={item:None for item in sz_genes}
    print ("sz genes: ", len(sz_genes))

    general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/"
    path_to_cortex_info = os.path.join(general_path, "cortex", "human_ISH_info.csv")
    cortex_info_df = pd.read_csv(path_to_cortex_info)
    cortex_genes = cortex_info_df['gene_symbol'].unique()
    not_sz_genes= [item for item in cortex_genes if item not in sz_genes_dict]
    print("not_sz_genes: ", len(not_sz_genes))

    total_genes = []
    total_diagnosis = []

    for item in sz_genes:
        total_genes.append(item)
        total_diagnosis.append(True)

    for item in not_sz_genes:
        total_genes.append(item)
        total_diagnosis.append(False)

    new_df = pd.DataFrame(columns=['ID', 'disease_diagnosis'])
    new_df['ID'] = total_genes
    new_df['disease_diagnosis'] = total_diagnosis

    file_name = "sz_diagnosis_gene_level.csv"
    new_df.to_csv(os.path.join(general_path, "sz", file_name), index=None)


def sz_diagnosis_create_training_files(path_to_embeddings, path_to_labels, levels, path_to_save_files):

    label_files = os.listdir(path_to_labels)
    embed_files = os.listdir(path_to_embeddings)
    label_df = pd.DataFrame()
    embed_df = pd.DataFrame()

    for level in levels:
        print ("----"*10)
        print ("level: ", level)

        # --- get label file
        level_label_file = None

        for item in label_files:
            if level in item:
                level_label_file = item
                break

        if level_label_file == None:
            print ("Could not find a label file at this level")
        else:
            print ("label file: ", level_label_file)
            label_df = pd.read_csv(os.path.join(path_to_labels, level_label_file))


        # --- get embed file
        if level == "image":
            level_embed_file = None
            for item in embed_files:
                if level in item and "schizophrenia" in item:
                    level_embed_file = item
                    break

            if level_embed_file == None:
                print ("Could not find a embed file at this level")
            else:
                print ("embed file: ", level_embed_file)
                embed_df = pd.read_csv(os.path.join(path_to_embeddings, level_embed_file))
                embed_df = embed_df.rename(columns={'image_id' : 'ID'})


        elif level == "donor":
            level_embed_file = None
            for item in embed_files:
                if level in item and "schizophrenia" in item:
                    level_embed_file = item
                    break

            if level_embed_file == None:
                print("Could not find a embed file at this level")
            else:
                print("embed file: ", level_embed_file)
                embed_df = pd.read_csv(os.path.join(path_to_embeddings, level_embed_file))
                embed_df = embed_df.rename(columns={'donor_id': 'ID'})


        elif level == "gene":

            level_no_sz_embed_file = None
            level_sz_embed_file = None
            for item in embed_files:
                if level in item and "schizophrenia" not in item:
                    level_no_sz_embed_file = item

                if level in item and "schizophrenia" in item:
                    level_sz_embed_file = item

            if level_no_sz_embed_file == None:
                print("Could not find a no-sz embed file at this level")

            if level_sz_embed_file == None:
                print("Could not find a sz embed file at this level")

            if level_no_sz_embed_file and level_sz_embed_file:
            
                no_sz_embed_df = pd.read_csv(os.path.join(path_to_embeddings, level_no_sz_embed_file))
                no_sz_embed_df = no_sz_embed_df.rename(columns={'gene_symbol': 'ID'})
                print (len(no_sz_embed_df))
                sz_embed_df = pd.read_csv(os.path.join(path_to_embeddings, level_sz_embed_file))
                sz_embed_df = sz_embed_df.rename(columns={'gene_symbol': 'ID'})
                print (len(sz_embed_df))


                embed_df = pd.concat([no_sz_embed_df, sz_embed_df], ignore_index=True)
                #print (embed_df.head())

            else:
                embed_df = None



        # at this point, i have the lavel df and the embed df
        # i want to merge them on their ID column
        # i want to keep the id, the 128 embedding vector values and the label


        merged_df = embed_df.merge(label_df, how="left", on="ID")

        # check number of columns
        cols = list(merged_df)
        if len(cols) != 130:
            print ("Something is wrong. There should be 130 columns (id, embeds, label")


        else:
            file_name = "sz_diagnosis_binary_class_" + level + "_level.csv"
            merged_df.to_csv(os.path.join(path_to_save_files, file_name), index=None)















if __name__ == "__main__":

    levels = ['gene']
    path_to_embeddings = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_3/1603427490"
    path_to_labels =  "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/sz"
    path_to_save_files = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/binary"
    sz_diagnosis_create_training_files(path_to_embeddings, path_to_labels, levels, path_to_save_files)


