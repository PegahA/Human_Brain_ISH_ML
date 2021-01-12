
import pandas as pd
import os
import numpy as np

general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/"
sz_general_path = os.path.join(general_path, "dummy_4/sz")



def get_info_on_GO_results(path_to_GO_results):

    GO_results_df = pd.read_csv(path_to_GO_results)
    auc_col = list(GO_results_df['AUC'])

    avg_auc = np.mean(auc_col)

    print ("The average AUC is {} over {} GO terms.".format(avg_auc, len(GO_results_df)))


def get_and_compare_avg_AUC():
    rnd_state = 42
    input_types = ['embed','demog' , 'demog_and_embed', 'random', 'plain_resnet']


    print ("all_genes")
    all_genes_path = os.path.join(sz_general_path, "all_genes")
    for input_type in input_types:
        print ("---"*20)
        print (input_type)

        res_files_path = os.path.join(all_genes_path, input_type)
        contents = os.listdir(res_files_path)
        for content in contents:
            if "_rnd_state_" + str(rnd_state) in content:
                print (content)
                if 'lr' in content:
                    lr_file_path  = os.path.join(res_files_path, content)
                    lr_file = pd.read_csv(lr_file_path)

                elif 'rf' in content:
                    rf_file_path = os.path.join(res_files_path, content)
                    rf_file = pd.read_csv(rf_file_path)


        lr_auc = list(lr_file['AUC'])
        print ("lr avg: ", np.mean(lr_auc))

        rf_auc = list(rf_file['AUC'])
        print("rf avg: ", np.mean(rf_auc))

        print ("---"*20)


    print ("*" *100)
    print ("top_20_genes")
    top_20_genes_path = os.path.join(sz_general_path, "top_20_genes")

    for input_type in input_types:
        print("---" * 20)
        print(input_type)

        res_files_path = os.path.join(top_20_genes_path, input_type)
        contents = os.listdir(res_files_path)
        for content in contents:
            if "_rnd_state_" + str(rnd_state) in content:
                if 'lr' in content:
                    lr_file_path = os.path.join(res_files_path, content)
                    lr_file = pd.read_csv(lr_file_path)

                elif 'rf' in content:
                    rf_file_path = os.path.join(res_files_path, content)
                    rf_file = pd.read_csv(rf_file_path)

        lr_auc = list(lr_file['AUC'])
        print("lr avg: ", np.mean(lr_auc))

        rf_auc = list(rf_file['AUC'])
        print("rf avg: ", np.mean(rf_auc))

        print("---" * 20)












if __name__ ==  "__main__":


    ts = "1603427156"
    path_to_GO_results = os.path.join(general_path, "dummy_3", ts, "1603427156_avg_across_folds_go_scores_40_200_rand_state_1.csv")
    #get_info_on_GO_results(path_to_GO_results)

    get_and_compare_avg_AUC()