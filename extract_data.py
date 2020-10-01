"""
The functions in this file are used to extract data and download images of the human brain ISH data set from
the Allen Brain Atlas.
Link: http://human.brain-map.org/ish/search
"""

import urllib.request
import xml.etree.ElementTree as et
import os
import glob
import time
import sys
import pandas as pd
from human_ISH_config import *


if (not os.path.exists(os.path.join(DATA_DIR , STUDY))):
    os.mkdir(os.path.join(DATA_DIR , STUDY))


if (os.path.exists(os.path.join(DATA_DIR , STUDY , "xml_files"))):
    print ("xml_files folder already exists.")
else:
    os.mkdir(os.path.join(DATA_DIR , STUDY , "xml_files"))

XML_DIR = os.path.join(DATA_DIR , STUDY , "xml_files")



if (os.path.exists(os.path.join(DATA_DIR, STUDY, "images"))):
    print ("images folder already exists.")
else:
    os.mkdir(os.path.join(DATA_DIR, STUDY, "images"))
IMAGES_DIR = os.path.join(DATA_DIR, STUDY, "images")


#IMAGES_DIR = "/genome/scratch/Neuroinformatics/pabed/human_ish_images"
#HUMAN_DIR = "/genome/scratch/Neuroinformatics/pabed/human_ish"


def progressbar(it, prefix="", size=60, file=sys.stdout):

    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


def get_study_xml_file():
    """
    :return: the path to the xml file of the chosen study if the xml files exists. If not, it will return None.
    """

    study_xml_file = os.path.join(DATA_DIR, STUDY, STUDY.lower()+".xml")

    if (not os.path.exists(study_xml_file)):
        return None

    return study_xml_file


def get_specimen_id_list(study_xml_file):
    """
    The study xml files contains the list of specimen IDs corresponding to that study.
    This functions returns a list of the specimen IDs.
    :param study_xml_file: the xml file corresponding to the study
    :return: list of strings. Each string is a specimen ID.
    """


    print ("getting the list of specimen IDs within this study ...")

    list_of_specimen_ids = []

    tree = et.parse(study_xml_file)
    root = tree.getroot()
    specimens = root.find('specimens')

    all_specimen = specimens.findall('specimen')

    for item in all_specimen:
        list_of_specimen_ids.append(item.find('id').text)


    return list_of_specimen_ids



def construct_xml_file_per_specimen_id(specimen_id):
    """
    This function constructs the Allen Brain website's url of the xml pages of the given specimen ID.
    It stores the xml data into a file and returns the path to that file.
    The xml file contains info of the experiments that have been performed on this specimen.
    :param specimen_id: the specimen ID for which we want to make a xml file
    :return: the path to the xml file of this specimen ID
    """


    print ("processing specimen " + specimen_id + " ...")

    url_to_xml = "http://human.brain-map.org/api/v2/data/query.xml?criteria=model::Specimen%5Bid$in" + specimen_id + "%5D," \
                       "rma::include,donor(age),structure,data_sets(genes),rma::options%5Bnum_rows$eq%27all%27%5D"

    if (os.path.exists(os.path.join(XML_DIR , specimen_id ))):
        print ("(the folder for specimen "+ specimen_id +" already exists)")
    else:
        os.mkdir(os.path.join(XML_DIR , specimen_id ))

    specimen_id_xml_file = os.path.join(XML_DIR , specimen_id , specimen_id + ".xml")


    response = urllib.request.urlopen(url_to_xml)
    """
    with open(specimen_id_xml_file, 'w') as f:
        f.write(response.read().decode('utf-8'))
    """



    return response #specimen_id_xml_file


def get_experiment_id_list(specimen_xml_file, specimen_id):
    """
    For the given specimen xml file, this function goes through the file and retrieves the list of experiments on this specimen ID.
    Each experiment corresponds to a certain gene. In each experiment more than one slice may have been evaluated.
    :param specimen_xml_file: the path to the specimen xml file.
    :return: a list of strings. Each string is an experiment ID.
    """

    print("getting the list of experiment IDs within " + specimen_id +" ...")

    list_of_experiment_ids = []

    tree = et.parse(specimen_xml_file)
    root = tree.getroot()

    specimens = root.find('specimens')
    specimen = specimens.find('specimen')
    datasets = specimen.find('data-sets')

    all_datasets = datasets.findall('data-set')


    for item in all_datasets:
        list_of_experiment_ids.append(item.find('id').text)

    return list_of_experiment_ids


def construct_xml_file_per_experiment_id(experiment_id, specimen_id):
    """
    This function constructs the Allen Brain website's url of the xml pages of the given experiment ID.
    It stores the xml data into a file and returns the path to that file.
    This experiment corresponds to a certain gene. More than one slice may have been evaluated in this experiment.
    :param experiment_id: the experiment ID for which we want to make a xml file
    :param specimen_id: the specimen ID of the specimen that this experiment belongs to
    :return: the path to the xml file of this experiment ID
    """

    url_to_xml = "http://human.brain-map.org/api/v2/data/query.xml?criteria=model::SectionDataSet%5Bid$in" + experiment_id \
                 + "%5D,rma::include,genes,plane_of_section,section_images(associates,alternate_images),treatments," \
                 "specimen(donor(age,organism,conditions),structure),probes(orientation,predicted_sequence," \
                 "forward_primer_sequence,reverse_primer_sequence,products),rma::options%5Bnum_rows$eq%27all%27%5D"

    experiment_id_xml_file = os.path.join(XML_DIR, specimen_id , experiment_id + ".xml")


    response = urllib.request.urlopen(url_to_xml)
    """
    with open(experiment_id_xml_file, 'w') as f:
        f.write(response.read().decode('utf-8'))
    """


    return  response #experiment_id_xml_file


def get_image_id_list(experiment_xml_file, experiment_id):
    """
    Each experiment corresponds to a certain gene. In each experiment more than one slice may have been evaluated.
    This function returns a list of image IDs that correspond to a cerain experiment.
    :param experiment_xml_file: the xml file of the experiment
    :return: list of strings. Each string is an image ID.
    """

    #print("getting the list of image IDs within experiment " + experiment_id + " ...")

    list_of_image_ids = []

    tree = et.parse(experiment_xml_file)
    root = tree.getroot()

    section_data_sets = root.find('section-data-sets')
    section_data_set = section_data_sets.find('section-data-set')

    section_images = section_data_set.find('section-images')

    all_section_images = section_images.findall('section-image')

    for item in all_section_images:
        list_of_image_ids.append(item.find('id').text)


    return list_of_image_ids


def redownload_small_images(threshold = 6, downsample_rate=2.5):
    """
    this function checks the size of the downloaded images and re-downloads images that are smaller than some threshold.
    This is to make sure there are no corrupted images in the dataset.
    Corrupted images could be a result of connection error while downloading the images.

    :param downsample_rate: default is 2.5
    :param threshold: a threshold to define small images
    :return: None
    """

    images_list = os.listdir(IMAGES_DIR)
    print ("there are " + str(len(images_list)) + " existing images")

    threshold = threshold * 1000000

    for image_item in images_list:
        image_path = os.path.join(IMAGES_DIR, image_item)
        if os.path.getsize(image_path) < threshold:
            #print (image_item + " is less than 10 MB. Redownloading...")
            print ("Redownloading...")
            image_id = image_item.split(".")[0]
            default_url = "http://api.brain-map.org/api/v2/image_download/" + image_id + "?downsample=" + str(downsample_rate)
            urllib.request.urlretrieve(default_url, os.path.join(IMAGES_DIR, image_id + ".jpg"))


def download_images(image_list_to_download, downsample_rate=2.5, skip=True):
    """
    Gets a list if image IDs to download. if skip==True, skips downloading the image IDs that already exist in the directory.
    :param image_list_to_download: list of image IDs to download
    :param downsample_rate: downsampling rate to determine the final size of downloaded images. Default is 2.5
    :return: None
    """

    total_num_of_images = len(image_list_to_download)
    print(str(total_num_of_images) + " images to download.")

    existing_images_list = [f for f in glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))]
    num_of_existing_images = len(existing_images_list)

    for i in range(num_of_existing_images):
        existing_images_list[i] = existing_images_list[i].split("/")[-1].split(".")[0]

    print(str(num_of_existing_images) + " images already exist.")

    if skip:
        remaining_images_list = list(set(image_list_to_download) - set(existing_images_list))
    else:
        remaining_images_list = image_list_to_download

    num_of_remaining_images = len(remaining_images_list)
    print("downloading " + str(num_of_remaining_images) + " images...")

    # draw progess bar
    for i in progressbar(range(num_of_remaining_images), "Downloading: ", 100):
        time.sleep(0.1)
        image_id = remaining_images_list[i]
        default_url = "http://api.brain-map.org/api/v2/image_download/" + image_id + "?downsample=" + str(
            downsample_rate)
        urllib.request.urlretrieve(default_url, os.path.join(IMAGES_DIR, image_id + ".jpg"))



def add_experiment_images_to_image_info_csv(image_info_df, experiment_xml_file):
    """
    Goes through the xml file of the experiment and adds the info of its images to the image info dataframe.
    If the gene name is missing in the experiment, then this experiment is considered invalid.

    :param image_info_df: the image info dataframe to append the new images
    :param experiment_xml_file: the xml file of the experiment that we want to add its images
    :return: the image info dataframe and also a boolean which determines whether this experiment is invalid.
    """

    invalid = False
    tree = et.parse(experiment_xml_file)
    root = tree.getroot()

    section_data_sets = root.find('section-data-sets')
    section_data_set = section_data_sets.find('section-data-set')

    experiment_id = section_data_set.find('id').text
    specimen_id = section_data_set.find('specimen-id').text

    section_images = section_data_set.find('section-images')
    genes =  section_data_set.find('genes')

    specimen = section_data_set.find('specimen')
    donor = specimen.find('donor')
    structure = specimen.find('structure')

    donor_id = donor.find('name').text
    donor_sex = donor.find('sex').text
    donor_age = donor.find('age-id').text
    pmi = donor.find('pmi').text
    donor_race = donor.find('race-only').text
    smoker = donor.find('smoker').text
    chemotherapy = donor.find('chemotherapy').text
    radiation_therapy = donor.find('radiation_therapy').text
    tumor_status = donor.find('tumor_status').text

    conditions = donor.find('conditions')
    condition = conditions.find('condition')
    description = condition.find('description').text





    region_name = structure.find('name').text
    region_acronym = structure.find('acronym').text

    tissue_ph = specimen.find('tissue-ph').text


    gene = genes.find('gene')

    if gene == None:
        print ("experiment " + experiment_id + " is invalid")
        invalid = True

    else:


        gene_symbol = gene.find('acronym').text
        gene_alias_tags = gene.find('alias-tags').text
        entrez_id = gene.find('entrez-id').text
        gene_original_name = gene.find('original-name')
        gene_original_symbol = gene.find('original-symbol')

        all_section_images = section_images.findall('section-image')

        image_id_list = []
        for item in all_section_images:
            image_id_list.append(item.find('id').text)


        for image_id in image_id_list:
            new_row =  pd.Series({'image_id': image_id, 'gene_symbol': gene_symbol, 'entrez_id': entrez_id,
                                  'alias_tags': gene_alias_tags, 'original_name': gene_original_name,
                                  'original_symbol': gene_original_symbol, 'experiment_id':experiment_id,'specimen_id': specimen_id,
                                  'description': description, 'donor_id': donor_id, 'donor_sex': donor_sex,
                                  'donor_age':donor_age, 'donor_race':donor_race,
                                  'smoker' : smoker, 'chemotherapy': chemotherapy, 'radiation_therapy': radiation_therapy,
                                  'tumor_status' : tumor_status,
                                  'region':region_name, 'region_acronym': region_acronym,
                                  'tissue_ph': tissue_ph, 'pmi': pmi })

            image_info_df = image_info_df.append(new_row, ignore_index=True)


    return image_info_df, invalid




def run():

    print("STUDY: ", STUDY)
    image_list_to_download = []

    study_xml_file = get_study_xml_file()

    if study_xml_file == None:
        print("--- The study xml file does not exist. Make sure you download it from the Allen Brain website.")

    else:

        columns = ["image_id", "gene_symbol", "entrez_id", "experiment_id", "specimen_id"]
        image_info_df = pd.DataFrame(columns=columns)

        total_invalid_experiments = {}

        specimen_id_list = get_specimen_id_list(study_xml_file)
        for specimen_id in specimen_id_list:
            print(" --- ")

            invalid_experiments = {
                specimen_id: []}  # for some experiments, the gene name is missing on the Allen Brain website.

            specimen_xml_file = construct_xml_file_per_specimen_id(specimen_id)

            experiment_id_list = get_experiment_id_list(specimen_xml_file, specimen_id)

            print("getting this list of image IDs within " + specimen_id + " and adding them to image_info.csv ...")
            for experiment_id in experiment_id_list:
                experiment_xml_file_1 = construct_xml_file_per_experiment_id(experiment_id, specimen_id)
                experiment_xml_file_2 = construct_xml_file_per_experiment_id(experiment_id, specimen_id)

                image_id_list = get_image_id_list(experiment_xml_file_1, experiment_id)

                image_list_to_download = image_list_to_download + image_id_list

                image_info_df, invalid = add_experiment_images_to_image_info_csv(image_info_df, experiment_xml_file_2)

                if invalid:
                    invalid_experiments[specimen_id] = invalid_experiments[specimen_id] + [experiment_id]

            print("finished processing image IDs for " + specimen_id)

            # total_invalid_experiments = dict(total_invalid_experiments.items() + invalid_experiments.items())
            #total_invalid_experiments.update(invalid_experiments)

        #download_images(image_list_to_download)
        #redownload_small_images()

        # image_info_df.to_csv(os.path.join(HUMAN_DIR, STUDY, "human_ISH_info.csv"), index=None)
        image_info_df.to_csv(os.path.join(DATA_DIR, STUDY, "human_ISH_info_r.csv"), index=None)
        print("finished creating image_info csv file ...")

        """
        invalids_df = pd.DataFrame(columns=['specimen_id', 'experiment_id'])
        for k in total_invalid_experiments.keys():
            for v in total_invalid_experiments[k]:
                new_row = pd.Series({'specimen_id': k, 'experiment_id': v})

                invalids_df = invalids_df.append(new_row, ignore_index=True)

        #invalids_df.to_csv(os.path.join(HUMAN_DIR, STUDY, "invalids.csv"), index=None)
        invalids_df.to_csv(os.path.join(DATA_DIR, STUDY, "invalids.csv"), index=None)
        print ("finished creating invalid images csv file ...")
        """

if __name__ == "__main__":

    run()



