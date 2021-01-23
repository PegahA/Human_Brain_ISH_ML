# Human_Brain_ISH_ML
DNN architecture with Triplet Loss on genes from Human Brain ISH data


**extract_data.py**  




This file contains the code to download the human ISH images from the Allen Brain website. 
To get the images we need to go through these steps:

Study --> Specimens within this study --> Experiments done on each specimen --> Images that correspond to each experiment

We use the Allen Brain website API to download the human ISH images.  
There are different studies with human ISH data:  
Neurotransmitter Study, Cortex Study, Subcortex Study, Schizophrenia Study, Autism Study

You need to have the xml file of the study that you are interested in. This step is not part of the code. You need to manually download the xml file and store it in:  
```os.path.join(DATA_DIR , STUDY , "xml_files")```    

The code will check to see if you have the ```xml_files``` folder. It will create it if you don't have it. 

To get the xml file:
Go to http://human.brain-map.org/ish/search   
Select the study.  
At the buttom of the page you should see ```This data is also available as XML```. Click on it to get the xml version and save it with .xml format.  

Assuming that you have this xml file on your system, the ```get_study_xml_file()``` function will return it.  


The next step is getting a list of the specimen IDs of this study from the study xml file.  
In the xml file, there is a ```<specimens>``` tag, and within that, there is a ```<specimen>``` tag for evey specimen. Each specimen will then have an ```<id>``` tag. The ```get_specimen_id_list()``` function goes through the tags and returns a list of specimen IDs.   

There is an xml file corresponding to each specimen. We want to have a piece of code to construct the url to that xml file.  

The ```construct_xml_file_per_specimen_id()``` function will construct the url for each specimen in the specimen ID list and return it. You can create a folder for each specimen ID and store the specimen xml file in it. This part has been commented out in the code. 


For each specimen, there are a set of experiments that have been performed on that specimen. We want to get a list of these experiment IDs per each specimen. The ```get_experiment_id_list()``` function will get the specimen ID and the xml file of this specific specimen as input and return a list of its experiment IDs.  
The experiment ID is the ```<id>``` tag inside the ```<data_set>``` tag, which itself is inside the ```data_sets>``` tag.  

Now that we have a list of experiment IDs for each specimen, we need to construc the url to access the xml file of each experiment. The  ```construct_xml_file_per_experiment_id()``` function will get the specimen ID and the experiment ID as input and return the url to the xml file of that specific experiment. 


Now that we have the experiment xml file, we need to go through the file and find the image IDs to be able to download the images. The ```get_image_id_list()``` function will get the experiment ID and its corresponding xml file as input and return a list of image IDs that belong to this experiment. The image IDs are the ```<id>``` tags inside the ```<section-data-set>``` tag, which itself is inside the ```<section-data-sets>``` tag. 

We add these image IDs to the list of images that we need to download.   
Also, we have a csv file that contains the info of the images. We need to complete this csv file as we add new images to our dataset. The ```add_experiment_images_to_image_info_csv()``` function will add the new info to the csv file. The info includes: image_id, gene_symbol, entrez_id, experiment_id, specimen_id, donor_id, donor_sex.  

Each experiment corresponds to a certain gene. However, for some of the experiments, there is no gene_symbol. These cases are called invalid in the code. For each specimen, we store a list of its invalid experiments.  


The next step is to download the images. The ```download_images()``` function will get the list of image IDs to download as well as a value for downsampling as input. Using the image IDs and the downsampling value, it will construct a url that corresponds to that image and will use the url to download the image. 

The ```redownload_small_images()``` function will redownload the images that have a smaller size than some given threshold. This is to make sure that if there have been any interruptions while donwloading an image, it can be redownloaded. 


**ISH_segmentation.py**
Once we have downloaded the images, we want to take random patches from each image. The number of patches to be taken from each image can be set through human_ISH_config.py. In our set up, we take 50 random patches. We are interested in patches that contain tissue and not the margins or backgrounds of an image. To ensure this, we perform foreground-background segmentation using a U-Net architecture. The code for this step can be access throught this repo: 




**process.py**   

After downlading the images and creating the patches, the next thing to do is to create the training, validation, and test sets. In this project, we want to create sets in a way that there are no shared genes between the sets. The ```define_sets_with_no_shared_genes``` function does this and returns the 3 resulted files as pandas dataframes.  

Another way of generating the sets is by defining them to not have any shared donors. The ```define_sets_with_no_shared_donors()``` function does this and returns the 3 resulted files as pandas dataframes.  

Sidenote, we chose to have roughly 10% of the data as test, 10% as validation, and the rest for training. These values are specified in the ```human_ISH_config.py``` file and can be changed.   

After creating the sets, the ```get_stats_on_sets()``` function can give some extra info and statistics on them.   


