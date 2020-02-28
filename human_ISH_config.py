import os
import shutil


LIST_OF_STUDIES = ["neurotransmitter", "cortex" , "subcortex", "schizophrenia", "autism"]
STUDY = "cortex"
#DATA_DIR = "/external/mgmt3/genome/scratch/Neuroinformatics/pabed/human_ish_data"
#DATA_DIR = "/Users/pegah_abed/Documents/old_Human_ISH"
DATA_DIR = "/external/rprshnas01/netdata_kcni/lflab/SiameseAllenData/human_ISH/human_ish_data"
#DATA_DIR = "/human_ISH/human_ish_data"
CODE_DIR = "/human_ISH/human_ish_code"
PATCH_TYPE = 'segmentation'    # options: 'r_per_image' and 'r_overall' and 'segmentation'

TEST_SPLIT = 10
VALIDATION_SPLIT = 10
TRAINING_SPLIT = 100 - (TEST_SPLIT + VALIDATION_SPLIT)

PATCH_HEIGHT = 256
PATCH_WIDTH = 256
NUMBER_OF_CIRCLES_IN_HEIGHT = 2
NUMBER_OF_CIRCLES_IN_WIDTH = 1
SEGMENTATION = False

PATCH_COUNT_PER_IMAGE = 10
FOREGROUND_THRESHOLD = 90
SEGMENTATION_PATCH_SIZE = 1024


if PATCH_TYPE == 'r_per_image':
    IMAGE_ROOT = os.path.join(DATA_DIR, STUDY, "per_image_r_patches")
    EXPERIMENT_ROOT = os.path.join(DATA_DIR, STUDY, "experiment_files")
    EMBEDDING_DEST =  os.path.join(DATA_DIR, STUDY, "per_image_r_embeddings")


elif  PATCH_TYPE == 'r_overall' :
    IMAGE_ROOT = os.path.join(DATA_DIR, STUDY, "overall_r_patches")
    EXPERIMENT_ROOT = os.path.join(DATA_DIR, STUDY, "experiment_files")
    EMBEDDING_DEST = os.path.join(DATA_DIR, STUDY, "overall_r_embeddings")

elif PATCH_TYPE == 'segmentation':
    IMAGE_ROOT = os.path.join(DATA_DIR, STUDY, "segmentation_data", "final_patches")
    EXPERIMENT_ROOT = os.path.join(DATA_DIR, STUDY, "experiment_files")
    EMBEDDING_DEST = os.path.join(DATA_DIR, STUDY, "segmentation__embeddings")


TRAIN_SET =  os.path.join(DATA_DIR, STUDY, "sets", "triplet_training.csv")
EMBED_SET = os.path.join(DATA_DIR, STUDY, "sets", "triplet_training_validation.csv")
INITIAL_CHECKPOINT = os.path.join(DATA_DIR, "resnet_v1_50", "resnet_v1_50.ckpt")
TRIPLET_DIR = os.path.join(DATA_DIR, "triplet-reid")
MODEL_NAME = 'resnet_v1_50'
HEAD_NAME = 'fc1024'
TRAIN_EMBEDDING_DIM = 128
TRAIN_BATCH_P = 50
TRAIN_BATCH_K = 2
EMBED_BATCH_SIZE = 128
NET_INPUT_HEIGHT = PATCH_HEIGHT  # do you want to try 240?
NET_INPUT_WIDTH = PATCH_WIDTH # do you want to try 240?
PRE_CROP_HEIGHT = PATCH_HEIGHT
PRE_CROP_WIDTH = PATCH_WIDTH
LOADING_THREADS = 8
MARGIN = 'soft'
METRIC = 'euclidean'
LOSS = 'batch_hard'
LEARNING_RATE = 3e-4
TRAIN_ITERATIONS = 25000
DECAY_START_ITERATION = 15000
CHECKPOINT_FREQUENCY = 1000
TRAIN_FLIP_AUGMENT = False
EMBED_FLIP_AUGMENT = False
TRAIN_CROP_AUGMENT = False
EMBED_CROP_AUGMENT = None
DETAILED_LOGS = False
EMBED_AGGREGATOR = None



