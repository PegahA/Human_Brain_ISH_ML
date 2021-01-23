import os
import shutil
import time


LIST_OF_STUDIES = ["neurotransmitter", "cortex" , "subcortex", "schizophrenia", "autism"]
STUDY = "cortex"

cwd = os.getcwd()
DATA_DIR = os.path.join(cwd, "human_ish_data")
if (not os.path.exists(DATA_DIR)):
    os.mkdir(DATA_DIR)

CODE_DIR = cwd
PATCH_TYPE = 'segmentation'

TRAIN_ON_ALL = False

TEST_SPLIT = 10
VALIDATION_SPLIT = 10
TRAINING_SPLIT = 100 - (TEST_SPLIT + VALIDATION_SPLIT)

PATCH_HEIGHT = 256
PATCH_WIDTH = 256
NUMBER_OF_CIRCLES_IN_HEIGHT = 2
NUMBER_OF_CIRCLES_IN_WIDTH = 1


PATCH_COUNT_PER_IMAGE = 50
FOREGROUND_THRESHOLD = 90
SEGMENTATION_PATCH_SIZE = 1024
SEGMENTATION_TRAINING_SAMPLES = 40

current_time  = int(time.time())
TIMESTAMP = str(current_time)



IMAGE_ROOT = os.path.join(DATA_DIR, STUDY, "segmentation_data" ,"trained_on_"+str(SEGMENTATION_TRAINING_SAMPLES), "results" , "final_patches_"+str(PATCH_COUNT_PER_IMAGE))
EXPERIMENT_ROOT = os.path.join(DATA_DIR, STUDY, "experiment_files", "experiment_" + TIMESTAMP)
EMBEDDING_DEST = os.path.join(DATA_DIR, STUDY, "segmentation_embeddings")



INCLUDE_SZ_DATA = False

SETS_DIR = os.path.join(DATA_DIR, STUDY, "sets_"+str(PATCH_COUNT_PER_IMAGE) + "_patches_"+str(SEGMENTATION_TRAINING_SAMPLES)+"_seg")

if TRAIN_ON_ALL == False:
    if INCLUDE_SZ_DATA == True:
        TRAIN_SET =  os.path.join(SETS_DIR, "triplet_training.csv")
        EMBED_SET = os.path.join(SETS_DIR, "triplet_training_validation.csv")
    else:
        TRAIN_SET = os.path.join(SETS_DIR, "triplet_no_sz_training.csv")
        EMBED_SET = os.path.join(SETS_DIR, "triplet_no_sz_training_validation.csv")
else:
    if INCLUDE_SZ_DATA == True:
        TRAIN_SET =  os.path.join(SETS_DIR, "triplet_no_sz_all_training.csv")
        EMBED_SET = os.path.join(SETS_DIR, "triplet_no_sz_all_training.csv")
    else:
        TRAIN_SET = os.path.join(SETS_DIR, "triplet_all_training.csv")
        EMBED_SET = os.path.join(SETS_DIR, "triplet_all_training.csv")

TEST_SET = os.path.join(SETS_DIR, "triplet_test.csv")


INITIAL_CHECKPOINT = os.path.join(DATA_DIR, "resnet_v1_50", "resnet_v1_50.ckpt")
TRIPLET_DIR = os.path.join(CODE_DIR, "triplet-reid")
MODEL_NAME = 'resnet_v1_50'
HEAD_NAME = 'fc1024'
TRAIN_EMBEDDING_DIM = 128
TRAIN_BATCH_P = 17
TRAIN_BATCH_K = 17
EMBED_BATCH_SIZE = 128
NET_INPUT_HEIGHT = PATCH_HEIGHT
NET_INPUT_WIDTH = PATCH_WIDTH
PRE_CROP_HEIGHT = PATCH_HEIGHT
PRE_CROP_WIDTH = PATCH_WIDTH
LOADING_THREADS = 20
MARGIN = 'soft'
METRIC = 'euclidean'
LOSS = 'batch_hard'
LEARNING_RATE = 7e-5
TRAIN_ITERATIONS = 30000
DECAY_START_ITERATION = 25000
CHECKPOINT_FREQUENCY = 0
TRAIN_STANDARDIZE = False
TRAIN_FLIP_AUGMENT = True
EMBED_FLIP_AUGMENT = False
TRAIN_CROP_AUGMENT = False
EMBED_CROP_AUGMENT = None
DETAILED_LOGS = False
EMBED_AGGREGATOR = None



