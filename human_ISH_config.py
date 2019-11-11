import os



LIST_OF_STUDIES = ["neurotransmitter", "cortex" , "subcortex", "schizophrenia", "autism"]
STUDY = "cortex"
DATA_DIR = "/external/mgmt3/genome/scratch/Neuroinformatics/pabed/human_ish_data"
CODE_DIR = "/external/mgmt3/genome/scratch/Neuroinformatics/pabed/human_brain_ish"
PATCH_TYPE = 'r_per_image'    # other option: 'r_overall'

TEST_SPLIT = 10
VALIDATION_SPLIT = 10
TRAINING_SPLIT = 100 - (TEST_SPLIT + VALIDATION_SPLIT)

PATCH_HEIGHT = 256
PATCH_WIDTH = 256
NUMBER_OF_CIRCLES_IN_HEIGHT = 2
NUMBER_OF_CIRCLES_IN_WIDTH = 1


if PATCH_TYPE == 'r_per_image':
    IMAGE_ROOT = os.path.join(DATA_DIR, STUDY, "per_image_r_patches")
    EXPERIMENT_ROOT = os.path.join(DATA_DIR, STUDY, "per_image_r_embeddings")
    if (not os.path.exists(EXPERIMENT_ROOT)):
        os.mkdir(EXPERIMENT_ROOT)

elif  PATCH_TYPE == 'r_overall' :
    IMAGE_ROOT = os.path.join(DATA_DIR, STUDY, "overall_r_patches")
    EXPERIMENT_ROOT = os.path.join(DATA_DIR, STUDY, "overall_r_embeddings")
    if (not os.path.exists(EXPERIMENT_ROOT)):
        os.mkdir(EXPERIMENT_ROOT)


TRAIN_SET =  os.path.join(DATA_DIR, STUDY, "sets", "triplet_training.csv")
INITIAL_CHECKPOINT = os.path.join(CODE_DIR, "resnet_v1_50", "resnet_v1_50.ckpt")
MODEL_NAME = 'resnet_v1_50'
HEAD_NAME = 'fc1024'
EMBEDDING_DIM = 128
BATCH_P = 50
BATCH_K = 2
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
FLIP_AUGMENT = False
CROP_AUGMENT = False
DETAILED_LOGS = False
