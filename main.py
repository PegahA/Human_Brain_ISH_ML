from human_ISH_config import *
#import extract_data
import process
import  ISH_segmentation
#import crop_and_rotate
#import evaluate_embeddings
from argparse import ArgumentParser
import os


TRAIN = True


parser = ArgumentParser(description='Train a ReID network.')

# ------ for training ------
parser.add_argument(
    '--experiment_root', default=EXPERIMENT_ROOT, type= str,
    help='Location used to store checkpoints and dumped data.')


parser.add_argument(
    '--train_set', default=TRAIN_SET,
    help='Path to the train_set csv file.')

parser.add_argument(
    '--image_root', default=IMAGE_ROOT, type=str,
    help='Path that will be pre-pended to the filenames in the train_set csv.')


parser.add_argument(
    '--train_resume',default=False,
    help='When this flag is provided, all other arguments apart from the '
         'experiment_root are ignored and a previously saved set of arguments '
         'is loaded.')

parser.add_argument(
    '--model_name', default=MODEL_NAME,
    help='Name of the model to use.')

parser.add_argument(
    '--head_name', default=HEAD_NAME,
    help='Name of the head to use.')

parser.add_argument(
    '--train_embedding_dim', default=TRAIN_EMBEDDING_DIM,
    help='Dimensionality of the embedding space.')

parser.add_argument(
    '--initial_checkpoint', default=INITIAL_CHECKPOINT,
    help='Path to the checkpoint file of the pretrained network.')

parser.add_argument(
    '--train_batch_p', default=TRAIN_BATCH_P,
    help='The number P used in the PK-batches')

parser.add_argument(
    '--train_batch_k', default=TRAIN_BATCH_K,
    help='The numberK used in the PK-batches')

parser.add_argument(
    '--net_input_height', default=NET_INPUT_HEIGHT,
    help='Height of the input directly fed into the network.')

parser.add_argument(
    '--net_input_width', default=NET_INPUT_WIDTH,
    help='Width of the input directly fed into the network.')

parser.add_argument(
    '--pre_crop_height', default=PRE_CROP_HEIGHT,
    help='Height used to resize a loaded image. This is ignored when no crop '
         'augmentation is applied.')

parser.add_argument(
    '--pre_crop_width', default=PRE_CROP_WIDTH,
    help='Width used to resize a loaded image. This is ignored when no crop '
         'augmentation is applied.')

parser.add_argument(
    '--loading_threads', default=LOADING_THREADS,
    help='Number of threads used for parallel loading.')

parser.add_argument(
    '--margin', default=MARGIN,
    help='What margin to use: a float value for hard-margin, "soft" for '
         'soft-margin, or no margin if "none".')

parser.add_argument(
    '--metric', default=METRIC,
    help='Which metric to use for the distance between embeddings.')

parser.add_argument(
    '--loss', default=LOSS,
    help='Enable the super-mega-advanced top-secret sampling stabilizer.')

parser.add_argument(
    '--learning_rate', default=LEARNING_RATE, type=float,
    help='The initial value of the learning-rate, before it kicks in.')

parser.add_argument(
    '--train_iterations', default=TRAIN_ITERATIONS, type=int,
    help='Number of training iterations.')

parser.add_argument(
    '--decay_start_iteration', default=DECAY_START_ITERATION, type=int,
    help='At which iteration the learning-rate decay should kick-in.'
         'Set to -1 to disable decay completely.')

parser.add_argument(
    '--checkpoint_frequency', default=CHECKPOINT_FREQUENCY, type=int,
    help='After how many iterations a checkpoint is stored. Set this to 0 to '
         'disable intermediate storing. This will result in only one final '
         'checkpoint.')

parser.add_argument(
    '--train_flip_augment', default=TRAIN_FLIP_AUGMENT,
    help='When this flag is provided, flip augmentation is performed.')

parser.add_argument(
    '--train_crop_augment', default=TRAIN_CROP_AUGMENT,
    help='When this flag is provided, crop augmentation is performed. Based on'
         'The `crop_height` and `crop_width` parameters. Changing this flag '
         'thus likely changes the network input size!')

parser.add_argument(
    '--detailed_logs', default=DETAILED_LOGS,
    help='Store very detailed logs of the training in addition to TensorBoard'
         ' summaries. These are mem-mapped numpy files containing the'
         ' embeddings, losses and FIDs seen in each batch during training.'
         ' Everything can be re-constructed and analyzed that way.')

# ------ for embedding ------

parser.add_argument(
    '--embed_dataset', default=EMBED_SET,
    help='Path to the dataset csv file to be embedded.')

parser.add_argument(
    '--embed_batch_size', default=EMBED_BATCH_SIZE,
    help='Batch size used during evaluation, adapt based on available memory.')


parser.add_argument(
    '--embed_flip_augment', default=EMBED_FLIP_AUGMENT,
    help='When this flag is provided, flip augmentation is performed.')

parser.add_argument(
    '--embed_crop_augment', choices=['center', 'avgpool', 'five'], default=EMBED_CROP_AUGMENT,
    help='When this flag is provided, crop augmentation is performed.'
         '`avgpool` means the full image at the precrop size is used and '
         'the augmentation is performed by the average pooling. `center` means'
         'only the center crop is used and `five` means the four corner and '
         'center crops are used. When not provided, by default the image is '
         'resized to network input size.')

parser.add_argument(
    '--embed_aggregator',  default=EMBED_AGGREGATOR,
    help='The type of aggregation used to combine the different embeddings '
         'after augmentation.')


# --------------


if __name__ == "__main__":
    #extract_data.run()
    #crop_and_rotate.create_patches(PATCH_TYPE)
    #process.make_sets() 
    """
    print ("i am here in main!")

    
    args = parser.parse_args()
    print ("\n------- Arguments:")
    print ("experiment root: ", args.experiment_root)
    print ("train set: ", args.train_set)
    print ("image root: ", args.image_root)
    print ("train resume: ", args.train_resume)
    print ("model name: ", args.model_name)
    print ("head name: ", args.head_name)
    print ("train embedding dim: ", args.train_embedding_dim)
    print ("initial checkpoint: ", args.initial_checkpoint)
    print ("train batch p: ", args.train_batch_p)
    print ("train batch k: ", args.train_batch_k)
    print ("net input height: ", args.net_input_height)
    print ("net input width: ", args.net_input_width)
    print ("pre crop height: ", args.pre_crop_height)
    print ("pre crop width: ", args.pre_crop_width)
    print ("loading threads: ", args.loading_threads)
    print ("margin: ", args.margin)
    print ("metric: ", args.metric)
    print ("loss: ", args.loss)
    print ("learning rate: ", args.learning_rate)
    print ("train iterations: ", args.train_iterations)
    print ("decay start iteration: ", args.decay_start_iteration)
    print ("checkpoint frequency: ", args.checkpoint_frequency)
    print ("train flip augment: ", args.train_flip_augment)
    print ("train crop augment: ", args.train_crop_augment)
    print ("detailed logs: ", args.detailed_logs)


    print ("embed dataset: ", args.embed_dataset)
    print ("embed_batch_size: ", args.embed_batch_size)
    print ("embed flip augment: ", args.embed_flip_augment)
    print ("embed crop augment: ", args.embed_crop_augment)
    print ('embed aggregator: ', args.embed_aggregator)
   
    

    train_py_path = os.path.join(TRIPLET_DIR, "train.py")
    train_command_line_string = "python " + train_py_path + \
                          " --experiment_root=" + "'" + args.experiment_root + "'" + \
                          " --train_set=" + "'" + args.train_set + "'" \
                          " --image_root=" + "'" + args.image_root + "'" + \
                          (" --resume" if args.train_resume else "") + \
                          " --model_name=" + "'" + args.model_name + "'" + \
                          " --head_name=" + "'" + args.head_name + "'" + \
                          " --embedding_dim=" + str(args.train_embedding_dim) + \
                          " --initial_checkpoint=" + "'" + args.initial_checkpoint + "'" \
                          " --batch_p=" + str(args.train_batch_p) + \
                          " --batch_k=" + str(args.train_batch_k) + \
                          " --net_input_height=" + str(args.net_input_height) + \
                          " --net_input_width=" + str(args.net_input_width) + \
                          " --pre_crop_height=" + str(args.pre_crop_height) + \
                          " --pre_crop_width=" + str(args.pre_crop_width) + \
                          " --loading_threads=" + str(args.loading_threads) + \
                          " --margin=" + "'" + args.margin + "'" + \
                          " --metric=" + "'" + args.metric + "'" +\
                          " --loss=" + "'" + args.loss + "'" +\
                          " --learning_rate="  + str(args.learning_rate) + \
                          " --train_iterations=" + str(args.train_iterations) + \
                          " --decay_start_iteration=" + str(args.decay_start_iteration) + \
                          " --checkpoint_frequency=" + str(args.checkpoint_frequency) + \
                          (" --flip_augment" if args.train_flip_augment else "") + \
                          (" --crop_augment" if args.train_crop_augment else "") + \
                          (" --detailed_logs" if args.detailed_logs else "")


    embed_py_path = os.path.join(TRIPLET_DIR, "embed.py")
    embed_command_line_string = "python " + embed_py_path  + \
                                " --experiment_root=" + "'" + args.experiment_root + "'" + \
                                " --dataset=" + "'" + args.embed_dataset + "'" +\
                                " --image_root=" + "'" + args.image_root + "'" + \
                                " --loading_threads=" + str(args.loading_threads) + \
                                " --batch_size=" + str(args.embed_batch_size) + \
                                (" --flip_augment" if args.embed_flip_augment else "") + \
                                (" --crop_augment=" + args.embed_crop_augment if args.embed_crop_augment else "") + \
                                (" --aggregator=" + args.embed_aggregator if args.embed_aggregator else "")

     
    if os.path.exists(EXPERIMENT_ROOT) and os.path.isdir(EXPERIMENT_ROOT):
        shutil.rmtree(EXPERIMENT_ROOT)

    os.system(train_command_line_string)
    os.system(embed_command_line_string)
    
    process.convert_h5_to_csv()
    filename = process.save_embedding_info_into_file()
    process.merge_embeddings_to_gene_level(filename)
    #evaluate_embeddings.evaluate(filename)
    """
   
    




