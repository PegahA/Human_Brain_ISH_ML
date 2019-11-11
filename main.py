from human_ISH_config import *
import extract_data
import process
import crop_and_rotate
from argparse import ArgumentParser
import os


parser = ArgumentParser(description='Train a ReID network.')


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
    '--resume',default=False,
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
    '--embedding_dim', default=EMBEDDING_DIM,
    help='Dimensionality of the embedding space.')

parser.add_argument(
    '--initial_checkpoint', default=INITIAL_CHECKPOINT,
    help='Path to the checkpoint file of the pretrained network.')

parser.add_argument(
    '--batch_p', default=BATCH_P,
    help='The number P used in the PK-batches')

parser.add_argument(
    '--batch_k', default=BATCH_K,
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
    '--flip_augment', default=FLIP_AUGMENT,
    help='When this flag is provided, flip augmentation is performed.')

parser.add_argument(
    '--crop_augment', default=CROP_AUGMENT,
    help='When this flag is provided, crop augmentation is performed. Based on'
         'The `crop_height` and `crop_width` parameters. Changing this flag '
         'thus likely changes the network input size!')

parser.add_argument(
    '--detailed_logs', default=DETAILED_LOGS,
    help='Store very detailed logs of the training in addition to TensorBoard'
         ' summaries. These are mem-mapped numpy files containing the'
         ' embeddings, losses and FIDs seen in each batch during training.'
         ' Everything can be re-constructed and analyzed that way.')


if __name__ == "__main__":
    #extract_data.run()
    #crop_and_rotate.run()
    #process.run()

    args = parser.parse_args()
    print ("\n------- Starting training with:")
    print ("experiment root: ", args.experiment_root)
    print ("train set: ", args.train_set)
    print ("image root: ", args.image_root)
    print ("resume: ", args.resume)
    print ("model name: ", args.model_name)
    print ("head name: ", args.head_name)
    print ("embedding dim: ", args.embedding_dim)
    print ("initial checkpoint: ", args.initial_checkpoint)
    print ("batch p: ", args.batch_p)
    print ("batch k: ", args.batch_k)
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
    print ("flip augment: ", args.flip_augment)
    print ("crop augment: ", args.crop_augment)
    print ("detailed logs: ", args.detailed_logs)



    command_line_string = "python train.py" + \
                          " --experiment_root=" + "'" + args.experiment_root + "'" + \
                          " --train_set=" + "'" + args.train_set + "'" \
                          " --image_root=" + "'" + args.image_root + "'" + \
                          " --resume=" + str(args.resume) + \
                          " --model_name=" + "'" + args.model_name + "'" + \
                          " --head_name=" + "'" + args.head_name + "'" + \
                          " --embedding_dim=" + str(args.embedding_dim) + \
                          " --initial_checkpoint= " + "'" + args.initial_checkpoint + "'" \
                          " --batch_p=" + str(args.batch_p) + \
                          " --batch_k=" + str(args.batch_k) + \
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
                          " --checkpoint_frequency=" + str(args.checkpoint_frequency) +\
                          " --flip_augment=" + str(args.flip_augment) + \
                          " --crop_augment=" + str(args.crop_augment) + \
                          " --detailed_logs=" + str(args.detailed_logs)


    print (command_line_string)












