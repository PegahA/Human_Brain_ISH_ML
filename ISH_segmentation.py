import fastai
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from fastai.callbacks import SaveModelCallback
from functools import partial
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import re, os
import random
from tqdm import tqdm
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
from PIL import Image
from pathlib import Path
import pandas as pd

from human_ISH_config import *

import torchvision

print(fastai.__version__,
torch.__version__,
torchvision.__version__,
cv.__version__)



SEGMENTATION_DATA_PATH = os.path.join(DATA_DIR,STUDY, "segmentation_data")
ORIGINAL_IMAGES_PATH =  os.path.join(DATA_DIR,STUDY, "images")
TRAIN_INPUT_IMAGE_SIZE = 224
PATCH_SIZE = SEGMENTATION_PATCH_SIZE



def pad_and_scale_for_training(images,labels,img_size):
    """
    In order to pre-process the images for training, we need to pad them so they become a square and also scale them into
    a smaller size that matches with training requirements.

    :param images: list of training images
    :param labels: list of training labels which are image masks
    :return: None
    """
    WHITE = [255, 255, 255]

    for i in range(len(images)):
        image = images[i]
        label = labels[i]


        img = cv.imread(image.as_posix(), cv.COLOR_BGR2GRAY)
        lbl = cv.imread(label.as_posix())  # , cv.COLOR_BGR2GRAY)

        img_h = img.shape[0]
        img_w = img.shape[1]

        h_w_dif = img_h - img_w

        # top bottom left right
        padded_img = cv.copyMakeBorder(img, 0, 0, h_w_dif // 2, h_w_dif // 2, cv.BORDER_CONSTANT, value=WHITE)
        padded_lbl = cv.copyMakeBorder(lbl, 0, 0, h_w_dif // 2, h_w_dif // 2, cv.BORDER_CONSTANT, value=WHITE)

        size = (img_size, img_size)
        scaled_img = cv.resize(padded_img, size)
        scaled_lbl = cv.resize(padded_lbl, size)

        cv.imwrite(os.path.join(image), scaled_img)
        cv.imwrite(os.path.join(label), scaled_lbl)

    print("done")



def make_tiles(images, labels, l, processed_dir):
    """
    Training requires images of size l*l. If the images in the dataset are not pre-processed to this size, this function
    will break them into tiles of size l*l
    If an image is l*l, the whole image becomes a single tile.

    :param images:  list of training images
    :param labels: list of training labels which are image masks
    :param l: the input size og images that the training model expects
    :param processed_dir: directory to save the tiles
    :return: None
    """
    empty_tile = 0
    populated_tile = 0

    for image_path, label_path in tqdm(zip(images, labels)):
        image = cv.imread(image_path.as_posix(), cv.COLOR_BGR2GRAY)
        # print (image.shape)
        label = cv.imread(label_path.as_posix(), cv.COLOR_BGR2GRAY)

        if image.shape != label.shape:
            raise ValueError(image_path.as_posix() + label_path.as_posix())
        i_max = image.shape[0] // l
        j_max = image.shape[1] // l

        # If the cells were labelled as 255, or something else mistakenly, instead of 1.
        label[label != 0] = 1

        for i in range(i_max):
            for j in range(j_max):
                cropped_image = image[l * i:l * (i + 1), l * j:l * (j + 1)]
                cropped_label = label[l * i:l * (i + 1), l * j:l * (j + 1)]

                if (cropped_label != 0).any():
                    populated_tile += 1
                    cropped_image_path = processed_dir / (
                                image_path.stem + "_i" + str(i) + "_j" + str(j) + image_path.suffix)
                    cropped_label_path = processed_dir / (
                                label_path.stem + "_i" + str(i) + "_j" + str(j) + label_path.suffix)
                else:
                    empty_tile += 1
                    cropped_image_path = processed_dir / (
                                image_path.stem + "_i" + str(i) + "_j" + str(j) + "_empty" + image_path.suffix)
                    cropped_label_path = processed_dir / (
                                label_path.stem + "_i" + str(i) + "_j" + str(j) + "_empty" + label_path.suffix)
                print(cropped_image.shape)
                cv.imwrite(cropped_image_path.as_posix(), cropped_image)
                cv.imwrite(cropped_label_path.as_posix(), cropped_label)

    print ("done")


def preprocess():

    raw_dir = Path("raw")

    raws = [raw_path for raw_path in raw_dir.ls() if ".tif" in raw_path.as_posix()]
    labels = sorted([raw_path for raw_path in raws if "_label" in raw_path.name])
    images = [Path(re.sub(r'_label', '_image', raw_path.as_posix())) for raw_path in labels]

    print("there are {} images".format(len(images)))

    for label, image in zip(labels, images):
        print(label, image)

    processed_dir = Path("processed")
    os.makedirs(processed_dir, exist_ok=True)
    # for f in processed_dir.ls(): os.remove(f)
    l = TRAIN_INPUT_IMAGE_SIZE

    pad_and_scale_for_training(images, labels, l)
    make_tiles(images, labels, l, processed_dir)


def fast_ai():
    transforms = get_transforms(
        do_flip=True,
        flip_vert=True,
        max_zoom=1,  # consider
        max_rotate=0,
        max_lighting=None,
        max_warp=None,
        p_affine=0.75,
        p_lighting=0.75)

    get_label_from_image = lambda path: re.sub(r'_image_', '_label_', path.as_posix())
    codes = ["TISSUE", "NON-TISSUE"]

    bs = 3

    src = (
        SegmentationItemList.from_folder(processed_dir)
            .filter_by_func(lambda fname: 'image' in Path(fname).name)
            .split_by_rand_pct(valid_pct=0.25, seed=2)
            .label_from_func(get_label_from_image, classes=codes)
    )
    data = (
        src.transform(transforms, tfm_y=True)
            .databunch(bs=bs, num_workers=0)
            .normalize(imagenet_stats)
    )


    print (data)

    # -------training ---------
    learn = unet_learner(
        data,
        models.resnet34,
        metrics=partial(dice, iou=True),
        model_dir='..')  # .to_fp16()

    print("done")

    lr_find(learn)
    learn.recorder.plot(suggestion=True)
    print("done")

    print("here")
    lr = 9.12E-05
    learn.fit_one_cycle(cyc_len=20,
                        callbacks=[SaveModelCallback(
                            learn,
                            every='improvement',
                            monitor='dice',
                            name='best-stage1')],
                        max_lr=lr)

    print("done")

    learn.recorder.plot_losses()

    learn.load("best-stage1");

    learn.unfreeze()

    lrs = slice(lr / 1000, lr / 10)

    learn.fit_one_cycle(cyc_len=12,
                        max_lr=lrs,
                        pct_start=0.8,
                        callbacks=[SaveModelCallback(
                            learn,
                            every='improvement',
                            monitor='dice',
                            name='best-stage2')])
    print("done")

    learn.export(file="../training_example.pkl")



def count_pixels(arr):
    """
    Given a 2D numpy array, this function counts how many elements there are of each unique value.
    # In our case, we expect the masks to have only 2 pixel values. 0 and 1 (or 0 and 255). If a mask image has pixels
    # of any other value something might be wrong.

    :param arr: a 2D numpy array
    :return: None
    """
    print ("counting pixel values...")
    unique, counts = np.unique(arr, return_counts=True)
    res = dict(zip(unique, counts))
    print (res)
    print ("----")


def check_percentage_foreground(arr, threshold, patch_size):
    """
    This function is used to determine if a patch image is valid or not.
    A patch image is valid if threshold % of its pixels are foreground which is the color black and the pixel value 1.


    :param arr: the numpy array that represents the patch image
    :param threshold: the threshold for determining whether a patch is valid
    :param patch_size: the patch image size
    :return: Boolean. Whether the patch image is valid or not.
    """

    total_count = patch_size * patch_size
    # white is 1
    # black is 0
    white_count = np.count_nonzero(arr)
    black_count = total_count - white_count

    if (black_count / total_count) * 100 >= threshold:
        return True

    return False


def use_trained_model(model_name, predict_new_masks=True):
    """
    Assuming that we have a trained model at this point, this function loads the model and runs all the images through
    it to get a predicted mask for each.

    :return:
    """

    model_path = os.path.join(SEGMENTATION_DATA_PATH, model_name)
    learner_path = Path(model_path)
    learn = fastai.basic_train.load_learner(path=learner_path.parent, file=learner_path.name)

    images_path = os.path.join(SEGMENTATION_DATA_PATH, "results")
    if not os.path.exists(images_path):
        os.mkdir(images_path)


    original_images_path = ORIGINAL_IMAGES_PATH
    dir_images_list = os.listdir(original_images_path)


    final_patches_path = os.path.join(images_path, "final_patches_"+str(PATCH_COUNT_PER_IMAGE))
    mask_patches_path = os.path.join(images_path, "mask_patches_"+str(PATCH_COUNT_PER_IMAGE))

    if not os.path.exists(final_patches_path):
        os.mkdir(final_patches_path)

    if not os.path.exists(mask_patches_path):
        os.mkdir(mask_patches_path)


    predicted_masks_path = os.path.join(SEGMENTATION_DATA_PATH, "predicted_masks")

    if not os.path.exists(predicted_masks_path):
        os.mkdir(predicted_masks_path)
        predicted_masks = []
    else:
        predicted_masks= os.listdir(predicted_masks_path)
        predicted_masks = [x.split("_")[0]+".jpg" for x in predicted_masks]
        #print (predicted_masks[:10])
        print ("There are already {} predicted masks".format(len(predicted_masks)))

    print ("Starting to pad and resize ISH images to predict a mask for them ...")

    
   
    for item in dir_images_list:
        if predict_new_masks:
            if item.endswith(".jpg") and item not in predicted_masks:  # the images are saved with jpg format
                print(item)

                final_name = item.split(".")[0] + "_mask.jpg"

                img = cv.imread(os.path.join(original_images_path, item), cv.COLOR_BGR2GRAY)
                img_h = img.shape[0]
                img_w = img.shape[1]

                h_w_dif = img_h - img_w

                # ------- pad the image to make it into a square
                padded_img = cv.copyMakeBorder(img, 0, 0, h_w_dif // 2, h_w_dif // 2, cv.BORDER_CONSTANT, value=WHITE)

                # ------- scale the image into TRAIN_INPUT_IMAGE_SIZExTRAIN_INPUT_IMAGE_SIZE  (224x244)
                size = (TRAIN_INPUT_IMAGE_SIZE, TRAIN_INPUT_IMAGE_SIZE)
                scaled_img = cv.resize(padded_img, size)

                # ------- convert to fastai.vision.image.Image type
                img_fastai = fastai.vision.image.Image(pil2tensor(scaled_img, dtype=np.float32).div_(255))

                # ------- get the prediction from the model
                segmented_img = learn.predict(img_fastai)[0].data.squeeze().numpy().astype('float32')
                segmented_img[segmented_img != 0] = 255


                # ------- save the predicted masks of size TRAIN_INPUT_IMAGE_SIZExTRAIN_INPUT_IMAGE_SIZE  (224x244)
                # we store them because we might want to go through them later to see if there is an image that gets
                # predicted is totally background or totally foreground.
                pred_name = item.split(".")[0] + "_pred.jpg"


                cv.imwrite(os.path.join(predicted_masks_path, pred_name), segmented_img)

                # ------- re-scale the segmented image back to original size
                size = (img_h, img_h)
                rescaled_img = cv.resize(segmented_img, size, interpolation=cv.INTER_NEAREST)

                # -------- append it to the list of rescaled masks along with its name
                #rescaled_masks.append((rescaled_img, final_name))


                print("Finished segementing and starting to get the patches for: ", final_name)

                create_patches(rescaled_img, final_name, original_images_path, final_patches_path, mask_patches_path)


        else:  # if we only want to create new patches. We already have the segmented masks.

            print ("using already existing prediction masks")

            if item.endswith(".jpg"):

                # read the image and get its size
                img = cv.imread(os.path.join(original_images_path, item), cv.COLOR_BGR2GRAY)
                img_h = img.shape[0]
                img_w = img.shape[1]

                # retrieve image's mask
                predicted_mask_name = item.split(".")[0] + "_pred.jpg"
                predicted_mask_path = os.path.join(predicted_masks_path, predicted_mask_name)
                predicted_mask = cv.imread(predicted_mask_path, cv.COLOR_BGR2GRAY)

                # rescale the mask to image's original size
                size = (img_h, img_h)
                rescaled_img = cv.resize(predicted_mask, size, interpolation=cv.INTER_NEAREST)

                final_name = item.split(".")[0] + "_mask.jpg"

                # call the function to create patches
                create_patches(rescaled_img, final_name, original_images_path, final_patches_path, mask_patches_path)



def create_patches(rescaled_img, final_name, original_images_path, final_patches_path, mask_patches_path):

    WHITE = [255, 255, 255]
    mask = rescaled_img
    mask_name = final_name  # the mask image name

    rand_state = int(mask_name.split("_")[0])

    this_image_lookup_count = 0
    this_image_patch_count = 0
    no_valid_patch = False
    while (this_image_patch_count < PATCH_COUNT_PER_IMAGE):
        patch_from_mask = image.extract_patches_2d(mask, (PATCH_SIZE, PATCH_SIZE), 1, rand_state)
        patch_from_mask[patch_from_mask > 128] = 255
        patch_from_mask[patch_from_mask <= 128] = 0

        valid = check_percentage_foreground(patch_from_mask, FOREGROUND_THRESHOLD, PATCH_SIZE)

        if valid:  # if this is a valid patch based on the mask, pick it from the original image

            patch_name = mask_name.split("_")[0] + "_" + str(this_image_patch_count) + ".jpg"
            cv.imwrite(os.path.join(mask_patches_path, patch_name), patch_from_mask[0])

            print("patch count: ", this_image_patch_count)
            this_image_patch_count += 1

            original_image_name = mask_name.split("_")[0] + ".jpg"
            original_image = cv.imread(os.path.join(original_images_path, original_image_name))
            img_h = original_image.shape[0]
            img_w = original_image.shape[1]
            h_w_dif = img_h - img_w

            original_image_padded = cv.copyMakeBorder(original_image, 0, 0, h_w_dif // 2, h_w_dif // 2,
                                                      cv.BORDER_CONSTANT, value=WHITE)
            patch_from_original = image.extract_patches_2d(original_image_padded, (PATCH_SIZE, PATCH_SIZE), 1,
                                                           rand_state)

            cv.imwrite(os.path.join(final_patches_path, patch_name), patch_from_original[0])

        rand_state += 1
        this_image_lookup_count += 1

        if this_image_lookup_count == 500:
            no_valid_patch == True
            break
           


def check_predicted_masks():
    """
    Returns the list of images for which we have a mask.
    The length of this list must be equal to the total number of original images. Because there should be a black and white
    mask for every image.
    :return: python list of strings
    """
    path_to_masks = os.path.join(SEGMENTATION_DATA_PATH, "predicted_masks")
    path_contents = os.listdir(path_to_masks)
    masks = [item for item in path_contents if item.endswith("_pred.jpg")]
    
    return masks


def check_final_patches():
    """
    Returns this list of final patches that we have from the images.
    These are the patches taken from original ISH images.

    :return: python list of strings
    """
    path_to_final_patches = os.path.join(SEGMENTATION_DATA_PATH, "results", "final_patches_"+ str(PATCH_COUNT_PER_IMAGE))
    path_contents = os.listdir(path_to_final_patches)
    final_patches = [item for item in path_contents if item.endswith(".jpg")]

    return final_patches

def check_mask_patches():
    """
    Returns this list of mask patches that we have from the images.
    There are patches taken from the masks.

    :return: python list of strings
    """

    path_to_mask_patches = os.path.join(SEGMENTATION_DATA_PATH, "results", "mask_patches_"+ str(PATCH_COUNT_PER_IMAGE ))
    path_contents = os.listdir(path_to_mask_patches)
    mask_patches = [item for item in path_contents if item.endswith(".jpg")]

    return mask_patches


def check_masks_and_patches_info():
    """
    This function checks to see for how many of the images there are no valid patches. And for how many of the images
    there are not enough valid patches. Enough means equal to PATCH_COUNT_PER_IMAGE.
    It creates a csv file and stores the image id and patch count of images that have less than PATCH_COUNT_PER_IMAGE valid patches.

    :returns: None
    """
    predicted_masks  = check_predicted_masks()
    print("There are {} masks.".format(len(predicted_masks)))
 
    final_patches = check_final_patches()
    mask_patches = check_mask_patches()

    if len(final_patches) != len(mask_patches):
        print ("something is wrong. The number of final patches does not match with number of mask patches")
	
    print ("Number of final patches: {} ".format(len(final_patches)))
    print ("Number of mask patches: {} ".format(len(mask_patches)))


    # -----------------------------------
    image_id_from_predicted_masks = [item.split("_pred.jpg")[0] for item in predicted_masks]

    image_id_from_final_patches = [item.split("_")[0] for item in final_patches]
    final_patches_values, final_patches_counts = np.unique(image_id_from_final_patches, return_counts=True)

    image_id_from_mask_patches = [item.split("_")[0] for item in mask_patches]
    mask_patches_values, mask_patches_counts = np.unique(image_id_from_mask_patches, return_counts=True)

    # -----------------------------------

    # There might be images that do not have any valid patches.
    # A valid patch is a patch with FOREGROUND_THRESHOLD % of tissue in it.
    # In these cases, we do have a mask, but there are no patch images generated from that mask.

    images_with_no_patches = []
    for item in image_id_from_predicted_masks:
        if item not in final_patches_values and item not in mask_patches_values:
             images_with_no_patches.append(item)
             

    print ("There are {} images with no valid patches".format(len(images_with_no_patches)))
    print (images_with_no_patches)

    print (len(final_patches_values))
    print (len(mask_patches_values))
    
    # ----------------------------------------
    final_patches_less_than_thresh_id = []
    final_patches_less_than_thresh_count = []


    # There are also cases were we have some valid patches from an image. But the count is less than PATCH_COUNT_PER_IMAGE.
    # So here, we keep track of the images that do not have enough patches and check how many patches they actually do have.
    for i in range(len(final_patches_values)):
         if final_patches_counts[i] != PATCH_COUNT_PER_IMAGE:
             final_patches_less_than_thresh_id.append(final_patches_values[i])
             final_patches_less_than_thresh_count.append(final_patches_counts[i])

   
    print ("-----")
    print ("There are {} images that have less than {} patches.".format(len(final_patches_less_than_thresh_id), PATCH_COUNT_PER_IMAGE))

    # ----------------------------
    for item in images_with_no_patches:
         final_patches_less_than_thresh_id.append(item)
         final_patches_less_than_thresh_count.append(0)

    # -----------------------------

    # we store the images that have less than PATCH_COUNT_PER_IMAGE patches in a csv file
    not_enough_patches_df = pd.DataFrame(columns=["image_id", "count"])
    not_enough_patches_df["image_id"] = final_patches_less_than_thresh_id
    not_enough_patches_df["count"] = final_patches_less_than_thresh_count

    csv_file_name = "less_than_" + str(PATCH_COUNT_PER_IMAGE) + ".csv"
    not_enough_patches_df.to_csv(os.path.join(SEGMENTATION_DATA_PATH, "outlier_images", csv_file_name), index=None)



def check_genes_in_images_with_not_enough_patches(file_name):
    """
    This is a helper function to check unique gene counts.
    The goal is to see if any genes will be removed from the data set if we remove the images from which we do not have
    enough patches.

    :param file_name: string: name of the csv files that has info of the images from which we do not have
    enough patches.

    :return: None
    """
    not_enough_patches_df = pd.read_csv(os.path.join(SEGMENTATION_DATA_PATH,"outlier_images", file_name))
    human_ish_info = pd.read_csv(os.path.join(DATA_DIR,STUDY, "human_ISH_info.csv"))

    general_unique_genes = set(human_ish_info["gene_symbol"])
 
    merge_res = not_enough_patches_df.merge(human_ish_info, how="left", on="image_id", )
    
    unique_genes = set(merge_res["gene_symbol"])
    
    print ("There are {} unique genes in human ISH info csv file".format(len(general_unique_genes)))
    print ("There are {} unique genes after merge".format(len(unique_genes)))

    remove_from_human_ish_info = human_ish_info[(~human_ish_info.image_id.isin(not_enough_patches_df.image_id))]
    print (len(remove_from_human_ish_info))
    

    print (len(set(remove_from_human_ish_info["gene_symbol"])))


def create_valid_patches_info_csv_file():
    
    print(IMAGE_ROOT)
    contents_list = os.listdir(IMAGE_ROOT)
    patches_list = [item for item in contents_list if item.endswith(".jpg")]

    patches_info_df = pd.DataFrame(columns=['patch_id'])
    patches_info_df['patch_id'] = patches_list


    image_info_df = pd.read_csv(os.path.join(DATA_DIR, STUDY, "human_ISH_info.csv"))
    columns = list(image_info_df)
    columns.insert(0, 'patch_id')

    patch_id_list = patches_info_df['patch_id']
    patch_index_list = [patch_id.split('_')[1].split(".")[0] for patch_id in patch_id_list]
    patch_id_list = [int(patch_id.split("_")[0]) for patch_id in patch_id_list]

    patches_info_df['patch_id'] = patch_id_list
    patches_info_df['patch_index'] = patch_index_list

    patches_info_df = patches_info_df.rename(columns={'patch_id': 'image_id'})
    valid_patches_df = pd.merge(patches_info_df, image_info_df, on='image_id')
    old_patch_id_list = [str(patch_id) + "_" + str(patch_index) for patch_id, patch_index in
                         zip(valid_patches_df['image_id'], valid_patches_df['patch_index'])]

    valid_patches_df['patch_id'] = old_patch_id_list
    valid_patches_df = valid_patches_df.drop(columns=['patch_index'])

    columns = list(valid_patches_df)
    columns = columns[-1:] + columns[:-1]

    valid_patches_df = valid_patches_df[columns]
    valid_patches_df = valid_patches_df.sort_values(by=['patch_id'])

    valid_patches_df.to_csv(os.path.join(IMAGE_ROOT, "valid_patches_info.csv"), index=None)

    print ("finished creating valid patches info csv file ...")
def main():
    #use_trained_model("training_example_feb_6.pkl")
    check_masks_and_patches_info()


if __name__ == "__main__":


    #create_valid_patches_info_csv_file()
    #main()

    #use_trained_model("training_example_feb_6.pkl",predict_new_masks=False )

    check_masks_and_patches_info()
    #check_genes_in_images_with_not_enough_patches("less_than_10.csv")



























