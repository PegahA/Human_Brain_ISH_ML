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

import torchvision
print(fastai.__version__,
torch.__version__,
torchvision.__version__,
cv.__version__)


MAIN_DATA_PATH = "/external/rprshnas01/netdata_kcni/lflab/SiameseAllenData/human_ISH/segmentation_data"
ORIGINAL_IMAGES_PATH =  "/genome/scratch/Neuroinformatics/pabed/human_ish_data/cortex/images"
TRAIN_INPUT_IMAGE_SIZE = 224
PATCH_SIZE = 1024
PATCH_COUNT_PER_IMAGE = 10
FOREGROUND_THRESHOLD = 90


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


def use_trained_model(model_name):
    """
    Assuming that we have a trained model at this point, this function loads the model and runs all the images through
    it to get a predicted mask for each.

    :return:
    """

    model_path = os.path.join(MAIN_DATA_PATH, model_name)
    learner_path = Path(model_path)
    learn = fastai.basic_train.load_learner(path=learner_path.parent, file=learner_path.name)

    images_path = os.path.join(MAIN_DATA_PATH, "results")
    original_images_path = ORIGINAL_IMAGES_PATH
    dir_images_list = os.listdir(original_images_path)


    final_patches_path = os.path.join(images_path, "final_patches")
    mask_patches_path = os.path.join(images_path, "mask_patches")

    if not os.path.exists(final_patches_path):
        os.mkdir(final_patches_path)

    if not os.path.exists(mask_patches_path):
        os.mkdir(mask_patches_path)


    predicted_masks_path = os.path.join(MAIN_DATA_PATH, "predicted_masks")

    if not os.path.exists(predicted_masks_path):
        os.mkdir(predicted_masks_path)
    else:
        predicted_masks= os.listdir(predicted_masks_path)
        predicted_masks = [x.split("_")[0]+".jpg" for x in predicted_masks]
        #print (predicted_masks[:10])
        print ("There are already {} predicted masks".format(len(predicted_masks)))

    WHITE = [255, 255, 255]

    print ("Starting to pad and resize ISH images to predict a mask for them ...")

    invalid_images_path = os.path.join(MAIN_DATA_PATH, "invalid_images", "invalid_images.txt")
    invalid_images = open(invalid_images_path, "w")
   
    for item in dir_images_list:
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




    #for element in rescaled_masks:

            mask = rescaled_img
            mask_name =  final_name  # the mask image name


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

                    print ("patch count: ", this_image_patch_count)
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
                this_image_lookup_count +=1
		
                if this_image_lookup_count == 100:
                    no_valid_patch == True			
                    break

            if no_valid_patch:
                invalid_images.write(item)
                invalid_images.write('\n')


def check_predicted_masks():
    path_to_masks = os.path.join(MAIN_DATA_PATH, "predicted_masks")
    path_contents = os.listdir(path_to_masks)
    masks = [item for item in path_contents if item.endswith("_pred.jpg")]
    
    return masks


def check_final_patches():
    path_to_final_patches = os.path.join(MAIN_DATA_PATH, "results", "final_patches")
    path_contents = os.listdir(path_to_final_patches)
    final_patches = [item for item in path_contents if item.endswith(".jpg")]

    return final_patches

def check_mask_patches():
    path_to_mask_patches = os.path.join(MAIN_DATA_PATH, "results", "mask_patches")
    path_contents = os.listdir(path_to_mask_patches)
    mask_patches = [item for item in path_contents if item.endswith(".jpg")]

    return mask_patches


def check_masks_and_patches_info():
    predicted_masks  = check_predicted_masks()
    print("There are {} masks.".format(len(predicted_masks)))
 
    final_patches = check_final_patches()
    mask_patches = check_mask_patches()

    if len(final_patches) != len(mask_patches):
        print ("something is wrong. The number of final patches does not match with number of mask patches")
	
    print ("Number of final patches: {} ".format(len(final_patches)))
    print ("Number of mask patches: {} ".format(len(mask_patches)))



    image_id_from_predicted_masks = [item.split("_pred.jpg")[0] for item in predicted_masks]

    image_id_from_final_patches = [item.split("_")[0] for item in final_patches]
    final_patches_values, final_patches_counts = np.unique(image_id_from_final_patches, return_counts=True)

    image_id_from_mask_patches = [item.split("_")[0] for item in mask_patches]
    mask_patches_values, mask_patches_counts = np.unique(image_id_from_mask_patches, return_counts=True)

    images_with_no_patches = []
    for item in image_id_from_predicted_masks:
        if item not in final_patches_values and item not in mask_patches_values:
             images_with_no_patches.append(item)

    print ("There are {} images with no valid patches".format(len(images_with_no_patches)))
    print (images_with_no_patches)

    print (len(final_patches_values))
    print (len(mask_patches_values))
    
    # -------------------------------
    final_patches_less_than_thresh_id = []
    final_patches_less_than_thresh_count = []
    
    #mask_patches_less_than_thresh = []
 
    for i in range(len(final_patches_values)):
         if final_patches_counts[i] != 10:
             #print ("{} : {} ".format(final_patches_values[i], final_patches_counts[i]))
             final_patches_less_than_thresh_id.append(final_patches_values[i])
             final_patches_less_than_thresh_count.append(final_patches_counts[i])

    #for i in range(len(mask_patches_values)):
         #if mask_patches_counts[i] != 10:
             #print ("{} : {} ".format(mask_patches_values[i], mask_patches_counts[i]))
             #mask_patches_less_than_thresh.append(mask_patches_values[i])
   
    print ("-----")
    #print (final_patches_less_than_thresh_count)
    #print (mask_patches_less_than_thresh_count)

    # -----------------------------
    not_enough_patches_df = pd.DataFrame(columns=["image_id", "count"])
    #final_patches_less_than_thresh_id = [item+".jpg" for item in final_patches_less_than_thresh_id]
    not_enough_patches_df["image_id"] = final_patches_less_than_thresh_id
    not_enough_patches_df["count"] = final_patches_less_than_thresh_count

    not_enough_patches_df.to_csv(os.path.join(MAIN_DATA_PATH, "outlier_images", "less_than_10.csv"), index=None)


if __name__ == "__main__":


    #use_trained_model("training_example_feb_6.pkl")

    check_masks_and_patches_info()




























