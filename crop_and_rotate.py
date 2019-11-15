import numpy as np
from PIL import Image, ImageDraw
import cv2
import random
random.seed(1)
import numpy as np
from PIL import Image, ImageDraw
import os
import math
import pandas as pd
import csv
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from human_ISH_config import *

RANDOM_SELECT_COUNT = 1000

images_path = os.path.join(DATA_DIR, STUDY, "sample_images")
per_image_r_patches_path = os.path.join(DATA_DIR, STUDY, "per_image_r_patches")
overall_r_patches_path = os.path.join(DATA_DIR, STUDY, "overall_r_patches")

if not os.path.exists(per_image_r_patches_path):
    os.mkdir(per_image_r_patches_path)

if not os.path.exists(overall_r_patches_path):
    os.mkdir(overall_r_patches_path)




def check_existing_files(patches_path):
    """
    Check the patches folder to see which images already have patches and return the image name list.
    Each image name should appear NUMBER_OF_CIRCLES_IN_HEIGHT * NUMBER_OF_CIRCLES_IN_WIDTH times.
    Because the number of patches is equal to the number of circles in each image.
    :return: list of the image name of the patches.
    """
    patches_list = []
    images_list = []

    patch_items_list = os.listdir(patches_path)
    for item in patch_items_list:
        if item.endswith(".jpg"):  # the patches are saved with jpg format
            patches_list.append(item)

    image_items_list = os.listdir(images_path)
    for item in image_items_list:
        if item.endswith(".jpg"):  # the original images have jpg format
            images_list.append(item)

    print ("There are {} images in total.".format(len(images_list)))
    print ("{} patches already exist.".format(len(patches_list)))
    patches_list = [patch_name.split(".")[0].split("_")[0] for patch_name in patches_list]

    return patches_list




def rotate_horizontal_to_vertical(image_name):
    """
    This function checks the height and width of an img to see if it is oriented horizontally.
    If yes, it will rotate the image 90 degrees counter clockwise.
    :param img: a PIL library Image object
    :param image_name: String. Name of the image. To be used to save the image after rotation.
    :return: a PIL library Image object
    """
    image_path = os.path.join(images_path, image_name)
    # Open the input image as numpy array, convert to RGB
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w> h:  # the image is horizontal
        print ("image {} is horizontal.".format(image_name))
        print(h, w)
        img = img.rotate(90, expand=True)
        img.save(os.path.join(images_path, image_name))

    return img


def crop_circle(r, img, image_name, patches_path, circles_in_h = NUMBER_OF_CIRCLES_IN_HEIGHT, circles_in_w= NUMBER_OF_CIRCLES_IN_WIDTH):
    """
    This function gets the radius and image name and crops NUMBER_OF_CIRCLES_IN_HEIGHT * NUMBER_OF_CIRCLES_IN_WIDTH number of
    patches in the image. It first tries to crop two circles that would have the patches inside them (circumscribed circles).

    :param r: int. Circles' radius
    :param image_name: string.
    :param patches_path: string. The directory to store the patches into.
    :param circles_in_h: int. Number of circles that should fit in the image height.
    :param circles_in_w: int. Number of circles that should fit in the image width.
    :return: a list of 2 sub-lists. Each sub-list contains the parameter values of one of the patches. The function does
    not create this list. It gets it as a return value from rotate_circle().
    """

    total_number_of_circles = circles_in_h * circles_in_w
    circle_coordinates = []
    image_parameters_list = []

    npImage=np.array(img)
    w,h=img.size

    #print(image_name, h, w)

    height_section_length = h / circles_in_h
    width_section_length = w / (circles_in_w*2)

    #print ("height section length: ", height_section_length)
    #print ("width section length: ", width_section_length)


    #---- circles in a column ----
    shift_h = 0
    for i in range (circles_in_h):


        #---- circles in a row -----
        shift_w = 0
        for j in range(circles_in_w):
            top_left_corner_x = width_section_length - r + shift_w
            top_left_corner_y = height_section_length - (2*r) + shift_h
            bottom_right_corner_x = top_left_corner_x + (2*r)
            bottom_right_corner_y = top_left_corner_y + (2*r)

            circle_coordinates.append(top_left_corner_x)
            circle_coordinates.append(top_left_corner_y)
            circle_coordinates.append(bottom_right_corner_x)
            circle_coordinates.append(bottom_right_corner_y)

            shift_w = shift_w + (2*r)


        shift_h = shift_h + (2*r)

    #----- check -----
    if len(circle_coordinates) == total_number_of_circles * 4:
        #print (image_name)
        #print ("You have all the coordinate values")

        #print (circle_coordinates)

        patch_index = 1
        for i in range(0,len(circle_coordinates), 4):
            top_left_corner_x = int(circle_coordinates[i])
            top_left_corner_y = int(circle_coordinates[i+1])
            bottom_right_corner_x = int(circle_coordinates[i+2])
            bottom_right_corner_y = int(circle_coordinates[i+3])

            #print ("coordinates are: ")
            #print (top_left_corner_x, top_left_corner_y)
            #print (bottom_right_corner_x, bottom_right_corner_y)

            # Create same size alpha layer with circle
            alpha = Image.new('L', img.size,0)
            draw = ImageDraw.Draw(alpha)

            # coordinates are (x,y) format. Top Left corner is (0, 0)
            draw.pieslice([top_left_corner_x,top_left_corner_y,bottom_right_corner_x,bottom_right_corner_y],0,360,fill=255)   # four points to define the bounding box

            # Convert alpha Image to numpy array
            npAlpha=np.array(alpha)

            # Add alpha layer to RGB
            new_npImage=np.dstack((npImage,npAlpha))

            # Save with alpha
            new_image_name = image_name.split(".")[0] + "_cricle_" + str(top_left_corner_x) + "_" + str(top_left_corner_y) + "_" + str(bottom_right_corner_x) + "_" + str(bottom_right_corner_y) + ".png"
            #Image.fromarray(new_npImage).save(os.path.join(images_path, new_image_name))

            # for crpping the np array, coordinates are different.
            # it is no longer (x,y). It is (row, column)

            #  top_left_corner_x, top_left_corner_y
            #
            #
            #
            #
            #                                             bottom_right_corner_x, bottom_right_corner_y

            # change in x axis is like changing columns
            # change in y axis is like changing rows
            crop_img = new_npImage[top_left_corner_y:bottom_right_corner_y,
                       top_left_corner_x:bottom_right_corner_x]


            #print(crop_img.shape)
            # Save with alpha
            #new_image_name = image_name.split(".")[0] +"_c_circle_" + str(top_left_corner_x) + "_" + str(top_left_corner_y) + "_" + str(bottom_right_corner_x) + "_" \
                             #+ str(bottom_right_corner_y) + ".png"

            new_image_name = image_name.split(".")[0] + "_" + str(top_left_corner_x) + "_" + str(
                top_left_corner_y) + "_" + str(bottom_right_corner_x) + "_" + str(bottom_right_corner_y) + "_" + str(h) + "_" + str(w) + ".png"

            #Image.fromarray(crop_img).save(os.path.join(images_path, new_image_name))

            parameters_list = rotate_circle(crop_img, new_image_name, r, patches_path, patch_index)
            image_parameters_list.append(parameters_list)
            patch_index += 1

        return image_parameters_list



    else:
        print ("The number of coordinate values does not match the number of circles.")



def rotate_circle(img, circle_image_name, r, patches_path, patch_index):
    """
    This function gets a image, its name, and the circle radius as input and rotates the image with a random angle.
    It then calls the crop_within_circle() which will crop a square from the rotated circle image.

    :param img: numpy array. This is the circle crop that needs to be rotated
    :param circle_image_name: string, this is the image name. It will be used to save the rotated circle image and also
    as input to crop_within_circle()
    :param r: int, this is the circle radius
    :param patches_path: string. The directory in which the patches are stored.
    :param patch_index: int. The index of this patch. Each image creates NUMBER_OF_CIRCLES_IN_HEIGHT * NUMBER_OF_CIRCLES_IN_WIDTH
    number of patches. This index helps keep track of the patch that we are working on.
    :return: list. The list contains the parameter values of the patch. The function does
    not create this list. It gets it as a return value from crop_within_circle().
    """

    image_id = circle_image_name.split("_")[0]

    npImage = np.array(img)
    h,w,c = npImage.shape  # h = row   w = col

    angle_seed = int(image_id) * patch_index
    angle = np.random.RandomState(seed=angle_seed).choice(np.arange(360))  # choose a random number between 0 and 360 as the rotation angle.

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1) # the first parameter is the center of rotation
    rotated = cv2.warpAffine(npImage, M, (w, h)) # applies an affine transformation to the image.  # rotation is counter clockwise

    """
    Affine transformation is a linear mapping method that preserves points, straight lines, and planes. 
    Sets of parallel lines remain parallel after an affine transformation. The affine transformation technique is 
    typically used to correct for geometric distortions or deformations that occur with non-ideal camera angles
    """


    circle_image_name = circle_image_name.split(image_id)[1]
    new_image_name = image_id + "_r_" + str(r)+ "_rotate_" + str(angle) + circle_image_name
    #Image.fromarray(rotated).save(os.path.join(images_path, new_image_name))
    parameters_list = crop_within_circle(rotated, new_image_name, r, patches_path, patch_index)
    return parameters_list


def crop_within_circle(img, circle_image_name, r, patches_path, patch_index):
    """
    This function gets a rotated circle image, its name, and the circle radius as input and crops a square patch within the circle.
    :param img: numpy array. This is the circle image.
    :param circle_image_name: String. Image name. This is used when saving the cropped square patches.
    :param r: int. Circle radius
    :param patches_path: string. The directory in which the patches are stored.
    :param patch_index: int. The index of this patch. Each image creates NUMBER_OF_CIRCLES_IN_HEIGHT * NUMBER_OF_CIRCLES_IN_WIDTH
    number of patches. This index helps keep track of the patch that we are working on.
    :return: The list contains the parameter values of the patch. Parameters are:image_id, patch_index,
    original_image_height, original_image_width, radius, rotation_angle,top_left_x, top_left_y, bottom_right_x, bottom_right_y
    """
    np_img = np.array(img)
    w = np_img.shape[0]
    h = np_img.shape[1]

    top_left_corner_outer_square_x = 0
    top_left_corner_outer_square_y = 0
    bottom_right_corner_outer_square_x = 2*r
    bottom_right_corner_outer_square_y = 2*r

    #print (top_left_corner_outer_square_x, top_left_corner_outer_square_y)
    #print (bottom_right_corner_outer_square_x, bottom_right_corner_outer_square_y)



    #---- the following lines are calculations to get the coordinates of the square patch that fits within the circle.
    circle_diameter = bottom_right_corner_outer_square_x - top_left_corner_outer_square_x
    #print ("circle diameter is: ", circle_diameter)

    outer_square_side = circle_diameter
    inner_square_diagonal = circle_diameter

    inner_square_side = inner_square_diagonal / math.sqrt(2)
    #print ("inner square side is: ", inner_square_side)

    side_difference = outer_square_side - inner_square_side
    #print ("side difference is: ", side_difference)

    half_side_difference = side_difference / 2
    #print ("half side difference is: ", half_side_difference)

    top_left_corner_inner_square_x = int(top_left_corner_outer_square_x + half_side_difference)
    top_left_corner_inner_square_y = int(top_left_corner_outer_square_y + half_side_difference)
    bottom_right_corner_inner_square_x = int(top_left_corner_inner_square_x + inner_square_side)
    bottom_right_corner_inner_square_y = int(top_left_corner_inner_square_y + inner_square_side)

    #print(top_left_corner_inner_square_x, top_left_corner_inner_square_y)
    #print(bottom_right_corner_inner_square_x, bottom_right_corner_inner_square_y)

    # change in x axis is like changing columns
    # change in y axis is like changing rows
    crop_img = np_img[top_left_corner_inner_square_y:bottom_right_corner_inner_square_y,
               top_left_corner_inner_square_x:bottom_right_corner_inner_square_x]


    #print (circle_image_name)
    # ------- parameters -------

    # sample:
    # 78745070_r_2350_rotate_291_200_5634_4900_10334_11268_5100
    #  0      1   2     3     4   5   6    7     8    9     10

    without_extension = circle_image_name.split(".")[0]
    parameters = without_extension.split("_")
    image_id = parameters[0]
    original_image_height = parameters[9]
    original_image_width = parameters[10]
    radius = parameters[2]
    rotation_angle = parameters[4]
    top_left_x = parameters[5]
    top_left_y = parameters[6]
    bottom_right_x = parameters[7]
    bottom_right_y = parameters[8]

    # ---------------------------

    new_image_name = image_id + "_" + str(patch_index) + ".jpg"

    img = Image.fromarray(crop_img)
    img = img.convert("RGB")
    img = img.resize((PATCH_HEIGHT, PATCH_WIDTH), resample=0)
    img.save(os.path.join(patches_path, new_image_name))
    img.save(os.path.join(patches_path, new_image_name))

    parameters_list = [image_id, patch_index, original_image_height, original_image_width, radius, rotation_angle,top_left_x,
                       top_left_y, bottom_right_x, bottom_right_y]

    return parameters_list



def get_average_height_width(select_random=False):
    """
    Loads all the images and gets their height and width and returns the average height and average width values.
    If select_random == True, we randomly choose a RANDOM_SELECT_COUNT number of images to load and use.
    :return: average height and average width values of all the images.
    """
    images_path = os.path.join(DATA_DIR, STUDY, "images")
    image_list = os.listdir(images_path)
    image_count = len(image_list)
    print (image_count)

    # ----
    image_index_list = list(np.arange(image_count))
    random_image_index_list = random.sample(image_index_list, RANDOM_SELECT_COUNT)

    if select_random:
        image_count = len(random_image_index_list)
        image_list = random_image_index_list
    # ----

    sum_h = 0
    sum_w = 0

    for i in range (image_count):
        print (i)
        image_name = image_list[i]
        img = cv2.imread(os.path.join(images_path, image_name))
        h, w, c = img.shape
        if w > h:
            temp = h
            h = w
            w = temp

        sum_h += h
        sum_w += w

    avg_h = sum_h / image_count
    avg_w = sum_w / image_count

    return avg_h, avg_w



def get_min_height_width(select_random=False):
    """
    Loads all the images and gets their height and width and returns the minimum height and minimum width values.
    If select_random == True, we randomly choose a RANDOM_SELECT_COUNT number of images to load and use.
    :return: minimum height and minimum width values of all the images.
    """

    images_path = os.path.join(DATA_DIR, STUDY, "images")
    image_list = os.listdir(images_path)
    image_count = len(image_list)
    print("finding the minimum height and minimum width within {} images" .format(image_count))

    #----
    image_index_list = list(np.arange(image_count))
    random_image_index_list = random.sample(image_index_list, RANDOM_SELECT_COUNT)

    if select_random:
        image_count = len(random_image_index_list)
        image_list = random_image_index_list
    #----

    min_h = 100000
    min_w = 100000

    min_h_name = ""
    min_w_name = ""

    for i in range(image_count):
        print(i)
        image_name = image_list[i]
        img = cv2.imread(os.path.join(images_path, image_name))
        h, w, c = img.shape
        if w > h:   # if the image is horizontal, swap height and width values
            temp = h
            h = w
            w = temp

        if h < min_h:
            min_h = h
            min_h_name = image_name
        if w < min_w:
            min_w = w
            min_w_name = image_name

    print ("min_h_name: ", min_h_name)
    print ("min_w_name: ", min_w_name)
    print (min_h, min_w)
    return min_h, min_w


def get_min_height_width_v2(patches_path):

    per_image_patch_info_df= pd.read_csv(os.path.join(patches_path, "patches_info.csv"))
    list_of_heights = per_image_patch_info_df["original_image_height"]
    list_of_widths = per_image_patch_info_df["original_image_width"]

    print ("total number of height values: " , len(list_of_heights)/2)
    print ("total number of width values: ", len(list_of_widths)/2)

    min_height = min(list_of_heights)
    min_width = min(list_of_widths)
    print("min height is {} and min width is {} ".format(min_height, min_width))

    second_min_height =np.amin(list_of_heights[list_of_heights != min_height])
    second_min_width = np.amin(list_of_widths[list_of_widths != min_width])

    print("second min height is {} and second min width is {} ".format(second_min_height, second_min_width))




    height_threshold = second_min_height
    width_threshold = second_min_width
    h_below_th_count = 0
    w_below_th_count = 0
    for index in range(len(list_of_heights)):
        if list_of_heights[index] < height_threshold:
            h_below_th_count += 1
        if list_of_widths[index] < width_threshold:
            w_below_th_count += 1

    print ("number of height values below {} is {} ".format(height_threshold, h_below_th_count /2))
    print ("number of width values below {} is {}".format(width_threshold, w_below_th_count/2))


    #draw_height_width_histogram(list_of_heights, list_of_widths)


    return min_height, min_width



def draw_height_width_histogram(list_of_heights, list_of_widths):

    plt.figure()
    plt.hist(list_of_heights, bins= 30)
    plt.ylabel('Count')
    plt.xlabel('Hieght')


    plt.figure()
    plt.hist(list_of_widths, bins=30)
    plt.ylabel('Count')
    plt.xlabel('Width')

    plt.show()



def get_circle_radius_overall(h, w, circles_in_h=NUMBER_OF_CIRCLES_IN_HEIGHT, circles_in_w=NUMBER_OF_CIRCLES_IN_WIDTH):
    """
    This function calculates a radius value based on the height and width values that it gets from the input.
    :param h: int . image height
    :param w: int. image width
    :param circles_in_h: int. Number of circles that fit in the image height
    :param circles_in_w: int. Number of circles that fit in the image width
    :return: int. Circle radius
    """

    ratio = circles_in_h / circles_in_w

    if w >= h/ratio:
        r = (h/ratio)/2
        print ("width is bigger")
    else:
        r = w/2
        print ("height is bigger")

    #r = r-200  # to avoid margins

    r = int(r)
    print ("radius is: ", r)
    return r


def get_circle_radius_for_single_image(img, image_name, circles_in_h=NUMBER_OF_CIRCLES_IN_HEIGHT, circles_in_w=NUMBER_OF_CIRCLES_IN_WIDTH):
    """
    This function takes an image name, loads that image, fits NUMBER_OF_CIRCLES_IN_HEIGHT*NUMBER_OF_CIRCLES_IN_WIDTH
    circles with the same size in the image, and returns the value of the circle radius.
    :param image_name: String.
    :param circles_in_h: int. Number of circles that fit in the image height
    :param circles_in_w: int. Number of circles that fit in the image width
    :return: int. Circle radius
    """

    ratio = circles_in_h / circles_in_w
    w, h = img.size
    if w >= h/ratio:
        r = (h/ratio)/2
        #print ("width is bigger")
    else:
        r = w/2
        #print ("height is bigger")

    r = r-200  # to avoid margins

    r = int(r)
    #print ("radius is: ", r)
    return r


def create_patches_info_csv_file(patches_path, all_images_parameters_list):
    """
    Creates a csv files that has the information and parameter values of each patch in row.
    Parameters are: image_id, patch_index, original_image_height, original_image_width, radius, rotation_angle,
    top_left_x, top_left_y, bottom_right_x, bottom_right_y
    If a csv file already exists, this function reads it and appends its contents to the new dataframe and saves the new one.

    :param patches_path: string. The patches directory tos ave the csv file.
    :param all_images_parameters_list: list. It contains the parameters values of every patch.
    :return: None
    """

    # parameters_list = [image_id, patch_index, original_image_height, original_image_width, radius, rotation_angle,
    #                   top_left_x, top_left_y, bottom_right_x, bottom_right_y]




    columns = ['patch_id', 'original_image_height', 'original_image_width', 'radius', 'rotation_angle',
               'top_left_x', 'top_left_y', 'bottom_right_x', 'bottom_right_y']


    patches_info_df = pd.DataFrame(columns=columns)

    patch_id_list = []
    original_image_height_list = []
    original_image_width_list = []
    radius_list = []
    rotation_angle_list = []
    top_left_x_list = []
    top_left_y_list = []
    bottom_right_x_list = []
    bottom_right_y_list = []



    for parameters_list in all_images_parameters_list:

        patch_id = str(parameters_list[0]) + "_" + str(parameters_list[1])
        patch_id_list.append(patch_id)
        original_image_height_list.append(parameters_list[2])
        original_image_width_list.append(parameters_list[3])
        radius_list.append(parameters_list[4])
        rotation_angle_list.append(parameters_list[5])
        top_left_x_list.append(parameters_list[6])
        top_left_y_list.append(parameters_list[7])
        bottom_right_x_list.append(parameters_list[8])
        bottom_right_y_list.append(parameters_list[9])

    patches_info_df['patch_id'] = patch_id_list
    patches_info_df['original_image_height'] = original_image_height_list
    patches_info_df['original_image_width'] = original_image_width_list
    patches_info_df['radius'] = radius_list
    patches_info_df['rotation_angle'] = rotation_angle_list
    patches_info_df['top_left_x'] = top_left_x_list
    patches_info_df['top_left_y'] = top_left_y_list
    patches_info_df['bottom_right_x'] = bottom_right_x_list
    patches_info_df['bottom_right_y'] = bottom_right_y_list


    csv_file_name = "patches_info.csv"
    df_path = os.path.join(patches_path, csv_file_name)

    if os.path.exists(df_path):
        print ("patches_info.csv already exists!")
        old_patches_info_df = pd.read_csv(df_path)
        patches_info_df = patches_info_df.append(old_patches_info_df)


    patches_info_df = patches_info_df.sort_values(by=['patch_id'])
    patches_info_df.to_csv(df_path, index=None)



def create_valid_patches_info_csv_file(patches_path):

    image_info_df = pd.read_csv(os.path.join(DATA_DIR, STUDY ,"human_ISH_info.csv"))
    patches_info_df = pd.read_csv(os.path.join(os.path.join(patches_path, "patches_info.csv")))


    patch_id_list = patches_info_df['patch_id']
    patch_index_list = [patch_id.split('_')[1].split(".")[0] for patch_id in patch_id_list]
    patch_id_list = [int(patch_id.split("_")[0]) for patch_id in patch_id_list]

    patches_info_df['patch_id'] = patch_id_list
    patches_info_df['patch_index'] = patch_index_list

    patches_info_df = patches_info_df.rename(columns={'patch_id': 'image_id'})
    valid_patches_df = pd.merge(patches_info_df, image_info_df,  on='image_id')

    old_patch_id_list = [str(patch_id) + "_" + str(patch_index) for patch_id, patch_index in
                    zip(valid_patches_df['image_id'], valid_patches_df['patch_index'] )]


    valid_patches_df['image_id'] = old_patch_id_list
    valid_patches_df = valid_patches_df.rename(columns={'image_id': 'patch_id'})
    valid_patches_df= valid_patches_df.drop(columns = ['patch_index'])

    valid_patches_df.to_csv(os.path.join(patches_path, "valid_patches_info.csv"))




def run():

    # ------- use a single radius based on the globally minimum height and width ---------

    h, w = get_min_height_width_v2(per_image_r_patches_path)
    # h=4796
    # w= 1892
    r = get_circle_radius_overall(h, w)

    all_images_parameters_list = []
    image_list = []
    temp_list = os.listdir(images_path)
    existing_patches = check_existing_files(overall_r_patches_path)

    for file in temp_list:
        if file.endswith(".jpg"):
            file_name = file.split(".")[0]
            if existing_patches.count(file_name) != NUMBER_OF_CIRCLES_IN_HEIGHT * NUMBER_OF_CIRCLES_IN_WIDTH:
                image_list.append(file)

    print("There are  {} images to process".format(len(image_list)))

    for c in range(len(image_list)):
        print(c)
        image_name = image_list[c]
        img = rotate_horizontal_to_vertical(image_name)
        parameters_list = crop_circle(r, img, image_name, overall_r_patches_path)
        all_images_parameters_list = all_images_parameters_list + parameters_list

    create_patches_info_csv_file(overall_r_patches_path, all_images_parameters_list)
    create_valid_patches_info_csv_file(per_image_r_patches_path)

    # -----------------------------------------------------------------------------------

    """
    #------- fit a circle within each image and use its radius -------------------------

    all_images_parameters_list = []
    image_list = []
    temp_list = os.listdir(images_path)
    existing_patches = check_existing_files(per_image_r_patches_path)

    for file in temp_list:
        if file.endswith(".jpg"):
            #image_list.append(file)
            file_name = file.split(".")[0]
            if existing_patches.count(file_name) != NUMBER_OF_CIRCLES_IN_HEIGHT * NUMBER_OF_CIRCLES_IN_WIDTH:
                image_list.append(file)

    print("There are  {} images to process".format(len(image_list)))

    for c in range(len(image_list)):
        print(c)
        image_name = image_list[c]
        img = rotate_horizontal_to_vertical(image_name)
        r =get_circle_radius_for_single_image(img, image_name)
        parameters_list = crop_circle(r, img, image_name, per_image_r_patches_path)
        all_images_parameters_list = all_images_parameters_list + parameters_list

    create_patches_info_csv_file(per_image_r_patches_path, all_images_parameters_list)
    create_valid_patches_info_csv_file(overall_r_patches_path)

    # -----------------------------------------------------------------------------------
    """


if __name__ == "__main__":

    run()













