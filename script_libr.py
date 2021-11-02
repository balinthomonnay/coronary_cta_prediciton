#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
import random
import math
import pydicom
import pandas as pd
import shutil
import tensorflow as tf
import xml.etree.ElementTree as ET
from functools import partial, update_wrapper

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# Retrieving blocks of a numpy array
from skimage.util import view_as_blocks
# Retrieving blocks of a numpy array with given stride sizes
from skimage.util.shape import view_as_windows
from random import randint

from tqdm import tqdm
from random import randint
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, \
    Conv2DTranspose, Conv3DTranspose
from keras.layers import Activation, add, multiply, Lambda
from keras.layers import AveragePooling2D, AveragePooling3D, average, UpSampling2D, UpSampling3D, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.initializers import glorot_normal, random_normal, random_uniform
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.callbacks import *
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.losses import binary_crossentropy
import numpy as np

from hyperopt import fmin, hp, tpe, Trials, space_eval
from hyperopt.pyll import scope as ho_scope
from hyperopt.pyll.stochastic import sample as ho_sample




class plaquetypes:
    limits = pd.DataFrame(
        [['DenseCalcium', 351, 10000], ['Fibrous', 131, 350], ['FibrousFatty', 76, 130], ['NecroticCore', -30, 75],
         ['NonCalcified', -1000, 350]], columns=['type', 'lower', 'upper'])


# ,['MedisDenseCalcium',351,10000],['MedisFibrous',151,350],['MedisFibrousFatty',31,150],['MedisNecroticCore',-100,30]


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def get_pairs(vesel_list):
    return_list = []
    for ID in vesel_list:
        if ID.endswith('NoContour.dcm'):
            return_list.append([ID, ID[:-13] + "Contour1.dcm", ID[:-13] + "Contour2.dcm", ID[:8]])
    return return_list


def find_included_segment(xml_file):
    try:
        doc = ET.parse(xml_file)
        root = doc.getroot()

        info = 0
        included = False
        while not (included):
            info += 1
            if (root[2][info][25].attrib['value'] == 'SEGMENT_TYPE_NORMAL'):
                breakpoints = [root[2][info][24].attrib['value'], root[2][info][1].attrib['value']]
                included = True
    except:
        doc = ET.parse(xml_file)
        root = doc.getroot()

        info = 0
        included = False
        while not (included):
            info += 1
            if (root[3][info][25].attrib['value'] == 'SEGMENT_TYPE_NORMAL'):
                breakpoints = [root[3][info][24].attrib['value'], root[3][info][1].attrib['value']]
                included = True
    return breakpoints


def get_xml(basic_path2, dicom):
    dicom = dicom.replace('_', '/')
    dicom = dicom[:-13] + 'data_model.xml'
    xml_file = os.path.join(basic_path2, dicom)
    return xml_file


def get_image_mask_for_ID_tuple(ID_tuple, basic_path, basic_path2, dir_to_save, plaques_only):
    image_path = os.path.join(basic_path, ID_tuple[0])
    mask_1_path = os.path.join(basic_path, ID_tuple[1])
    mask_2_path = os.path.join(basic_path, ID_tuple[2])
    image = pydicom.dcmread(image_path)
    mask_1 = pydicom.dcmread(mask_1_path)
    mask_2 = pydicom.dcmread(mask_2_path)
    if plaques_only==True:
        xml_file = get_xml(basic_path2, ID_tuple[0])
        breakpoints = np.array(find_included_segment(xml_file))
        breakpoints = breakpoints.astype(float)
        breakpoints = breakpoints / image.SpacingBetweenSlices
        if 'Plakk' in dir_to_save:
            image_array = (image.pixel_array)[int(breakpoints[0]):int(breakpoints[1]), :, :]
            mask_1_array = (mask_1.pixel_array)[int(breakpoints[0]):int(breakpoints[1]), :, :]
            mask_2_array = (mask_2.pixel_array)[int(breakpoints[0]):int(breakpoints[1]), :, :]
        else:
            image_array = (image.pixel_array)[int(breakpoints[0]):, :, :]
            mask_1_array = (mask_1.pixel_array)[int(breakpoints[0]):, :, :]
            mask_2_array = (mask_2.pixel_array)[int(breakpoints[0]):, :, :]
    else:
        image_array = (image.pixel_array)
        mask_1_array = (mask_1.pixel_array)
        mask_2_array = (mask_2.pixel_array)

    return [image_array, mask_1_array, mask_2_array]


def apply_breakpoints(image_array, breakpoints, dir_to_save):
    if 'Plakk' in dir_to_save:
        image_array = (image_array)[int(breakpoints[0]):int(breakpoints[1]), :, :]
    else:
        image_array = (image_array)[int(breakpoints[0]):, :, :]
    return image_array


def osszefuz(x):
    x.append(x[0][:7])
    x.append(x[0][16:18])
    return x


def save_all_patch_for_image_mask_pair(ID_tuple,
                                       dir_to_save,
                                       patch_shape,
                                       stride_size,
                                       train_val_test,
                                       basic_path,
                                       basic_path2,
                                       truncate=True,
                                       plaques_only=False
                                       ):
    """Saves all 3 dimensional patches
    
    Arguments
    -----------
    ID_tuple
        
    dir_to_save : string
        Folder to save the patches.
        
    train_val_test : string 
        possible values: 'train', 'val', or 'test'.
        Subfolders for dataset split.
        
        
    Outputs
    -----------
    None 
    """

    image_array, mask_1, mask_2 = get_image_mask_for_ID_tuple(ID_tuple, basic_path, basic_path2, dir_to_save,plaques_only)
    mask_1 = np.where(mask_1 > 0, 1, 0)
    mask_2 = np.where(mask_2 > 0, 1, 0)
    dif_array = np.where(mask_2 - mask_1 == 1, 1, 0)
    # dif_array = get_all_plaques(dif_array)
    # Count saved patches
    total_patches_for_ID = 0
    image_to_pad = image_array
    nodule_to_pad = mask_1
    lung_to_pad = mask_2
    dif_to_pad = dif_array
    # Order of the saved patch, appended to filename
    patch_count = 0

    # Shape of original images
    size_X = image_to_pad.shape[2]
    size_Y = image_to_pad.shape[1]
    size_Z = image_to_pad.shape[0]

    image_to_block = np.zeros((size_Z + patch_shape[2],
                               size_Y,
                               size_X))
    image_to_block[:size_Z, :size_Y, :size_X] = image_to_pad

    nodule_to_block = np.zeros((size_Z + patch_shape[2],
                                size_Y,
                                size_X))
    nodule_to_block[:size_Z, :size_Y, :size_X] = nodule_to_pad

    lung_to_block = np.zeros((size_Z + patch_shape[2],
                              size_Y,
                              size_X))
    lung_to_block[:size_Z, :size_Y, :size_X] = lung_to_pad

    dif_to_block = np.zeros((size_Z + patch_shape[2],
                             size_Y,
                             size_X))
    dif_to_block[:size_Z, :size_Y, :size_X] = dif_to_pad

    # patch_shape is originally in order XYZ, however for view as window we need it in ZYX
    patch_shape_ZYX = [patch_shape[2], patch_shape[1], patch_shape[0]]
    # Same as patch_shape
    stride_size_ZYX = [stride_size[2], stride_size[1], stride_size[0]]
    # Create blocks of the numpy arrays using view_as_blocks from skimage.util
    image_patches = view_as_windows(image_to_block, window_shape=patch_shape_ZYX, step=stride_size_ZYX)
    nodule_patches = view_as_windows(nodule_to_block, window_shape=patch_shape_ZYX, step=stride_size_ZYX)
    lung_patches = view_as_windows(lung_to_block, window_shape=patch_shape_ZYX, step=stride_size_ZYX)
    dif_patches = view_as_windows(dif_to_block, window_shape=patch_shape_ZYX, step=stride_size_ZYX)

    # view_as_windows creates 6 dimensional numpy arrays: 
    # first 3 dimensions encode the position of the patch, last 3 dimensions encode patch shape.
    # We will iterate through the number of patches
    number_of_patches = image_patches.shape[0] * image_patches.shape[1] * image_patches.shape[2]
    for counter in range(number_of_patches):
        patch_coor_1 = int(counter // (image_patches.shape[1] * image_patches.shape[2]))

        patch_coor_2 = int(((counter - patch_coor_1 * image_patches.shape[1] * image_patches.shape[2])
                            // image_patches.shape[2]))

        patch_coor_3 = int(counter - patch_coor_1 * image_patches.shape[1] * image_patches.shape[2]
                           - patch_coor_2 * image_patches.shape[2])

        image_patch = image_patches[patch_coor_1][patch_coor_2][patch_coor_3]
        nodule_patch = nodule_patches[patch_coor_1][patch_coor_2][patch_coor_3]
        lung_patch = lung_patches[patch_coor_1][patch_coor_2][patch_coor_3]
        dif_patch = dif_patches[patch_coor_1][patch_coor_2][patch_coor_3]
        if truncate == True:
            # vedd ki a 16:48
            image_patch = image_patch[:, 16:48, 16:48]
            nodule_patch = nodule_patch[:, 16:48, 16:48]
            lung_patch = lung_patch[:, 16:48, 16:48]
            dif_patch = dif_patch[:, 16:48, 16:48]
        if plaques_only and np.count_nonzero(dif_patch) > 0:
            image_patch_file = os.path.join(dir_to_save, train_val_test, "images",
                                            ID_tuple[0][:-13] + str(patch_count) + '.npy')
            np.save(image_patch_file, image_patch.astype(np.float32))

            nodule_patch_file = os.path.join(dir_to_save, train_val_test, "masks_1",
                                             ID_tuple[0][:-13] + str(patch_count) + '.npy')

            np.save(nodule_patch_file, nodule_patch.astype(np.uint8))

            lung_patch_file = os.path.join(dir_to_save, train_val_test, "masks_2",
                                           ID_tuple[0][:-13] + str(patch_count) + '.npy')

            np.save(lung_patch_file, lung_patch.astype(np.uint8))

            plaque_patch_file = os.path.join(dir_to_save, train_val_test, "plaques",
                                             ID_tuple[0][:-13] + str(patch_count) + '.npy')

            np.save(plaque_patch_file, dif_patch.astype(np.uint8))
            patch_count += 1
            total_patches_for_ID += 1

        if plaques_only == False:
            image_patch_file = os.path.join(dir_to_save, train_val_test, "images",
                                            ID_tuple[0][:-13] + str(patch_count) + '.npy')
            np.save(image_patch_file, image_patch.astype(np.float32))

            nodule_patch_file = os.path.join(dir_to_save, train_val_test, "masks_1",
                                             ID_tuple[0][:-13] + str(patch_count) + '.npy')

            np.save(nodule_patch_file, nodule_patch.astype(np.uint8))

            lung_patch_file = os.path.join(dir_to_save, train_val_test, "masks_2",
                                           ID_tuple[0][:-13] + str(patch_count) + '.npy')

            np.save(lung_patch_file, lung_patch.astype(np.uint8))

            plaque_patch_file = os.path.join(dir_to_save, train_val_test, "plaques",
                                             ID_tuple[0][:-13] + str(patch_count) + '.npy')

            np.save(plaque_patch_file, dif_patch.astype(np.uint8))
            patch_count += 1
            total_patches_for_ID += 1


def save_all_patch(ID_tuple_list,
                   dir_to_save,
                   patch_shape,
                   stride_size,
                   basic_path,
                   basic_path2,
                   truncate=True,
                   train_val_test_split=[0.8, 0.2, 0.0],
                   plaques_only=True,
                   val_patients=None,
                   test_patients=None):
    # First delete the directory, where we would like to save the patches to avoid naming collisions
    if os.path.exists(dir_to_save):
        shutil.rmtree(dir_to_save)
    # Create parent directory
    os.mkdir(dir_to_save)
    os.mkdir(os.path.join(dir_to_save, "file_logs"))
    # Then create folders train, test, val  containing images and masks folders.
    train_dir, test_dir, val_dir = [os.path.join(dir_to_save, "train"),
                                    os.path.join(dir_to_save, "test"),
                                    os.path.join(dir_to_save, "val")]
    # Create train_dir
    os.mkdir(train_dir)
    os.mkdir(os.path.join(train_dir, "images"))
    os.mkdir(os.path.join(train_dir, "plaques"))
    os.mkdir(os.path.join(train_dir, "masks_1"))
    os.mkdir(os.path.join(train_dir, "masks_2"))

    # Create test_dir
    os.mkdir(test_dir)
    os.mkdir(os.path.join(test_dir, "images"))
    os.mkdir(os.path.join(test_dir, "plaques"))
    os.mkdir(os.path.join(test_dir, "masks_1"))
    os.mkdir(os.path.join(test_dir, "masks_2"))

    # Create val_dir
    os.mkdir(val_dir)
    os.mkdir(os.path.join(val_dir, "images"))
    os.mkdir(os.path.join(val_dir, "plaques"))
    os.mkdir(os.path.join(val_dir, "masks_1"))
    os.mkdir(os.path.join(val_dir, "masks_2"))

    total_number_of_IDs = len(ID_tuple_list)
    # Create thresholds for train-val-test split
    number_of_IDs_train = int(train_val_test_split[0] * total_number_of_IDs)
    number_of_IDs_val = int(train_val_test_split[1] * total_number_of_IDs)
    number_of_IDs_test = int(train_val_test_split[2] * total_number_of_IDs)

    patients = []
    for counter, ID_tuple in tqdm(enumerate(ID_tuple_list)):
        patients.append(ID_tuple[3])
    patients = np.unique(patients)

    random.seed(42)
    if test_patients==None:
        test_patients = random.sample(set(patients).difference(set(val_patients)),
                                  int(len(patients) * train_val_test_split[2]))
    if val_patients==None:
        val_patients = random.sample(set(patients), int(len(patients) * train_val_test_split[1]))



    
    
    # Save images to the corresponding subfolders using the functions above.
    for counter, ID_tuple in tqdm(enumerate(ID_tuple_list)):
        if ID_tuple[3].rstrip('_') in val_patients:
            train_val_test = "val"
        elif ID_tuple[3].rstrip('_') in test_patients:
            train_val_test = "test"
        else:
            train_val_test = "train"
        save_all_patch_for_image_mask_pair(ID_tuple,
                                           patch_shape=patch_shape,
                                           stride_size=stride_size,
                                           truncate=truncate,
                                           dir_to_save=dir_to_save,
                                           train_val_test=train_val_test,
                                           basic_path=basic_path,
                                           basic_path2=basic_path2,
                                           plaques_only=plaques_only)

    return val_patients


epsilon = 1e-5
smooth = 1


def dsc(y_true, y_pred, args):
    smooth = args.smooth
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred, args):
    loss = 1 - dsc(y_true, y_pred, args)
    return loss


def tp(y_true, y_pred, args):
    smooth = args.smooth
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth)
    return tp


def tn(y_true, y_pred, args):
    smooth = args.smooth
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)
    return tn


def fp(y_true, y_pred, args):
    smooth = args.smooth
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    fp = (K.sum(y_neg * y_pred_pos) + smooth) / (K.sum(y_pred) + smooth)
    return fp


def fn(y_true, y_pred, args):
    smooth = args.smooth
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    fn = (K.sum(y_pos * y_pred_neg) + smooth) / (K.sum(y_pred_neg) + smooth)
    return fn


def tversky(y_true, y_pred, args):
    smooth = args.smooth
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = args.alpha
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred, args):
    return 1 - tversky(y_true, y_pred, args)


def focal_tversky(y_true, y_pred, args):
    gamma = args.gamma = 0.75
    pt_1 = tversky(y_true, y_pred, args)
    return K.pow((1 - pt_1), gamma)


def multiloss(y_true, y_pred, args):
    if args.loss_type == 1:
        return focal_tversky(y_true, y_pred, args)
    else:
        return dice_loss(y_true, y_pred, args)


# Visaulization functions

def gray_to_colored(im):
    colored = np.repeat(np.expand_dims(im, axis=-1), 3, axis=-1).astype(float)
    colored = 1 * (colored - np.amin(colored)) / (np.amax(colored) - np.amin(colored))
    return colored


def superimpose_mask(image_array, mask_array, opacity=0.8):
    superimposed = gray_to_colored(image_array)
    reds = np.zeros(mask_array.shape + (3,)).astype(np.bool)
    reds[:, :, 0] = mask_array == 1
    superimposed[reds] = opacity * 1 + (1 - opacity) * superimposed[reds]
    return superimposed


def visualize_slice_mask_pair(image_array, mask_1_array, mask_2_array, plaque_array, opacity=0.8, name=""):
    ax, plots = plt.subplots(2, 4, figsize=(25, 10))
    ax.suptitle(name)
    plots[0, 0].axis('off')
    plots[0, 0].imshow(mask_2_array - mask_1_array, cmap=plt.cm.bone)
    plots[0, 1].axis('off')
    plots[0, 1].imshow(mask_1_array, cmap=plt.cm.bone)
    plots[0, 2].axis('off')
    plots[0, 2].imshow(mask_2_array, cmap=plt.cm.bone)
    plots[0, 3].axis('off')
    plots[0, 3].imshow(plaque_array, cmap=plt.cm.bone)

    plots[1, 0].axis('off')
    plots[1, 0].imshow(superimpose_mask(image_array, mask_2_array - mask_1_array, opacity=opacity))
    plots[1, 1].axis('off')
    plots[1, 1].imshow(superimpose_mask(image_array, mask_1_array, opacity=opacity))
    plots[1, 2].axis('off')
    plots[1, 2].imshow(superimpose_mask(image_array, mask_2_array, opacity=opacity))
    plots[1, 3].axis('off')
    plots[1, 3].imshow(superimpose_mask(image_array, plaque_array, opacity=opacity))
    plt.show()



def expand_as_3d(tensor, rep, name):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4), arguments={'repnum': rep},
                       name='psi_up' + name)(tensor)
    return my_repeat


def AttnGatingBlock3D(x, g, inter_shape, name):
    '''
    Analogous implementation of the 3D attention gate used in the Attention U-Net 3D.
    '''
    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = Conv3D(inter_shape, (2, 2, 2), strides=(2, 2, 2), padding='same', name='xl' + name)(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv3D(inter_shape, (1, 1, 1), padding='same')(g)
    upsample_g = Conv3DTranspose(inter_shape, (3, 3, 3), strides=(
    shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2], shape_theta_x[3] // shape_g[3]), padding='same',
                                 name='g_up' + name)(phi_g)  # 16

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv3D(1, (1, 1, 1), padding='same', name='psi' + name)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling3D(
        size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2], shape_x[3] // shape_sigmoid[3]))(
        sigmoid_xg)  # 32

    upsample_psi = expand_as_3d(upsample_psi, shape_x[4], name)
    y = multiply([upsample_psi, x], name='q_attn' + name)

    result = Conv3D(shape_x[4], (1, 1, 1), padding='same', name='q_attn_conv' + name)(y)
    result_bn = BatchNormalization(name='q_attn_bn' + name)(result)
    return result_bn


def UnetConv3D(input, outdim, is_batchnorm, name):
    '''
    Analogous implementation of the pair of convolutional layers used by the U-Net 3D.
    '''
    x = Conv3D(outdim, (3, 3, 3), strides=(1, 1, 1), kernel_initializer="glorot_normal", padding="same",
               name=name + '_1')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_act')(x)

    x = Conv3D(outdim, (3, 3, 3), strides=(1, 1, 1), kernel_initializer="glorot_normal", padding="same",
               name=name + '_2')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_act')(x)
    return x


def UnetGatingSignal3D(input, is_batchnorm, name):
    '''
    Implementation of the gating signal appearing in the upsampling branch of the Attention U-Net 3D:
    simply a 1x1 convolution followed by batch normalization and ReLU.
    '''
    shape = K.int_shape(input)
    x = Conv3D(shape[4] * 1, (1, 1, 1), strides=(1, 1, 1), padding="same", kernel_initializer="glorot_normal",
               name=name + '_conv')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name=name + '_act')(x)
    return x


def tiny_attn_unet3D(opt, input_size, args):
    '''
    Analogue of the above defined attn_unet3D with less layers, resulting in a smaller U shape.
    '''
    inputs = Input(shape=input_size)
    conv1 = UnetConv3D(inputs, 8*args.kernel_power, is_batchnorm=True, name='conv1')
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    pool1 = Dropout(args.dropout)(pool1)

    conv2 = UnetConv3D(pool1, 8*args.kernel_power, is_batchnorm=True, name='conv2')
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    pool2 = Dropout(args.dropout)(pool2)

    center = UnetConv3D(pool2, 16*args.kernel_power, is_batchnorm=True, name='center')

    g3 = UnetGatingSignal3D(center, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock3D(conv2, g3, 8*args.kernel_power, '_3')
    up3 = concatenate([Conv3DTranspose(8*args.kernel_power, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu',
                                       kernel_initializer="glorot_normal")(center), attn3], name='up3')

    up4 = concatenate([Conv3DTranspose(8*args.kernel_power, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu',
                                       kernel_initializer="glorot_normal")(up3), conv1], name='up4')

    mask_1 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='mask_1')(up4)
    mask_2 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='mask_2')(up4)
    dif = Conv3D(1, (1, 1, 1), activation='sigmoid', name='dif')(up4)

    model = Model(inputs=[inputs], outputs=[mask_1, mask_2, dif])
    model.compile(optimizer=opt,
                  loss=[wrapped_partial(dice_loss, args=args), wrapped_partial(dice_loss, args=args),
                        wrapped_partial(multiloss, args=args)],
                  loss_weights=[0.1, 0.1, 0.8],
                  metrics=[[wrapped_partial(dsc, args=args)], [wrapped_partial(dsc, args=args)],
                           [wrapped_partial(dsc, args=args), wrapped_partial(tp, args=args),
                            wrapped_partial(tn, args=args)]])
    return model




def full_ct_model_evaluation(image, model, z_stride, which_prediction):
    # Shape of original images
    size_X = image.shape[2]
    size_Y = image.shape[1]
    size_Z = image.shape[0]

    image_paded = np.zeros((size_Z + 24,
                            size_Y,
                            size_X))

    image_paded[:size_Z, :size_Y, :size_X] = image / 512

    prediction_array = np.zeros((size_Z + 24,
                                 size_Y,
                                 size_X))

    coverage_array = np.zeros((size_Z + 24,
                               size_Y,
                               size_X))

    # Containers for batch predictions
    patch_boundaries_list = []
    counter = 0

    # Stride along Z axis:  
    for z_start in range(0, prediction_array.shape[2], z_stride):
        z_end = z_start + 24
        if (np.count_nonzero(image[z_start:z_end, :, :]) > 1):
            patch_boundaries_list.append([z_start, z_end])
    for patch_index in range(0, len(patch_boundaries_list)):
        # patch_boundaries in current batch
        temporal_boundaries = patch_boundaries_list[patch_index]
        temp_patches = []
        # Extracting patches for prediction
        current_patch = image_paded[temporal_boundaries[0]:temporal_boundaries[1],
                        16:48,
                        16:48]
        current_patch = np.expand_dims(current_patch, axis=0)
        # Updating prediction_array and coverage_array
        predicted_patch = model.predict(np.expand_dims(current_patch, axis=-1))

        # 0 belső maszk 1 külső maszk 2 differencia

        prediction = predicted_patch[which_prediction]

        prediction = np.reshape(prediction, [24, 32, 32])

        prediction_array[temporal_boundaries[0]:temporal_boundaries[1],
        16:48,
        16:48] += prediction

        # print(prediction_array[32, 32, 32])

        coverage_array[temporal_boundaries[0]:temporal_boundaries[1],
        :,
        :] += 1

    coverage_array = np.maximum(coverage_array, np.ones(coverage_array.shape))
    # Taking the average prediction value for the pixels
    prediction_array = np.divide(prediction_array, coverage_array)
    # print(prediction_array[32,32,32])
    # Removing the prediction values outside of the CT  
    prediction_array = prediction_array[0:size_Z, 0:size_Y, 0:size_X]

    # The average prediction value is continuous between 0 and 1,   
    # so for the segmentation we have to threshold it   
    prediction_array = (prediction_array > 1 / 2) * 1

    return prediction_array



def mask_from_dicom(contour_file_name):
    ds = pydicom.dcmread(contour_file_name)
    pixels = np.array(ds.pixel_array)
    pixels = np.where(pixels > 0, 1, 0)
    return pixels


def read_in(file_path, patient, vessel):
    ds = pydicom.dcmread(os.path.join(file_path, patient + '_' + vessel + '_NoContour.dcm'))
    pixelSpacing = ds.PixelSpacing
    sliceSpacing = ds.SpacingBetweenSlices
    Image = np.array(ds.pixel_array)
    Mask_1 = get_all_plaques(mask_from_dicom(os.path.join(file_path, patient + '_' + vessel + '_Contour1.dcm')))
    Mask_2 = get_all_plaques(mask_from_dicom(os.path.join(file_path, patient + '_' + vessel + '_Contour2.dcm')))
    Diff = Mask_2 - Mask_1
    originalds = pydicom.dcmread(os.path.join(file_path, patient, vessel, 'volume.dcm'))
    oripixelSpacing = originalds.PixelSpacing
    orisliceSpacing = originalds.SpacingBetweenSlices
    return Image, Diff, Mask_1, Mask_2, pixelSpacing, sliceSpacing, oripixelSpacing, orisliceSpacing


def xml_parse(root):
    itemlist = []
    for child in root:
        itemlist.append(np.array(child[1].text.split(' ')).reshape(4, 4))
        return itemlist


def xml_load(file):
    root = ET.parse(file).getroot()
    return root


def xml_parse_path(root):
    itemlist = []
    for child in root:
        itemlist.append(np.array(child))
        return itemlist


def dice_coefficient(mask1, mask2):
    # print(np.sum(mask1))
    return ((2.0 * np.sum(mask1 * mask2)) / (np.sum(mask1) + np.sum(mask2)))



def filter_supp2(filename):
    if '.dcm' in filename:
        return True
    else:
        return False


def get_pairs2(vesel_list):
    return_list = []
    for ID in vesel_list:
        if ID.endswith('NoContour.dcm'):
            return_list.append([ID[:-14]])
    return return_list

class CyclicLR(Callback):
    """Copied from https://github.com/bckenstler/CLR/
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())
