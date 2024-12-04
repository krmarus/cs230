import pydicom
import numpy as np
import tensorflow as tf
import os
import wget
from typing import List, Union
import zipfile
import SimpleITK as sitk
import nibabel as nib
import pydicom
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_confusion_matrix(dataset, loaded_model, save_figure=False, figname=''):
    data_list, label_list = [], []
    for d, l in dataset:
        data_list.append(d.numpy())
        label_list.append(l.numpy())

    # Convert lists to tensors
    data_tensor = tf.convert_to_tensor(data_list)
    labels_tensor = tf.convert_to_tensor(label_list)

    predictions = loaded_model.predict(data_tensor)

    predictions = tf.argmax(predictions, axis=-1)
    labels = tf.reshape(labels_tensor, [-1])
    predictions = tf.reshape(predictions, [-1])
    # Compute confusion matrix
    cm = tf.math.confusion_matrix(
        labels,
        predictions,
        num_classes=9
    ).numpy()

    # Plot the confusion matrix
    # figure = plot_confusion_matrix(cm, class_names=[f"Class {i}" for i in range(9)])
    class_names=[f"Class {i}" for i in range(9)]
    class_names=["Background", "L Psoas", "R Psoas", "L Paraspinals", "R Paraspinals", "L Laterals", "R Laterals", 'L Rect. Abdominis', "R Rect. Abdominis"]
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Annotate cells with values
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]:.2f}",
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    if save_figure:
        plt.savefig(figname, dpi=300, bbox_inches='tight')

def display_outputs(display_list, **kwargs):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        # plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.imshow(display_list[i])
        plt.axis('off')
        plt.tight_layout()

    print(kwargs.keys())
    if 'save_figure' in kwargs and kwargs['save_figure']:
        figname = kwargs['figname']
        print(f'saving figure {figname}')
        plt.savefig(kwargs['figname'], dpi=300, bbox_inches='tight')

    
def show_predictions(model, dataset=None, num=1, **kwargs):
    """
    Displays the first image of each of the num batches
    """
    if dataset:

        for image, mask in dataset.take(num):
            # print(image[0].shape)
            pred_mask = model.predict(image, batch_size=1)
            print(create_mask(pred_mask).shape)
            display_outputs([image[0,:,:,0], mask[0], create_mask(pred_mask)[0]], **kwargs)

def worst_per_examples(model, dataset, save_figure=False, figname=''):
    # features, labels = dataset
    data_list, label_list = [], []
    for d, l in dataset:
        data_list.append(d.numpy())
        label_list.append(l.numpy())

    # Convert lists to tensors
    data_tensor = tf.convert_to_tensor(data_list)
    labels_tensor = tf.convert_to_tensor(label_list)
    predictions = model.predict(data_tensor)
    
    loss_fn = tf.keras.losses.get(model.loss)
    # print(len(data_list))
    example_losses = np.zeros((len(data_list)))
    for i, (label, prediction) in enumerate(zip(labels_tensor,predictions)):
        # print(i)
        example_losses[i] = loss_fn(tf.expand_dims(label,axis=0), tf.expand_dims(prediction,axis=0)).numpy()
    
    worst_indices = np.argsort(-example_losses)  # Sort in descending order

    # Optionally: Get the worst N examples (e.g., top 5)
    top_n = 5
    worst_n_indices = worst_indices[:top_n]
    worst_n_losses = example_losses[worst_n_indices]
    print('worst 5 losses', worst_n_losses)
    for i in range(5):
        ds_to_plot = tf.data.Dataset.from_tensors(((tf.expand_dims(data_tensor[worst_n_indices[i]], axis=0)), \
                                 tf.expand_dims(labels_tensor[worst_n_indices[i]],axis=0)))
        show_predictions(model, ds_to_plot, \
                                num=1, save_figure=save_figure, figname=figname+f'_{i}')

## Function from Comp2Comp 
def parse_windows(windows):
    """Parse windows provided by the user.

    These windows can either be strings corresponding to popular windowing
    thresholds for CT or tuples of (upper, lower) bounds.

    Args:
        windows (list): List of strings or tuples.

    Returns:
        list: List of tuples of (upper, lower) bounds.
    """
    windowing = {
        "soft": (400, 50),
        "bone": (1800, 400),
        "liver": (150, 30),
        "spine": (250, 50),
        "custom": (500, 50),
    }
    vals = []
    for w in windows:
        if len(w) == 2:
            assert_msg = "Expected tuple of (lower, upper) bound"
            assert len(w) == 2, assert_msg
            assert isinstance(w[0], (float, int)), assert_msg
            assert isinstance(w[1], (float, int)), assert_msg
            assert w[0] < w[1], assert_msg
            vals.append(w)
            continue

        if w not in windowing:
            raise KeyError("Window {} not found".format(w))
        window_width = windowing[w][0]
        window_level = windowing[w][1]
        upper = window_level + window_width / 2
        lower = window_level - window_width / 2

        vals.append((lower, upper))

    return tuple(vals)

## Apply Windowing to CT Images, Stack Images

def _window(xs, bounds):
    """Apply windowing to an array of CT images.

    Args:
        xs (ndarray): NxHxW
        bounds (tuple): (lower, upper) bounds

    Returns:
        ndarray: Windowed images.
    """

    imgs = []
    for lb, ub in bounds:
        imgs.append(np.clip(xs, a_min=lb, a_max=ub))

    if len(imgs) == 1:
        return imgs[0]
    elif xs.shape[-1] == 1:
        return np.concatenate(imgs, axis=-1)
    else:
        return np.stack(imgs, axis=-1)

def find_model_weights(file_name, model_dir):
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.startswith(file_name):
                filename = os.path.join(root, file)
    return filename

def download_muscle_adipose_tissue_model(model_dir):
    download_dir = Path(
        os.path.join(
            model_dir,
            ".totalsegmentator/nnunet/results/nnUNet/2d/Task927_FatMuscle/nnUNetTrainerV2__nnUNetPlansv2.1",
        )
    )
    all_path = download_dir / "all"
    if not os.path.exists(all_path):
        download_dir.mkdir(parents=True, exist_ok=True)
        wget.download(
            "https://huggingface.co/stanfordmimi/multilevel_muscle_adipose_tissue/resolve/main/all.zip",
            out=os.path.join(download_dir, "all.zip"),
        )
        with zipfile.ZipFile(os.path.join(download_dir, "all.zip"), "r") as zip_ref:
            zip_ref.extractall(download_dir)
        os.remove(os.path.join(download_dir, "all.zip"))
        wget.download(
            "https://huggingface.co/stanfordmimi/multilevel_muscle_adipose_tissue/resolve/main/plans.pkl",
            out=os.path.join(download_dir, "plans.pkl"),
        )
        print("Muscle and adipose tissue model downloaded.")
    else:
        print("Muscle and adipose tissue model already downloaded.")

## Function from Coursera -- Using this
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask#[0]


#### Functions from Comp2Comp, not currently used ####
def preds_to_mask(ys):
    labels = np.zeros_like(ys, dtype=np.uint8)
    l_argmax = np.argmax(ys, axis=-1)
    for c in range(labels.shape[-1]):
        labels[l_argmax == c, c] = 1
    # plt.figure()
    # plt.imshow(l_argmax[0,:,:],cmap='gray')
    # plt.colorbar()
    return labels.astype(bool)

def _fill_holes(mask,mask_id):
        int_mask = ((1 - mask) > 0.5).astype(np.int8)
        components, output, stats, _ = cv2.connectedComponentsWithStats(
            int_mask, connectivity=8
        )
        sizes = stats[1:, -1]
        components = components - 1
        # Larger threshold for SAT
        # TODO make this configurable / parameter
        if mask_id == 2:
            min_size = 50
            # min_size = 0
        else:
            min_size = 5
            # min_size = 0
        img_out = np.ones_like(mask)
        for i in range(0, components):
            if sizes[i] > min_size:
                img_out[output == i + 1] = 0
        return img_out

def fill_holes(ys):
    segs = []
    for n in range(len(ys)):
        ys_out = [
            _fill_holes(ys[n][..., i], i) for i in range(ys[n].shape[-1])
        ]
        segs.append(np.stack(ys_out, axis=2).astype(float))

    return segs

def process_path_C2C(image_path, mask_path):
    image_path = image_path.numpy().decode("utf-8")
    mask_path = mask_path.numpy().decode("utf-8")
    img = pydicom.read_file(image_path, force=True)
    img = (img.pixel_array + int(img.RescaleIntercept)).astype("float32")
    mask = pydicom.read_file(mask_path, force=True)
    mask = (mask.pixel_array + int(mask.RescaleIntercept)).astype("float32")

    return img, mask

def tf_process_path_C2C(image_path, mask_path):
    return tf.py_function(process_path_C2C, [image_path, mask_path], [tf.float32, tf.float32])

def preprocess_C2C(image, mask, windows=['soft','bone','custom']):
    # windows = ['soft','bone','custom']
    window_imgs = _window(image, parse_windows(windows))
    return window_imgs, mask

def tf_preprocess_C2C(image, mask):
    return tf.py_function(preprocess_C2C, [image, mask], [tf.float32, tf.float32])


def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_coef_weighted_all_classes(y_true, y_pred, numLabels=9):
    y_true_one_hot = tf.one_hot(tf.cast(y_true,tf.uint8), depth=numLabels, axis=-1)
    dice_weighted_all_classes = []
    sum_weights = 0.
    for index in range(numLabels):
        # dice -= dice_coef(y_true_one_hot[:,:,:,index], y_pred[:,:,:,index])
        weight = 1/ tf.keras.backend.sum(tf.keras.backend.flatten(y_true_one_hot[:,:,:,index]))
        dice = dice_coef(y_true_one_hot[:,:,:,index], y_pred[:,:,:,index])
        dice_weighted_all_classes.append(dice * weight)
        sum_weights += weight
    return dice_weighted_all_classes / sum_weights
         

def dice_coef_multilabel(y_true, y_pred, numLabels=9):
    dice_weighted_all_classes = dice_coef_weighted_all_classes(y_true, y_pred, numLabels)
    dice = - tf.reduce_sum(dice_weighted_all_classes)
    return dice


def dice_multilabel_results(y_true, y_pred, numLabels=9):
    dice = np.zeros(numLabels)
    y_true_one_hot =  tf.one_hot(tf.cast(y_true,tf.uint8), depth=numLabels, axis=-1)
    for index in range(numLabels):
        # dice -= dice_coef(y_true_one_hot[:,:,:,index], y_pred[:,:,:,index])
        # weight = 1/ tf.keras.backend.sum(tf.keras.backend.flatten(y_true_one_hot[:,:,:,index]))
        dice[index] =  dice_coef(y_true_one_hot[:,:,:,index], y_pred[:,:,:,index])
    
    return dice
        
def average_dice(model, dataset):
    # features, labels = dataset
    data_list, label_list = [], []
    for d, l in dataset:
        data_list.append(d.numpy())
        label_list.append(l.numpy())

    # Convert lists to tensors
    data_tensor = tf.convert_to_tensor(data_list)
    labels_tensor = tf.convert_to_tensor(label_list)
    predictions = model.predict(data_tensor)
    
    example_dices = np.zeros((len(data_list),9))
    for i, (label, prediction) in enumerate(zip(labels_tensor,predictions)):
        example_dices[i,:] = dice_multilabel_results(tf.expand_dims(label,axis=0), tf.expand_dims(prediction,axis=0))
    
    return example_dices
    

def plot_boxplots(model, datasets, labels, modelname, save_figure=False, figname=None):
    class_names=[f"Class {i}" for i in range(9)]
    class_names=["Background", "L Psoas", "R Psoas", "L Paraspinals", "R Paraspinals", "L Laterals", "R Laterals", 'L Rect. Abdominis', "R Rect. Abdominis"]

    dice_all_datasets = []
    for i in range(3):
        dice_dataset = average_dice(model, datasets[i])
        dice_dataset_pd = pd.DataFrame(dice_dataset, columns=class_names)
        dice_dataset_pd['Type'] = labels[i]
        dice_all_datasets.append(dice_dataset_pd)

    dice_examples = pd.concat(dice_all_datasets, ignore_index=True)
    data_melted = dice_examples.melt(id_vars='Type', var_name='Class', value_name='Dice Score [0 to 1]')

    sns.boxplot(data=data_melted[data_melted['Class'] != 'Background'], x='Class',y='Dice Score [0 to 1]',hue='Type', showfliers=False)
    plt.xticks(rotation=45)
    plt.title(f'model_{modelname}')
    plt.ylim([0.775, 1.0])
    plt.tight_layout()
    if save_figure:
        if figname is None:
            figname = f'./figures/{modelname}_dice_per_muscle'
        plt.savefig(figname, dpi=300, bbox_inches='tight')
