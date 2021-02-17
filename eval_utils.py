'''
Filename: eval_utils.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 02/03/2021
Description: Various utility functions used for evaluating performance
    of the model on detection.
'''

import cv2
import numpy as np
import pandas as pd

def pytorch_resize(image, min_side=608, max_side=1024):
    """
    Resizes and outputs the image and scale.
    Adopted from https://github.com/yhenon/pytorch-retinanet.
    """
    # Pull out shape of the image.
    rows, cols, cns = image.shape
    # Find the smaller side.
    smallest_side = min(rows, cols)
    # Define scale based on smallest side.
    scale = min_side / smallest_side

    # Check if larger side is now greater than max_side.
    # Can happen when images have a large aspect ratio.
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # Resize the image with the computed scale
    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    return image, scale

def draw_box(image, box, color, thickness=2):
    """
    Draw a bounding box on the image.

    Parameters
    ----------
    image : ndarray
        the image to draw on
    box : list, [x1, y1, x2, y2]
        top left and bottom right coordinates of the bounding box
    color : list, [B,G,R]
        color to make the box
    thickness : int
        thickness/width of the bounding box lines
    """
    box = np.array(box).astype(int)
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness, cv2.LINE_AA)

def draw_caption(image, box, caption, loc=''):
    """
    Write a caption above or below the bounding box.

    Parameters
    ----------
    image : ndarray
        the image to write on
    box : list, [x1, y1, x2, y2]
        top left and bottom right coordinates of the bounding box
    caption : str
        text string to write on image
    """
    box = np.array(box).astype(int)
    if loc == 'bottom':
        cv2.putText(image, caption, (box[0], box[3] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (box[0], box[3] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    else:
        cv2.putText(image, caption, (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (box[0], box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def get_bounding_boxes(patient_nm, anno_df=None, info_loc=None):
    """
    Extracts the bounding box locations of the specific patient from the annotations file.

    Parameters
    ----------
    patient_nm : str
        name of the image file (patientid_#.png)
    df : DataFrame
        DataFrame containing annotation information
    info_loc : str
        path/location of the annotation file

    Returns
    -------
    boxes : list
        list of lists of bounding box information in the order [x1, y1, x2, y2]
    """
    # Load in the DL_info.csv file
    if info_loc is not None:
        frac_info = pd.read_csv(info_loc, names=['ID', 'x1', 'y1', 'x2', 'y2', 'class'])
    else:
        frac_info = anno_df

    # Make an empty list to append possible bounding boxes to
    boxes = []

    # Make a subset DataFrame with just current image information
    subset_df = frac_info[frac_info['ID'].str.contains(patient_nm[patient_nm.rfind('/')+1:])]

    # Extract the window values
    for _, row in subset_df.iterrows():
        boxes.append((row['x1'], row['y1'], row['x2'], row['y2']))

    return boxes

def intersection_over_union(predict_box, truth_box):
    """
    Computes the intersection-over-union of two bounding boxes.

    Parameters
    ----------
    predict_box : list, [x1, y1, x2, y2]
        the predicted bounding box
    truth_box : list, [x1, y1, x2, y2]
        the ground truth bounding box

    Returns
    -------
    iou : float
        the calculated intersection-over-union score
    """
    # Find the coordinates of the intersection rectangle
    int_x1 = max(predict_box[0], truth_box[0])
    int_y1 = max(predict_box[1], truth_box[1])
    int_x2 = min(predict_box[2], truth_box[2])
    int_y2 = min(predict_box[3], truth_box[3])

    # Compute area of the intersection rectangle
    int_area = max(0, int_x2 - int_x1) * max(0, int_y2 - int_y1)

    # Compute area of predicted and truth boxes
    pred_area = (predict_box[2] - predict_box[0]) * (predict_box[3] - predict_box[1])
    truth_area = (truth_box[2] - truth_box[0]) * (truth_box[3] - truth_box[1])

    # Calculate the intersection over union
    iou = int_area / float(pred_area + truth_area - int_area)

    return iou

def calc_performance(predictions, truths, iou_threshold=0.50):
    """
    Calculate how well the model performs at predicting the correct
    bounding boxes. Performance is measured in terms of how many
    true positives, false negatives, and false positives the model outputs.

    Parameters
    ----------
    predictions : list
        list of lists of bounding boxes predicted by the model
    truths : list
        list of lists of ground truth bounding boxes
    iou_threshold : float
        IOU value to be considered true positive

    Returns
    -------
    true_pos : int
        number of true positives
    false_pos : int
        number of false positives
    false_neg : int
        number of false negatives
    ious : list
        list of intersection-over-union values for each box pair
    """
    # Initialize output values
    true_pos = 0
    false_pos = 0
    false_neg = 0
    ious = []

    if len(predictions) == 0: # If no predictions, count all false negatives
        false_neg = len(truths)
    else:
        for box in predictions:
            iou = 0
            for truth in truths:
                temp_iou = intersection_over_union(box, truth)
                if temp_iou > iou_threshold:
                    iou = temp_iou
            if iou == 0: # If no IoUs > 0.50, count as false positives
                false_pos += 1
            else:
                ious.append(iou)
                true_pos += 1
        # Add to false negative count if truth box has no overlaps with predictions
        for truth in truths:
            iou = 0
            for box in predictions:
                temp_iou = intersection_over_union(box, truth)
                if temp_iou > iou_threshold:
                    iou = temp_iou
            if iou == 0:
                false_neg += 1

    return true_pos, false_pos, false_neg, ious

def calc_cohens(true_pos, false_pos, false_neg, true_neg):
    """
    Calculate Cohen's Kappa between two reads done by radiologists.

    Parameters
    ----------
    true_pos : int
        number of true positives between reads (agreements)
    false_pos : int
        number of false positives between reads (in read 2 but not 1)
    false_neg : int
        number of false negatives between reads (in read 1 but not 2)
    true_neg : int
        number of true negatives between reads (in neither read 1 or 2)

    Returns
    -------
    coh_kappa : float
        Cohen's Kappa value for the two reads
    """
    # Sum the total number of events
    sum_events = float(true_pos + false_pos + false_neg + true_neg)

    # Calculate observed proportionate agreement
    obs_agree = (true_pos + true_neg) / sum_events

    # Calculate probability of randomly marking
    prop_yes = ((true_pos + false_neg) / sum_events) * ((true_pos + false_pos) / sum_events)

    # Calculate probability of randomly not marking
    prop_no = ((true_neg + false_pos) / sum_events) * ((true_neg + false_neg) / sum_events)

    # Add prop_yes and prop_no for probability of random agreement
    prop_e = prop_yes + prop_no

    # Calculate Cohen's Kappa
    coh_kappa = (obs_agree - prop_e) / (1 - prop_e)

    return coh_kappa

def calc_kappa_fr(true_pos, false_pos, false_neg):
    """
    Calculate Free-Reponse Kappa between two reads done by radiologists.
    Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5395923/

    Parameters
    ----------
    true_pos : int
        number of true positives between reads (agreements)
    false_pos : int
        number of false positives between reads (in read 2 but not 1)
    false_neg : int
        number of false negatives between reads (in read 1 but not 2)

    Returns
    -------
    fr_kappa : float
        Free-response Kappa for the dataset
    """
    # Calculate the free-response Kappa
    fr_kappa = (2 * true_pos) / (false_pos + false_neg + 2 * true_pos)

    return fr_kappa

def calc_bbox_area(box):
    """
    Calculates the pixel area of the bounding box.

    Parameters
    ----------
    box : list, [x1, y1, x2, y2]
        the annotated/predicted bounding box

    Returns
    -------
    bbox_area : int
        area of the bounding box
    """
    # Calculate height and width of the bounding box
    bbox_height = box[3] - box[1]
    bbox_width = box[2] - box[0]

    # Calculate area of the bounding box
    bbox_area = bbox_height * bbox_width

    return bbox_area
