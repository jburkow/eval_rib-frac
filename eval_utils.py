'''
Filename: eval_utils.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 05/18/2021
Description: Various utility functions used for evaluating performance
    of the model on detection.
'''

import sys
import cv2
import numpy as np
import pandas as pd
from scipy.integrate import simps


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


def get_bounding_boxes(patient_nm, anno_df=None, info_loc=None, has_probs=False, conf_threshold=0.00):
    """
    Extracts the bounding box locations of the specific patient from the annotations file.

    Parameters
    ----------
    patient_nm : str
        name of the image file (patientid_#.png)
    anno_df : DataFrame
        DataFrame containing annotation information
    info_loc : str
        path/location of the annotation file
    has_probs : bool
        whether or not DataFrame includes probabilities (for model predictions)
    conf_threshold : float
        probability/confidence threshold of model bounding box predictions to keep

    Returns
    -------
    boxes : list
        list of lists of bounding box information in the order [x1, y1, x2, y2]
    probs : list
        list of probabilities/confidence of model predicted bounding boxes
    """
    # Load in the DL_info.csv file
    if info_loc is not None:
        frac_info = pd.read_csv(info_loc, names=['ID', 'x1', 'y1', 'x2', 'y2', 'class'])
    else:
        frac_info = anno_df

    # Make a subset DataFrame with just current patient information
    subset_df = frac_info[frac_info['ID'].str.contains(patient_nm.split('/')[-1], case=False)]

    # If any values are NaN (i.e., there are no fractures labeled), return empty lists
    if subset_df.isnull().values.any():
        return ([],[]) if has_probs else []

    # Make an empty list to append possible bounding boxes to
    boxes = []

    if has_probs:
        # Mask out rows with model confidence values below given threshold
        subset_df = subset_df[subset_df.Prob >= conf_threshold]

        # If the resultant DataFrame is empty, return empty lists
        if len(subset_df) == 0:
            return ([],[]) if has_probs else []

        # Loop through DataFrame to pull out bounding boxes and corresponding probabilities
        probs = []
        for _, row in subset_df.iterrows():
            boxes.append((row['x1'], row['y1'], row['x2'], row['y2']))
            probs.append(row['Prob'])

        return boxes, probs

    # If no probabilities, just return boxes
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

    # Calculate percent overlap (percent of ground truth box overlapped by prediction box)
    overlap = int_area / float(truth_area)

    # Calculate the intersection over union
    iou = int_area / float(pred_area + truth_area - int_area)

    return iou, overlap


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
    overlaps : list
        list of overlap values for each box pair
    """
    # Initialize output values
    true_pos = 0
    false_pos = 0
    false_neg = 0
    true_neg = 0
    ious = []
    overlaps = []

    # If neither ground truth or model has fractures, return a true negative
    if len(predictions) == 0 and len(truths) == 0:
        true_neg = 1
        return true_pos, false_pos, false_neg, true_neg, ious, overlaps

    # No model predictions but ground truths, return all as false negatives
    if len(predictions) == 0 and len(truths) > 0:
        false_neg = len(truths)
        return true_pos, false_pos, false_neg, true_neg, ious, overlaps

    # No ground truths but model has fractures, return all as false positives
    if len(truths) == 0 and len(predictions) > 0:
        false_pos = len(predictions)
        return true_pos, false_pos, false_neg, true_neg, ious, overlaps

    # Calculate true positive, false positive, and false negatives if
    # both model predicted fractures and ground truth has labels
    for box in predictions:
        iou = 0
        overlap = 0
        for truth in truths:
            temp_iou, temp_overlap = intersection_over_union(box, truth)
            if temp_iou > iou_threshold:
                iou = temp_iou if temp_iou > iou else iou
                overlap = temp_overlap if temp_overlap > overlap else overlap
        if iou == 0: # If no IoUs > iou_threshold, count as false positives
            false_pos += 1
        else:
            ious.append(iou)
            overlaps.append(overlap)
            true_pos += 1
    # Add to false negative count if truth box has no overlaps with predictions
    for truth in truths:
        iou = 0
        for box in predictions:
            temp_iou, _ = intersection_over_union(box, truth)
            if temp_iou > iou_threshold:
                iou = temp_iou
        if iou == 0:
            false_neg += 1

    return true_pos, false_pos, false_neg, true_neg, ious, overlaps


def calc_metric(true_pos, false_pos, false_neg, true_neg, metric='accuracy'):
    """
    Given the values from a confusion matrix, calculate various evaluation metrics.

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
    metric : str
        metric to calculate; acceptable strings:
        ('accuracy', 'precision', 'positive_predictive_value', 'recall', 'sensitivity', 'f1_score',
         'true_positive_rate', 'false_positive_rate', 'cohens_kappa', 'kappa_fr')

    Notes
    -----
    Wikipedia page for metric names and formulas: https://en.wikipedia.org/wiki/Confusion_matrix

    Free-response Kappa: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5395923/
    """
    if metric not in ('accuracy', 'precision', 'positive_predictive_value', 'recall', 'sensitivity',
                      'f1_score', 'f2_score', 'true_positive_rate', 'false_positive_rate', 'cohens_kappa', 'kappa_fr'):
        raise ValueError('Unsupported metric string provided in calc_metric: ' + metric)

    if metric == 'accuracy':
        return (true_pos + true_neg) / float(true_pos + false_pos + false_neg + true_neg)

    if metric in ('precision', 'positive_predictive_value'):
        return true_pos / float(true_pos + false_pos)

    if metric in ('recall', 'sensitivity', 'true_positive_rate'):
        return true_pos / float(true_pos + false_neg)

    if metric == 'f1_score':
        precision = true_pos / float(true_pos + false_pos)
        recall = true_pos / float(true_pos + false_neg)
        return 2 * (precision * recall) / (precision + recall)

    if metric == 'f2_score':
        precision = true_pos / float(true_pos + false_pos)
        recall = true_pos / float(true_pos + false_neg)
        return (1 + 2**2) * (precision * recall) / (2**2 * precision + recall)

    if metric == 'false_positive_rate':
        return false_pos / float(false_pos + true_neg)

    if metric == 'cohens_kappa':
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

    if metric == 'kappa_fr':
        return 2 * true_pos / float(false_pos + false_neg + 2 * true_pos)

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

def calc_auc(afroc_df):
    """
    Calculates the AUC of the AFROC curve for evaluating model performance.

    Parameters
    ----------
    afroc_df : DataFrame
        DataFrame that has LLF and FPR values for each threshold

    Returns
    -------
    auc : float
        area under the AFROC curve
    """
    # Calculate AUC using Simpson's Rule
    auc = simps(afroc_df['LLF'], afroc_df['Threshold'])

    # Backup AUC calculation if Simpson's Rule fails
    if not auc >= 0.0 and not auc <= 1.0:
        auc = 0
        for i in range(len(afroc_df) - 1):
            dx = afroc_df['Threshold'].iloc[i+1] - afroc_df['Threshold'].iloc[i]
            auc += dx * (afroc_df['LLF'].iloc[i+1] + afroc_df['LLF'].iloc[i]) / 2

    return auc

def calc_mAP(preds, annots, iou_threshold=0.3):
    """
    Evaluates detector predictions by mean average precision (mAP) at an IOU threshold. Adapted from https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/csv_eval.py.

    Parameters
    ----------
    preds : DataFrame
        DataFrame containing detector predicted boxes and scores (output of `output_model_predictions.py`)
    annots : DataFrame
        DataFrame containing ground truth boxes
    iou_threshold : float
        Intersection over union (IOU) threshold used for mAP. If a predicted box has IOU >= iou_threshold, then it is a true positive.

    Returns
    -------
    """
    # Add label column for fracture class (to conform to "all_annotations" desired format)
    annots['label'] = 0

    # Gather annotations in desired format
    img_paths = list(set(annots['ID']))

    all_annotations = [[None for i in range(1)] for j in range(len(img_paths))]
    for i, img_path in enumerate(img_paths):
        sub_annots = annots[annots['ID'] == img_path]

        all_annotations[i][0] = sub_annots[['x1', 'y1', 'x2', 'y2', 'label']].values
    
    # Gather detections in desired format
    all_detections = [[None for i in range(1)] for j in range(len(img_paths))]
    for i, img_path in enumerate(img_paths):
        sub_preds = preds[preds['ID'] == img_path]

        all_detections[i][0] = sub_preds[['x1', 'y1', 'x2', 'y2', 'Prob']].values

    average_precisions = {}

    for label in range(1):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(img_paths)):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    try:
        mAP = average_precisions[0][0]

        return mAP
    except:
        print('Some precision/recall/AP issue')
        return 0

def compute_overlap(a, b):
    """
    Compute pairwise IOUs between two lists of boxes. Code from https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/csv_eval.py.

    Parameters
    ----------
    a : ndarray (float)
        Numpy array of boxes of shape (N, 4)
    b : ndarray (float)
        Numpy array of boxes of shape (K, 4)

    Returns
    -------
    overlaps : ndarray (float)
        Numpy array of all pairwise IOUs between boxes in a and b of shape (N, K)
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def _compute_ap(recall, precision):
    """
    Compute average precision given recall and precision curves. Code from https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/csv_eval.py.

    Parameters
    ----------
    recall : list (float)
        list of recall values at each threshold
    precision : list (float)
        list of precision values at each threshold
        
    Returns
    -------
    ap : float
        Average precision
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
