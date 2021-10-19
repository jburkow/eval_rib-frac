'''
Filename: compare_reads.py
Author: Jonathan Burkow (burkowjo@msu.edu), Michigan State University
Last Updated: 10/18/2021
Description: Goes through two separate radiologist read annotation files
    and either creates images with annotations drawn on, or calculates
    a Kappa metric across the dataset.
'''

import argparse
import os
import time
from typing import Iterable, Union, Optional
import numpy as np
import pandas as pd
import cv2

from tqdm import tqdm
from sklearn.metrics import auc
from scipy.linalg import block_diag

from args import ARGS
from plot_utils import draw_box
from eval_utils import (get_bounding_boxes, calc_performance, calc_performance_OLD,
                        calc_bbox_area, calc_mAP, intersection_over_union, MetricsConfMatrix)


def make_images(first_reads: pd.DataFrame,
                second_reads: pd.DataFrame,
                im_path: Optional[str] = None,
                save_dir: Optional[str] = None) -> None:
    """
    Creates new image files with drawn-on annotations from both reads.

    Parameters
    ----------
    first_reads  : DataFrame with image and bounding box locations from the first radiologist reads
    second_reads : DataFrame with image and bounding box locations from the second radiologist reads
    im_path      : path containing images to draw annotations on
    save_dir     : path to save images with drawn-on annotations
    """
    # Define list of images
    # img_list = [os.path.join(root, file) for root, _, files in os.walk(im_path) for file in files] # For annotating all images
    img_list = first_reads['ID'].unique() # Annotating only a test set

    # Loop through original/cropped images, draw annotations, and save JPEGs
    for _, img_nm in tqdm(enumerate(img_list), desc='Saving Annotated Images', total=len(img_list)):
        # Pull the Patient ID from the annotations file
        patient_id = img_nm.split('/')[-1].replace('.png', '')

        # Set temp DataFrames to pull annotations from
        patient_read1 = first_reads[first_reads['ID'].str.contains(patient_id, case=False)]
        patient_read2 = second_reads[second_reads['ID'].str.contains(patient_id, case=False)]

        # Import image
        img = cv2.imread(img_nm)

        # Loop through first read annotations and draw boxes
        for row in patient_read1.itertuples():
            if not pd.isnull(row[-1]):
                box = [int(row[2]), int(row[3]), int(row[4]), int(row[5])]
                draw_box(img, box, [255, 255, 0], thickness=4) # cv2 saves JPG as BGR -> this is teal

        # Loop through second read annotations and draw boxes
        for row in patient_read2.itertuples():
            if not pd.isnull(row[-1]):
                box = [int(row[2]), int(row[3]), int(row[4]), int(row[5])]
                draw_box(img, box, [0, 255, 255], thickness=4) # cv2 saves JPG as BGR -> this is yellow

        # Save image to file
        save_to = ARGS['COMPARE_READS_IMAGES_FOLDER'] if save_dir is None else save_dir
        cv2.imwrite(os.path.join(save_to, patient_id + '.jpg'), img)


def make_images_from_file(filename: str,
                          first_reads: pd.DataFrame,
                          im_path: Optional[str] = None,
                          save_dir: Optional[str] = None) -> None:
    """
    Creates new image files with drawn-on annotations from both reads. Uses a pre-made CSV file
    with bounding boxes from both reads already classified as true positive, false positive, and
    false negative.

    Parameters
    ----------
    filename    : path to the CSV file containing both read bounding box information
    first_reads : DataFrame with image and bounding box locations from the first radiologist reads
    im_path     : path containing images to draw annotations on
    save_dir    : path to save images with drawn-on annotations
    """
    # Define list of images
    img_list = [os.path.join(root, file) for root, _, files in os.walk(im_path) for file in files]

    # Load in DataFrame from CSV file
    bbox_df = pd.read_csv(filename, names=(['Patient', 'Read1 Box', 'Read1 Area', 'Read2 Box', 'Read2 Area', 'Result', 'Max IOU']))

    # Remove fracture absent images from first read
    first_reads = first_reads.dropna()

    # Loop through original/cropped images, draw annotations, and save JPEGs
    for _, img_nm in tqdm(iterable=enumerate(img_list), desc='Saving Annotated Images', total=len(img_list)):
        # Pull the Patient ID from the annotations file
        patient_id = img_nm.split('/')[-1].replace('.png', '')

        # Pull current patient bounding boxes from the first read
        patient_read1 = first_reads[first_reads['ID'].str.contains(patient_id)]

        if len(patient_read1) < 1:
            continue

        # Pull current patient bounding boxes from comparison CSV file
        patient_bboxes = bbox_df[bbox_df['Patient'].str.contains(patient_id)]

        # Define box colors in BGR format
        box_color_gt = [255, 255, 0] # teal
        box_color_tp = [0, 255, 0]   # green
        box_color_fp = [0, 255, 255] # yellow
        box_color_fn = [0, 0, 255]   # red

        # Import image
        img = cv2.imread(img_nm)

        # Loop through first read annotations and draw boxes as ground truth
        for _, row in patient_read1.iterrows():
            box = [row[1], row[2], row[3], row[4]]
            draw_box(img, box, box_color_gt)

        # Loop through comparison DataFrame and draw boxes with certain colors depending on result
        for _, row in patient_bboxes.iterrows():
            if row[5] == 'true_positive':
                row_box = row[3].replace('(', '').replace(')', '').replace(',', '')
                box = [int(val) for val in row_box.split()]
                draw_box(img, box, box_color_tp)
            elif row[5] == 'false_positive':
                row_box = row[3].replace('(', '').replace(')', '').replace(',', '')
                box = [int(val) for val in row_box.split()]
                draw_box(img, box, box_color_fp)
            else:
                row_box = row[1].replace('(', '').replace(')', '').replace(',', '')
                box = [int(val) for val in row_box.split()]
                draw_box(img, box, box_color_fn)

        # Save image to file
        save_to = ARGS['COMPARE_READS_IMAGES_COLORED_FOLDER'] if save_dir is None else save_dir
        cv2.imwrite(os.path.join(save_to, patient_id + '.jpg'), img)


def calculate_metrics(first_reads: pd.DataFrame,
                      second_reads: pd.DataFrame,
                      iou_threshold: Union[float, Iterable[float]] = None,
                      verbose: bool = False,
                      model: bool = False,
                      model_conf: Optional[float] = None,
                      bootstrap_iters: int = 0,
                      save_name: Optional[str] = None) -> None:
    """
    Calculates various performance metrics across the two reads.

    Parameters
    ----------
    first_reads   : DataFrame with image and bounding box locations from the first radiologist reads
    second_reads  : DataFrame with image and bounding box locations from the second radiologist reads
    iou_threshold : if float, threshold to consider bounding box overlap as a true positive.
                    if array, will loop through all thresholds and save performance to CSV
    verbose       : whether to print out metrics to console
    model         : whether model predictions are being used as one of the reads (use read2)
    model_conf    : threshold to keep bounding box predictions from the model
    save_name     : name to save the file as
    """
    # Pull out unique PatientID.png from ID column of both reads
    read1_names = np.unique([name.split('/')[-1].upper() for name in first_reads.ID])
    read2_names = np.unique([name.split('/')[-1].upper() for name in second_reads.ID])

    # Find matching PatientIDs
    match_annos = np.intersect1d(read1_names, read2_names)
    print(f'{len(match_annos)} MATCHING IDs -- TEST SET SIZE {len(read1_names)}')

    if isinstance(iou_threshold, (list, tuple, set, np.ndarray, pd.Series)):
        # Instantiate lists
        accuracies = []
        recalls = []

        for _, thresh in enumerate(iou_threshold):
            # Create an empty DataFrame to add calculations per image
            calc_df = pd.DataFrame(columns=(['Patient', 'BBoxes Read 1', 'BBoxes Read 2', 'True Positives', 'False Positives', 'False Negatives', 'True Negatives']))

            for _, patient in tqdm(enumerate(match_annos), desc=f'Calculating Metrics at IOU {thresh}', total=len(match_annos)):
                # Get first- and second-read bounding boxes for patient
                read1_bboxes = get_bounding_boxes(patient, anno_df=first_reads)
                read2_bboxes = get_bounding_boxes(patient, anno_df=second_reads)

                # Calculate performance between bounding boxes
                true_pos, false_pos, false_neg, true_neg, _, _ = calc_performance(read2_bboxes, read1_bboxes, iou_threshold=thresh)

                # Add values to calc_df
                calc_df = calc_df.append({'Patient' : patient,
                                          'BBoxes Read 1' : len(read1_bboxes),
                                          'BBoxes Read 2' : len(read2_bboxes),
                                          'True Positives' : true_pos,
                                          'False Positives' : false_pos,
                                          'False Negatives' : false_neg,
                                          'True Negatives' : true_neg}, ignore_index=True)

            # Create a Metrics object with the confusion matrix totals
            metric_calc = MetricsConfMatrix(calc_df['True Positives'].sum(),
                                            calc_df['False Positives'].sum(),
                                            calc_df['False Negatives'].sum(),
                                            calc_df['True Negatives'].sum()) #for some reason used 0 before?

            # Add values to lists
            accuracies.append(metric_calc.accuracy())
            recalls.append(metric_calc.recall())

        # Write accuracies, recalls, and thresholds to CSV
        print('Writing to file...')
        with open(save_name, 'w') as out_file:
            for thresh, acc, rec in zip(iou_threshold, accuracies, recalls):
                out_str = ','.join([str(thresh), str(acc), str(rec)]) + '\n'
                out_file.write(out_str)

    else:
        # Create an empty DataFrame to add calculations per image
        calc_df = pd.DataFrame(columns=(['Patient', 'BBoxes Read 1', 'BBoxes Read 2', 'True Positives', 'False Positives', 'False Negatives', 'True Negatives']))

        all_overlaps = []
        all_ious = []
        frac_abs_imgs = 0
        for _, patient in tqdm(enumerate(match_annos), desc='Calculating Metrics', total=len(match_annos)):
            # Get first- and second-read bounding boxes for patient
            read1_bboxes = get_bounding_boxes(patient, anno_df=first_reads)
            frac_abs_imgs += 1 if len(read1_bboxes) == 0 else 0
            if not model:
                read2_bboxes = get_bounding_boxes(patient, anno_df=second_reads)
            else:
                read2_bboxes, _ = get_bounding_boxes(patient, anno_df=second_reads, has_probs=True, conf_threshold=model_conf)

            # Calculate performance between bounding boxes
            true_pos, false_pos, false_neg, true_neg, ious, overlaps = calc_performance(read2_bboxes, read1_bboxes, iou_threshold=iou_threshold)

            # Add percent overlaps to all_overlaps
            for val in overlaps:
                all_overlaps.append(val)
            for val in ious:
                all_ious.append(val)

            # Add values to calc_df
            calc_df = calc_df.append({'Patient' : patient,
                                      'BBoxes Read 1' : len(read1_bboxes),
                                      'BBoxes Read 2' : len(read2_bboxes),
                                      'True Positives' : true_pos,
                                      'False Positives' : false_pos,
                                      'False Negatives' : false_neg,
                                      'True Negatives' : true_neg}, ignore_index=True)

        # Convert lists to arrays
        all_overlaps = np.array(all_overlaps)
        all_ious = np.array(all_ious)

        # Create a Metrics object with the confusion matrix totals
        metric_calc = MetricsConfMatrix(calc_df['True Positives'].sum(),
                                        calc_df['False Positives'].sum(),
                                        calc_df['False Negatives'].sum(),
                                        calc_df['True Negatives'].sum())

        mAP = calc_mAP(preds=second_reads, annots=first_reads, iou_threshold=iou_threshold)
        curr_auc = compute_afroc(first_reads, second_reads, iou_threshold, no_save=True)

    # Print out performance stats and metrics
    if verbose:
        frac_pres_df = calc_df[calc_df['True Negatives'] == 0]
        print('')
        print('|{:^24}|{:^10}|{:^10}|'.format('METRIC', 'Read 1', 'Read 2'))
        print('|{}|'.format('-'*46))
        print('|{:^24}|{:^21}|'.format('Total Images', len(match_annos)))
        print('|{:^24}|{:^10}|{:^10}|'.format('Fracture Present Images', len(match_annos) - frac_abs_imgs, len(calc_df[calc_df['True Negatives'] == 0])))
        print('|{:^24}|{:^10}|{:^10}|'.format('Fracture Absent Images', frac_abs_imgs, len(calc_df[calc_df['True Negatives'] == 1])))
        print('|{:^24}|{:^10}|{:^10}|'.format('Total Ribs Labeled', frac_pres_df['BBoxes Read 1'].sum(), frac_pres_df['BBoxes Read 2'].sum()))
        print('|{:^24}|{:^10.5}|{:^10.5}|'.format('Avg. Ribs/Image', frac_pres_df['BBoxes Read 1'].mean(), frac_pres_df['BBoxes Read 2'].mean()))
        print('|{:^24}|{:^10.5}|{:^10.5}|'.format('StdDev. Ribs/Image', frac_pres_df['BBoxes Read 1'].std(), frac_pres_df['BBoxes Read 2'].std()))
        print('|{:^24}|{:^10}|{:^10}|'.format('Min Ribs/Image', frac_pres_df['BBoxes Read 1'].min(), frac_pres_df['BBoxes Read 2'].min()))
        print('|{:^24}|{:^10}|{:^10}|'.format('Max Ribs/Image', frac_pres_df['BBoxes Read 1'].max(), frac_pres_df['BBoxes Read 2'].max()))
        print('|{:^24}|{:^10}|{:^10}|'.format('Median Ribs/Image', frac_pres_df['BBoxes Read 1'].median(), frac_pres_df['BBoxes Read 2'].median()))
        print('|{:^24}|{:^10.2}|{:^10.2}|'.format('Q1 Ribs/Image', frac_pres_df['BBoxes Read 1'].quantile(0.25), frac_pres_df['BBoxes Read 2'].quantile(0.25)))
        print('|{:^24}|{:^10.2}|{:^10.2}|'.format('Q3 Ribs/Image', frac_pres_df['BBoxes Read 1'].quantile(0.75), frac_pres_df['BBoxes Read 2'].quantile(0.75)))
        print('|{:^24}|{:^10.2}|{:^10.2}|'.format('IQR Ribs/Image', frac_pres_df['BBoxes Read 1'].quantile(0.75) - frac_pres_df['BBoxes Read 1'].quantile(0.25), frac_pres_df['BBoxes Read 2'].quantile(0.75) - calc_df['BBoxes Read 2'].quantile(0.25)))
        print('|{}|'.format('-'*46))
        print('|{:^24}|{:^21.3}|'.format('IOU Threshold', iou_threshold))
        print('|{:^24}|{:^21.5}|'.format('Avg. Percent Overlap', all_overlaps.mean()))
        print('|{:^24}|{:^21.5}|'.format('Avg. IOU', all_ious.mean()))
        print('|{:^24}|{:^21.3}|'.format('Model Confidence', model_conf if model else ''))
        print('|{}|'.format('-'*46))
        print('|{:^24}|{:^21}|'.format('True Positives', metric_calc.true_pos))
        print('|{:^24}|{:^21}|'.format('False Positives', metric_calc.false_pos))
        print('|{:^24}|{:^21}|'.format('False Negatives', metric_calc.false_neg))
        print('|{:^24}|{:^21}|'.format('True Negatives', metric_calc.true_neg))
        print('|{:^24}|{:^21.5}|'.format(f'mAP@{parser_args.iou_thresh:.1f}', mAP))
        print('|{:^24}|{:^21.5}|'.format('Accuracy', metric_calc.accuracy()))
        print('|{:^24}|{:^21.5}|'.format('Precision', metric_calc.precision()))
        print('|{:^24}|{:^21.5}|'.format('Recall/TPR/Sens', metric_calc.recall()))
        print('|{:^24}|{:^21.5}|'.format('F1 Score', metric_calc.f1_score()))
        print('|{:^24}|{:^21.5}|'.format('F2 Score', metric_calc.f2_score()))
        print('|{:^24}|{:^21.5}|'.format('AUC', curr_auc))
        print('|{:^24}|{:^21.5}|'.format('Cohen\'s Kappa', metric_calc.cohens_kappa()))
        print('|{:^24}|{:^21.5}|'.format('Free-Response Kappa', metric_calc.free_kappa()))
        print('')

        # Bootstrapping to obtain 95% CIs and measures of variation in precision, recall, and F2 score
        if bootstrap_iters > 0:
            boot_precisions, boot_recalls, boot_f2s = [], [], []
            for i in tqdm(range(bootstrap_iters), desc='Bootstrapping...'):
                # Take stratified bootstrap sample: n_pos fracture-present cases w/ replacement + n_neg fracture-absent cases w/ replacement
                bootstrap_df_pos = calc_df[calc_df['BBoxes Read 1'] > 0].sample(frac=1, replace=True, random_state=i)
                bootstrap_df_neg = calc_df[calc_df['BBoxes Read 1'] == 0].sample(frac=1, replace=True, random_state=i)
                bootstrap_df = pd.concat([bootstrap_df_pos, bootstrap_df_neg])

                true_pos, false_pos, false_neg, true_neg = bootstrap_df['True Positives'].sum(), bootstrap_df['False Positives'].sum(), bootstrap_df['False Negatives'].sum(), bootstrap_df['True Negatives'].sum()

                boot_metric_calc = MetricsConfMatrix(true_pos, false_pos, false_neg, true_neg)

                precision = boot_metric_calc.precision()
                recall = boot_metric_calc.recall()
                f2_score = boot_metric_calc.f2_score()

                boot_precisions.append(precision)
                boot_recalls.append(recall)
                boot_f2s.append(f2_score)

            sorted_boot_precisions = sorted(boot_precisions)
            sorted_boot_recalls = sorted(boot_recalls)
            sorted_boot_f2s = sorted(boot_f2s)

            boot_precision_lb, boot_precision_ub = sorted_boot_precisions[int(0.05*len(sorted_boot_precisions))], sorted_boot_precisions[int(0.95*len(sorted_boot_precisions))]
            boot_recall_lb, boot_recall_ub = sorted_boot_recalls[int(0.05*len(sorted_boot_recalls))], sorted_boot_recalls[int(0.95*len(sorted_boot_recalls))]
            boot_f2_lb, boot_f2_ub = sorted_boot_f2s[int(0.05*len(sorted_boot_f2s))], sorted_boot_f2s[int(0.95*len(sorted_boot_f2s))]

            print(f'95% CI for Precision: ({boot_precision_lb:.5f}, {boot_precision_ub:.5f}) | CI Length: {boot_precision_ub-boot_precision_lb:.5f}')
            print(f'\tMean +/- std for Precision: {np.mean(boot_precisions):.5f} +/- {np.std(boot_precisions):.5f} | Coeff. of Variation: {np.std(boot_precisions)/np.mean(boot_precisions):.5f}')
            print(f'95% CI for Recall: ({boot_recall_lb:.5f}, {boot_recall_ub:.5f}) | CI Length: {boot_recall_ub-boot_recall_lb:.5f}')
            print(f'\tMean +/- std for Recall: {np.mean(boot_recalls):.5f} +/- {np.std(boot_recalls):.5f} | Coeff. of Variation: {np.std(boot_recalls)/np.mean(boot_recalls):.5f}')
            print(f'95% CI for F2 Score: ({boot_f2_lb:.5f}, {boot_f2_ub:.5f}) | CI Length: {boot_f2_ub-boot_f2_lb:.5f}')
            print(f'\tMean +/- std for F2 Score: {np.mean(boot_f2s):.5f} +/- {np.std(boot_f2s):.5f} | Coeff. of Variation: {np.std(boot_f2s)/np.mean(boot_f2s):.5f}')


def test_calc_perf(first_reads: pd.DataFrame,
                   second_reads: pd.DataFrame,
                   iou_threshold: Optional[float] = None,
                   model: bool = False,
                   model_conf: Optional[float] = None) -> None:
    """
    Print out a CSV with ground truth vs model prediction bbox performance from the old and new
    versions of calc_performance.

    Parameters
    ----------
    first_reads   : image and bounding box locations from the first radiologist reads
    second_reads  : image and bounding box locations from the second radiologist reads
    iou_threshold : threshold at which to consider bounding box overlap as a true positive
    model         : whether model predictions are being used as one of the reads (use read2)
    model_conf    : threshold to keep bounding box predictions from the model
    """
    # Pull out unique PatientID.png from ID column of both reads
    read1_names = np.unique([name.split('/')[-1].upper() for name in first_reads.ID])
    read2_names = np.unique([name.split('/')[-1].upper() for name in second_reads.ID])

    # Find matching PatientIDs
    match_annos = np.intersect1d(read1_names, read2_names)
    print(f'{len(match_annos)} MATCHING IDs -- TEST SET SIZE {len(read1_names)}')

    orig_tps = []
    orig_fps = []
    orig_fns = []
    orig_tns = []
    orig_overlaps = []
    orig_ious = []

    new_tps = []
    new_fps = []
    new_fns = []
    new_tns = []
    new_overlaps = []
    new_ious = []

    for _, patient in tqdm(enumerate(match_annos), desc='Calculating Metrics', total=len(match_annos)):
        # Get first- and second-read bounding boxes for patient
        read1_bboxes = get_bounding_boxes(patient, anno_df=first_reads)
        if not model:
            read2_bboxes = get_bounding_boxes(patient, anno_df=second_reads)
        else:
            read2_bboxes, _ = get_bounding_boxes(patient, anno_df=second_reads, has_probs=True, conf_threshold=model_conf)

        # Calculate performance between bounding boxes
        true_pos, false_pos, false_neg, true_neg, ious, overlaps = calc_performance_OLD(read2_bboxes, read1_bboxes, iou_threshold=iou_threshold)

        # Add percent overlaps to all_overlaps
        orig_tps.append(true_pos)
        orig_fps.append(false_pos)
        orig_fns.append(false_neg)
        orig_tns.append(true_neg)
        orig_ious.append(ious)
        orig_overlaps.append(overlaps)

        true_pos, false_pos, false_neg, true_neg, ious, overlaps = calc_performance(read2_bboxes, read1_bboxes, iou_threshold=iou_threshold)

        # Add percent overlaps to all_overlaps
        new_tps.append(true_pos)
        new_fps.append(false_pos)
        new_fns.append(false_neg)
        new_tns.append(true_neg)
        new_ious.append(ious)
        new_overlaps.append(overlaps)

    with open('test_calc_performances.csv', 'w') as out_file:
        for patient, orig_tp, orig_fp, orig_fn, orig_tn, new_tp, new_fp, new_fn, new_tn, in zip(
                match_annos, orig_tps, orig_fps, orig_fns, orig_tns, new_tps, new_fps, new_fns, new_tns
        ):
            out_str = ','.join([patient, 'ORIG', str(orig_tp), str(orig_fp), str(orig_fn), str(orig_tn), 'NEW', str(new_tp), str(new_fp), str(new_fn), str(new_tn)]) + '\n'
            out_file.write(out_str)


def avalanche_scheme(first_reads: pd.DataFrame, second_reads: pd.DataFrame, iou_threshold: float = None) -> None:
    """
    Loop through all possible model confidence value and calculate the performance of the model with
    and without an avalanche decision scheme.

    Parameters
    ----------
    first_reads   : DataFrame with image and bounding box locations from the first radiologist reads
    second_reads  : DataFrame with image and bounding box locations from the second radiologist reads
    iou_threshold : threshold at which to consider bounding box overlap as a true positive
    """
    # Pull out unique PatientID.png from ID column of both reads
    read1_names = np.unique([name.split('/')[-1].upper() for name in first_reads.ID])
    read2_names = np.unique([name.split('/')[-1].upper() for name in second_reads.ID])

    # Find matching PatientIDs
    match_annos = np.intersect1d(read1_names, read2_names)
    print(f'{len(match_annos)} MATCHING IDs -- TEST SET SIZE {len(read1_names)}')

    methods = ["", "conservative", "posterior"]
    rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    for method in methods:
        avalanche_calc(first_reads, second_reads, match_annos, iou_threshold, method=method)

    for rate in rates:
        avalanche_calc(first_reads, second_reads, match_annos, iou_threshold, method="avalanche", rate=rate)


def avalanche_calc(first_reads, second_reads, match_annos, iou_threshold=None, method="", rate=0.05):
    """
    Create a detailed metric CSV with Precision, Recall, F1, and F2 scores across the entire range
    of possible model confidences, with choice between standard or avalanche confidence schemes.

    Parameters
    ----------
    first_reads : DataFrame
        contains image and bounding box locations from the first radiologist reads
    second_reads : DataFrame
        contains image and bounding box locations from the second radiologist reads
    match_annos : ndarray
        list of matching patient IDs from both reads
    iou_threshold : float
        threshold at which to consider bounding box overlap as a true positive
    method : str
        which avalanche scheme to use
    rate : float, optional
        constant rate to decrease each successive step of avalanche scheme
    """
    # Each value is the percentage of the prior value for the threshold calculation
    # (e.g., if a = 1, the first thresh is 1*0.736, and second is 1*0.736*0.764)
    if method == "conservative":
        avalanche_percentages = [1, 0.7643979057591623, 0.8424657534246576, 0.7723577235772358, 0.7842105263157895]
    elif method == "posterior":
        avalanche_percentages = [1, 1-0.7643979057591623, 1-0.8424657534246576, 1-0.7723577235772358, 1-0.7842105263157895]
    elif method == "avalanche":
        avalanche_percentages = [1-rate for _ in range(10)]
        avalanche_percentages.insert(0, 1)
    else:
        avalanche_percentages = []

    # Create an empty DataFrame to add cumulative metrics across the dataset per base value
    metrics_df = pd.DataFrame(columns=(['base_val', 'precision', 'recall', 'f1_score', 'f2_score']))

    # for base_val in np.arange(0, 21) / 20.0:
    for base_val in np.arange(0, 1.01, 0.05):
        # Create an empty DataFrame to add calculations per image
        # Placed here so that it is reset for each base value
        calc_df = pd.DataFrame(columns=(['Patient', 'BBoxes Read 1', 'BBoxes Read 2', 'True Positives', 'False Positives', 'False Negatives', 'True Negatives'])) 
        for _, patient in tqdm(enumerate(match_annos), desc=f'Calculating Metrics at {base_val=:.2}', total=len(match_annos)):
            # Pull boxes from ground truth/first reads
            read1_bboxes = get_bounding_boxes(patient, anno_df=first_reads)
            # print(f'{read1_bboxes=}')

            if method in ["avalanche", "conservative", "posterior"]:
                test_thresholds = [base_val * val for val in [np.prod(np.array(avalanche_percentages[:k])) for k in range(1, len(avalanche_percentages)+1)]]
                # Get initial number of boxes from list of thresholds
                read2_bboxes, _ = get_bounding_boxes(patient, anno_df=second_reads, has_probs=True, conf_threshold=test_thresholds[0])
                num_model_boxes = len(read2_bboxes)

                prior_num_model_boxes = 0
                while num_model_boxes > prior_num_model_boxes:
                    prior_num_model_boxes = num_model_boxes
                    # Set threshold level based on number of already accepted fractures
                    test_thresh_ind = min(len(test_thresholds)-1, num_model_boxes)
                    thresh = test_thresholds[test_thresh_ind]
                    read2_bboxes, _ = get_bounding_boxes(patient, anno_df=second_reads, has_probs=True, conf_threshold=thresh)
                    num_model_boxes = len(read2_bboxes)

                    if test_thresh_ind == len(test_thresholds):
                        break
            else:
                # Get the standard, unchanging metrics for each base value
                read2_bboxes, _ = get_bounding_boxes(patient, anno_df=second_reads, has_probs=True, conf_threshold=base_val)

            # print(f'{read2_bboxes=}')
            # Calculate performance between bounding boxes
            true_pos, false_pos, false_neg, true_neg, _, _ = calc_performance(read2_bboxes, read1_bboxes, iou_threshold=iou_threshold)

            # Add values to calc_df
            calc_df = calc_df.append({'Patient' : patient,
                                      'BBoxes Read 1' : len(read1_bboxes),
                                      'BBoxes Read 2' : len(read2_bboxes),
                                      'True Positives' : true_pos,
                                      'False Positives' : false_pos,
                                      'False Negatives' : false_neg,
                                      'True Negatives' : true_neg}, ignore_index=True)

        # Create a Metrics object with the confusion matrix totals and calculate metrics for avalanche confidence
        metric_calc = MetricsConfMatrix(calc_df['True Positives'].sum(),
                                        calc_df['False Positives'].sum(),
                                        calc_df['False Negatives'].sum(),
                                        calc_df['True Negatives'].sum())

        metrics_df = metrics_df.append({'base_val'  : round(base_val, 2),
                                        'precision' : metric_calc.precision(),
                                        'recall'    : metric_calc.recall(),
                                        'f1_score'  : metric_calc.f1_score(),
                                        'f2_score'  : metric_calc.f2_score()}, ignore_index=True)
    # print(metrics_df)
    filename = f'metrics_{"standard" if method == "" else method}{"_" + str(rate) if method == "avalanche" else ""}'
    metrics_df.to_csv(f'out_files/{filename}.csv', index=False)


def create_bbox_dataframe(first_reads, second_reads, iou_threshold, save_name='', model=None, model_conf=None):
    """
    Create a DataFrame with each bounding box for each patient for both reads.

    Parameters
    ----------
    first_reads : DataFrame
        contains image and bounding box locations from the first radiologist reads
    second_reads : DataFrame
        contains image and bounding box locations from the second radiologist reads
    save_name : str
        name to save the file as
    """
    # Remove any rows containing NaN
    first_reads = first_reads.dropna()
    second_reads = second_reads.dropna()

    # Pull out unique PatientID.png from ID column of both reads
    read1_names = np.unique([name[name.rfind('/')+1:] for name in first_reads.ID])
    read2_names = np.unique([name[name.rfind('/')+1:] for name in second_reads.ID])

    # Find matching PatientIDs
    match_annos = np.intersect1d(read1_names, read2_names)

    # Create an empty DataFrame to add calculations per image
    bbox_df = pd.DataFrame(columns=(['Patient', 'Read1 Box', 'Read1 Area', 'Read2 Box', 'Read2 Area', 'Result', 'Max IOU']))

    for _, patient in tqdm(enumerate(match_annos), desc='Calculating Metric DataFrame', total=len(match_annos)):
        # Get first- and second-read bounding boxes for patient
        read1_bboxes = get_bounding_boxes(patient, anno_df=first_reads)
        if model:
            read2_bboxes, _ = get_bounding_boxes(patient, anno_df=second_reads, has_probs=True, conf_threshold=model_conf)
        else:
            read2_bboxes = get_bounding_boxes(patient, anno_df=second_reads)

        # Loop through each read 1 box and find overlapping boxes from read 2
        for box1 in read1_bboxes:
            if len(read2_bboxes) > 0:
                # Loop through each read 2 box and calculate the IOU with the current read 1 box
                temp_ious = np.array([])
                for box2 in read2_bboxes:
                    tmp_iou, _ = intersection_over_union(box2, box1)
                    temp_ious = np.append(temp_ious, tmp_iou)

                # Pull out the largest IOU and corresponding index
                max_ind = np.argmax(temp_ious)
                max_iou = temp_ious.max()

                # If the IOU is above the threshold, add info to the DataFrame as a true positive
                if max_iou > iou_threshold:
                    box2 = read2_bboxes[max_ind]
                    bbox_df = bbox_df.append({'Patient' : patient,
                                              'Read1 Box' : box1,
                                              'Read1 Area' : calc_bbox_area(box1),
                                              'Read2 Box' : box2,
                                              'Read2 Area' : calc_bbox_area(box2),
                                              'Result' : 'true_positive',
                                              'Max IOU' : max_iou}, ignore_index=True)
                    read2_bboxes.remove(box2)
                # If the max IOU is below the threshold, add info as a false negative
                else:
                    bbox_df = bbox_df.append({'Patient' : patient,
                                              'Read1 Box' : box1,
                                              'Read1 Area' : calc_bbox_area(box1),
                                              'Read2 Box' : None,
                                              'Read2 Area' : None,
                                              'Result' : 'false_negative',
                                              'Max IOU' : max_iou}, ignore_index=True)
            # If there are no more second read boxes, add rest of read 1 boxes as false negatives
            else:
                bbox_df = bbox_df.append({'Patient' : patient,
                                          'Read1 Box' : box1,
                                          'Read1 Area' : calc_bbox_area(box1),
                                          'Read2 Box' : None,
                                          'Read2 Area' : None,
                                          'Result' : 'false_negative',
                                          'Max IOU' : 0}, ignore_index=True)
        # If there are still read 2 boxes after looping through all read 1 boxes, add to DataFrame
        # as false positives
        if len(read2_bboxes) > 0:
            for box2 in read2_bboxes:
                bbox_df = bbox_df.append({'Patient' : patient,
                                          'Read1 Box' : None,
                                          'Read1 Area' : None,
                                          'Read2 Box' : box2,
                                          'Read2 Area' : calc_bbox_area(box2),
                                          'Result' : 'false_positive',
                                          'Max IOU' : 0}, ignore_index=True)

    # Output bbox_df to a file
    print('Writing to file...')
    bbox_df.to_csv(save_name, index=False)


def compute_afroc_OLD(ground_truths: pd.DataFrame,
                    model_predictions: pd.DataFrame,
                    save_name: str = '',
                    no_save: bool = False) -> None:
    """
    Calculates LLF and FPR for each threshold to plot an AFROC curve of the model performance

    Parameters
    ----------
    ground_truths     : contains image and bounding box locations from radiologist reads
    model_predictions : contains image and bounding box locations with probabilities from the neural network
    save_name         : name to save the file as
    no_save           : determines whether to save the afroc values to a file or just output AUC
    """
    # Sort all the probabilities from the model
    sorted_scores = model_predictions.Prob.sort_values().values

    # Pull out unique PatientID.png from ID column of both reads
    gt_names = np.unique([name[name.rfind('/')+1:] for name in ground_truths.ID])
    pred_names = np.unique([name[name.rfind('/')+1:] for name in model_predictions.ID])

    # Find matching PatientIDs
    match_annos = np.intersect1d(gt_names, pred_names)

    # Instantiate lists and variables
    llf_list = []
    fpr_list = []
    total_fractures = len(ground_truths)
    total_images = len(match_annos)

    # Get ground truth and predicted boxes for every patient (BEFORE the loop so we only do this once for each patient)
    truth_boxes = [get_bounding_boxes(patient, anno_df=ground_truths) for patient in match_annos]
    pred_boxes, pred_scores = [], []
    for patient in match_annos:
        b, s = get_bounding_boxes(patient, anno_df=model_predictions, has_probs=True)
        pred_boxes.append(b)
        pred_scores.append(s)

    # Loop through sorted_scores and calculate LLF and FPR for each score
    for _, threshold in tqdm(enumerate(sorted_scores), total=sorted_scores.size):
        total_true_pos = 0
        total_false_pos = 0

        # Loop through each patient and compare ground truth and predicted bounding boxes
        for j, patient in enumerate(match_annos):
            # Filter out predicted boxes below score threshold
            keep = np.where(np.array(pred_scores[j]) >= threshold)[0]
            good_preds = [np.array(pred_boxes[j][k], dtype=int) for k in keep]

            # Calculate number of true and false positives for current patient
            true_pos, false_pos, _, _, _, _ = calc_performance(good_preds, truth_boxes[j], iou_threshold=parser_args.iou_thresh)

            # Add true and false positives to total for the current threshold
            total_true_pos += true_pos
            total_false_pos += false_pos

        # Calculate LLF, NLF, and FPR for current threshold
        curr_llf = min(1., total_true_pos / total_fractures)
        curr_nlf = total_false_pos / total_images
        est_fpr = 1 - np.exp(-curr_nlf)

        # Add LLF and FPR values to lists
        llf_list.append(curr_llf)
        fpr_list.append(est_fpr)

    if no_save:
        return auc(fpr_list, llf_list)

    # Output values to a file
    print('Writing to file...')
    with open(save_name, 'w') as out_file:
        for threshold, llf, fpr in zip(sorted_scores, llf_list, fpr_list):
            out_file.write('{},{},{}\n'.format(threshold, llf, fpr))

    # Print area under AFROC curve
    print(f'Area under AFROC: {auc(fpr_list, llf_list):.5f}')


def compute_afroc(ground_truths: pd.DataFrame,
                  model_predictions: pd.DataFrame,
                  iou_threshold: float,
                  save_name: str = '',
                  no_save: bool = False) -> None:
    """
    Calculates LLF and FPR for each threshold to plot an AFROC curve of the model performance

    Parameters
    ----------
    ground_truths     : contains image and bounding box locations from radiologist reads
    model_predictions : contains image and bounding box locations with probabilities from the neural network
    save_name         : name to save the file as
    no_save           : determines whether to save the afroc values to a file or just output AUC
    """
    model_predictions = model_predictions.sort_values(by=['ID', 'Prob'], ascending=[True, False], ignore_index=True) # Sort by ID to be consistent with match_annos later

    sorted_scores = model_predictions.Prob.sort_values().values
    sorted_scores = sorted_scores[~np.isnan(sorted_scores)] # Get rid of nan values

    # Pull out unique PatientID.png from ID column of both reads
    gt_names = np.unique([name[name.rfind('/')+1:] for name in ground_truths.ID])
    pred_names = np.unique([name[name.rfind('/')+1:] for name in model_predictions.ID])

    # Find matching PatientIDs
    match_annos = np.intersect1d(gt_names, pred_names)

    iou_array = np.array([])
    for patient in match_annos:
        truth_boxes = get_bounding_boxes(patient, anno_df=ground_truths)
        pred_boxes, _ = get_bounding_boxes(patient, anno_df=model_predictions, has_probs=True)

        num_preds = len(pred_boxes) if pred_boxes else 1
        num_truths = len(truth_boxes) if truth_boxes else 1
        temp_iou_array = np.zeros((num_preds, num_truths))
        for i, pred in enumerate(pred_boxes):
            for k, truth in enumerate(truth_boxes):
                if pred and truth: # Continue only if both pred and truth have boxes
                    temp_iou, _ = intersection_over_union(pred, truth)
                    if temp_iou >= iou_threshold:
                        temp_iou_array[i, k] = temp_iou

        iou_array = block_diag(iou_array, temp_iou_array)
    iou_array = np.delete(iou_array, 0, axis=0) # delete first row -- always empty

    assert len(iou_array) == len(model_predictions)

    # Instantiate lists and variables
    llf_list = []
    fpr_list = []
    total_fractures = len(ground_truths)
    total_images = len(set(ground_truths.ID))

    # Loop through sorted_scores and calculate LLF and FPR for each score
    for _, threshold in tqdm(enumerate(sorted_scores), desc='Calculating AUC', total=sorted_scores.size):

        thresholded_iou_array = iou_array[model_predictions.Prob >= threshold]
        
        true_pos = np.where(thresholded_iou_array.any(axis=0))[0].size
        false_pos = np.where(~thresholded_iou_array.any(axis=1))[0].size

        # Calculate LLF, NLF, and FPR for current threshold
        curr_llf = min(1., true_pos / total_fractures)
        curr_nlf = false_pos / total_images
        est_fpr = 1 - np.exp(-curr_nlf)

        # Add LLF and FPR values to lists
        llf_list.append(curr_llf)
        fpr_list.append(est_fpr)

    if no_save:
        return auc(fpr_list, llf_list)

    # Print area under AFROC curve
    print(f'Area under AFROC: {auc(fpr_list, llf_list):.5f}')

    # Output values to a file
    print('Writing to file...')
    with open(save_name, 'w') as out_file:
        for threshold, llf, fpr in zip(sorted_scores, llf_list, fpr_list):
            out_file.write('{},{},{}\n'.format(threshold, llf, fpr))


def main(parse_args):
    """Main Function"""
    # Import first and second reads
    if parse_args.old:
        first_reads = pd.read_csv(parse_args.first_read_csv, names=('ID', 'x1', 'y1', 'x2', 'y2', 'class'))
    else:
        first_reads = pd.read_csv(parse_args.first_read_csv, names=('ID', 'Height', 'Width', 'x1', 'y1', 'x2', 'y2'))

    if parse_args.model:
        # if parse_args.old:
        #     second_reads = pd.read_csv(parse_args.second_read_csv, names=('ID', 'x1', 'y1', 'x2', 'y2', 'Prob'))
        # else:
        #     second_reads = pd.read_csv(parse_args.second_read_csv, names=('ID', 'Height', 'Width', 'x1', 'y1', 'x2', 'y2', 'Prob'))
        second_reads = pd.read_csv(parse_args.second_read_csv, header=0)
        if len(second_reads.columns) > 6:
            second_reads = pd.read_csv(parse_args.second_read_csv, names=('ID', 'Height', 'Width', 'x1', 'y1', 'x2', 'y2', 'Prob'))
        else:
            second_reads = pd.read_csv(parse_args.second_read_csv, names=('ID', 'x1', 'y1', 'x2', 'y2', 'Prob'))
    else:
        second_reads = pd.read_csv(parse_args.second_read_csv, names=('ID', 'Height', 'Width', 'x1', 'y1', 'x2', 'y2'))

    # Drop Height and Width columns
    if not parse_args.old:
        first_reads = first_reads.drop(columns=['Height', 'Width'])

    if parse_args.images:
        # if parse_args.model:
        #     second_reads = second_reads[second_reads.Prob >= parse_args.model_conf]
        make_images(first_reads, second_reads, parse_args.images_path, parse_args.save_dir)

    if parse_args.color_images:
        make_images_from_file(parse_args.bbox_filename, first_reads, parse_args.images_path, parse_args.save_dir)

    if parse_args.metrics:
        calculate_metrics(first_reads, second_reads, iou_threshold=parse_args.iou_thresh,
                          verbose=True, model=parse_args.model, model_conf=parse_args.model_conf, bootstrap_iters=parse_args.bootstrap_iters)

    if parse_args.metrics2:
        test_calc_perf(first_reads, second_reads, iou_threshold=parse_args.iou_thresh,
                          verbose=True, model=parse_args.model, model_conf=parse_args.model_conf, bootstrap_iters=parse_args.bootstrap_iters)

    if parse_args.avalanche:
        avalanche_scheme(first_reads, second_reads, iou_threshold=parse_args.iou_thresh)

    save_dir = ARGS['COMPARE_READS_FOLDER'] if parse_args.save_dir is None else parse_args.save_dir

    if parse_args.bboxes:
        filename = os.path.join(save_dir, 'read1_vs_read2_bboxes.csv' if parse_args.filename is None else parse_args.filename)
        create_bbox_dataframe(first_reads, second_reads, iou_threshold=parse_args.iou_thresh,
                              save_name=filename, model=parse_args.model, model_conf=parse_args.model_conf)

    if parse_args.plot:
        filename = os.path.join(save_dir, 'perf_across_iou.csv' if parse_args.filename is None else parse_args.filename)
        if parse_args.model:
            iou_threshold = second_reads.Prob.sort_values()
        else:
            iou_threshold = np.arange(0, 101, 1) / 100
        calculate_metrics(first_reads, second_reads, iou_threshold=iou_threshold, verbose=False,
                          model=parse_args.model, model_conf=parse_args.model_conf, save_name=filename)

    if parse_args.afroc:
        filename = os.path.join(save_dir, 'read1_afroc.csv' if parse_args.filename is None else parse_args.filename)
        compute_afroc(first_reads, second_reads, save_name=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-function script to compare two radiologist reads, or one set of reads with model predictions.')

    parser.add_argument('--images', action='store_true',
                        help='Create images with drawn-on annotations from both reads.')

    parser.add_argument('--color_images', action='store_true',
                        help='Create images with drawn-on annotations from both reads; boxes colored based on TP/FP/FN.')

    parser.add_argument('--metrics', action='store_true',
                        help='Calculate various performance metrics and print them out to console.')

    parser.add_argument('--metrics2', action='store_true',
                        help='Calculate various performance metrics and print them out to console.')

    parser.add_argument('--avalanche', action='store_true',
                        help='Test thresholds based on an avalanching scheme; outputs image.')

    parser.add_argument('--bboxes', action='store_true',
                        help='Calculate statistics on bounding boxes between two reads and save to a CSV.')

    parser.add_argument('--plot', action='store_true',
                        help='Compare first and second radiologist reads across IOU thresholds and save to a CSV')

    parser.add_argument('--afroc', action='store_true',
                        help='Calculate LLF/FPR values for an AFROC plot to show model performance.')

    parser.add_argument('--model', action='store_true',
                        help='Boolean for whether model predictions are being used')

    parser.add_argument('--old', action='store_true',
                        help='If using older annotation files without height and width.')

    parser.add_argument('--images_path', type=str, default=ARGS['8_BIT_CROP_HISTEQ_IMAGE_FOLDER'],
                        help='Path to the images for --images and --color_images.')

    parser.add_argument('--first_read_csv', type=str, default=os.path.join(ARGS['COMPARE_READS_FOLDER'], 'read1_annotations_crop.csv'),
                        help='Filename to CSV containing first read annotations. Also used a ground truth annotations in --afroc.')

    parser.add_argument('--second_read_csv', type=str, default=os.path.join(ARGS['COMPARE_READS_FOLDER'], 'read2_annotations_crop.csv'),
                        help='Filename to CSV containing second read annotations. Also used as model predictions.')

    parser.add_argument('--bbox_filename', type=str,
                        help='Filename to CSV containing bounding box information from --bboxes.')

    parser.add_argument('--iou_thresh', type=float, default=0.30,
                        help='The threshold to use for determining if IOU counts toward a True Positive.')

    parser.add_argument('--model_conf', type=float, default=0.50,
                        help='The threshold to keep model bounding box predictions.')

    parser.add_argument('--save_dir',
                        help='Default folder to save files to.')

    parser.add_argument('--filename',
                        help='Name to save file as.')

    parser.add_argument('--bootstrap_iters', type=int, default=0,
                        help='Number of bootstrap samples to take of test set for confidence intervals. 0 if no bootstrapping desired.')


    parser_args = parser.parse_args()

    if not any([parser_args.images, parser_args.color_images, parser_args.metrics, parser_args.metrics2, parser_args.avalanche, parser_args.bboxes, parser_args.plot, parser_args.afroc]):
        parser.error('Please choose one of --images, --color_images, --metrics, --avalanche, --bboxes, --plot, or --afroc.')

    if parser_args.afroc and not parser_args.model:
        parser.error('Use --model flag when using --afroc.')

    # Print out start of execution
    print('\nStarting execution...')
    start_time = time.perf_counter()

    # Run main function
    main(parser_args)

    # Print out time to complete
    print('Done!')
    end_time = time.perf_counter()
    print(f'Execution finished in {end_time - start_time:.3f} seconds.\n')
