'''
Filename: compare_reads.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 03/06/2021
Description: Goes through two separate radiologist read annotation files
    and either creates images with annotations drawn on, or calculates
    a Kappa metric across the dataset.
'''

import argparse
import os
import time
import numpy as np
import pandas as pd
import cv2

from args import ARGS
from general_utils import print_iter
from eval_utils import (draw_box, get_bounding_boxes, calc_performance, calc_metric,
                        calc_bbox_area, intersection_over_union)

def make_images(first_reads, second_reads):
    """
    Creates new image files with drawn-on annotations from both reads.

    Parameters
    ----------
    first_reads : DataFrame
        contains image and bounding box locations from the first radiologist reads
    second_reads : DataFrame
        contains image and bounding box locations from the second radiologist reads
    """
    # Define list of images
    img_list = [os.path.join(root, file) for root, _, files in os.walk(ARGS['8_BIT_OG_IMAGE_FOLDER']) for file in files]

    # Loop through original/cropped images, draw annotations, and save JPEGs
    for i, img_nm in enumerate(img_list):
        print_iter(len(img_list), i, 'image')

        # Pull the Patient ID from the annotations file
        patient_id = img_nm[img_nm.rfind('Anon_'):-4]

        # Set temp DataFrames to pull annotations from
        patient_read1 = first_reads[first_reads['ID'].str.contains(patient_id)]
        patient_read2 = second_reads[second_reads['ID'].str.contains(patient_id)]

        # Import image
        img = cv2.imread(img_nm)

        # Loop through first read annotations and draw boxes
        for _, row in patient_read1.iterrows():
            box = [row[1], row[2], row[3], row[4]]
            draw_box(img, box, [255, 255, 0]) # cv2 saves JPG as BGR -> this is teal

        # Loop through second read annotations and draw boxes
        for _, row in patient_read2.iterrows():
            box = [row[1], row[2], row[3], row[4]]
            draw_box(img, box, [0, 255, 255]) # cv2 saves JPG as BGR -> this is yellow

        # Save image to file
        cv2.imwrite(os.path.join(ARGS['COMPARE_READS_IMAGES_FOLDER'], patient_id + '.jpg'), img)
    print('') # End print stream from loop

def make_images_from_file(filename, first_reads):
    """
    Creates new image files with drawn-on annotations from both reads. Uses a pre-made CSV file
    with bounding boxes from both reads already classified as true positive, false positive, and
    false negative.

    Parameters
    ----------
    filename : str
        path to the CSV file containing both read bounding box information
    first_reads : DataFrame
        contains image and bounding box locations from the first radiologist reads
    """
    # Define list of images
    img_list = [os.path.join(root, file) for root, _, files in os.walk(ARGS['8_BIT_OG_IMAGE_FOLDER']) for file in files]

    # Load in DataFrame from CSV file
    bbox_df = pd.read_csv(filename, names=(['Patient', 'Read1 Box', 'Read1 Area', 'Read2 Box', 'Read2 Area', 'Result', 'Max IOU']))

    # Loop through original/cropped images, draw annotations, and save JPEGs
    for i, img_nm in enumerate(img_list):
        print_iter(len(img_list), i, 'image')

        # Pull the Patient ID from the annotations file
        patient_id = img_nm[img_nm.rfind('Anon_'):-4]

        # Pull current patient bounding boxes from the first read
        patient_read1 = first_reads[first_reads['ID'].str.contains(patient_id)]

        # Pull current patient bounding boxes from comparison CSV file
        patient_bboxes = bbox_df[bbox_df['Patient'].str.contains(patient_id)]

        # Set box colors
        box_color_gt = [255, 255, 0] # cv2 saves JPG as BGR -> this is teal
        box_color_tp = [0, 255, 0] # cv2 saves JPG as BGR -> this is green
        box_color_fp = [0, 255, 255] # cv2 saves JPG as BGR -> this is yellow
        box_color_fn = [0, 0, 255] # cv2 saves JPG as BGR -> this is red

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
        cv2.imwrite(os.path.join(ARGS['COMPARE_READS_IMAGES_COLORED_FOLDER'], patient_id + '.jpg'), img)
    print('') # End print stream from loop

def calculate_metrics(first_reads, second_reads, iou_threshold=None, verbose=False, model=False, model_conf=None):
    """
    Calculates various performance metrics across the two reads.

    Parameters
    ----------
    first_reads : DataFrame
        contains image and bounding box locations from the first radiologist reads
    second_reads : DataFrame
        contains image and bounding box locations from the second radiologist reads
    iou_threshold : float or array
        threshold at which to consider bounding box overlap as a true positive. If array,
        it will loop through all thresholds and save performance to CSV
    verbose : bool
        whether to print out metrics to console
    model : bool
        whether model predictions are being used as one of the reads (use read2)
    model_conf : float
        threshold to keep bounding box predictions from the model
    """
    # Pull out unique PatientID.png from ID column of both reads
    read1_names = np.unique([name[name.rfind('/')+1:] for name in first_reads.ID])
    read2_names = np.unique([name[name.rfind('/')+1:] for name in second_reads.ID])

    # Find matching PatientIDs
    match_annos = np.intersect1d(read1_names, read2_names)

    if isinstance(iou_threshold, (list, tuple, set, np.ndarray, pd.Series)):
        # Instantiate lists
        accuracies = []
        recalls = []

        for i, thresh in enumerate(iou_threshold):
            print_iter(len(iou_threshold), i, 'IOU')

            # Create an empty DataFrame to add calculations per image
            calc_df = pd.DataFrame(columns=(['Patient', 'BBoxes Read 1', 'BBoxes Read 2', 'True Positives', 'False Positives', 'False Negatives']))

            for _, patient in enumerate(match_annos):
                # Get first- and second-read bounding boxes for patient
                read1_bboxes = get_bounding_boxes(patient, anno_df=first_reads)
                read2_bboxes = get_bounding_boxes(patient, anno_df=second_reads)

                # Calculate performance between bounding boxes
                true_pos, false_pos, false_neg, _ = calc_performance(read2_bboxes, read1_bboxes, iou_threshold=thresh)

                # Add values to calc_df
                calc_df = calc_df.append({'Patient' : patient,
                                          'BBoxes Read 1' : len(read1_bboxes),
                                          'BBoxes Read 2' : len(read2_bboxes),
                                          'True Positives' : true_pos,
                                          'False Positives' : false_pos,
                                          'False Negatives' : false_neg}, ignore_index=True)

            accuracy = calc_metric(calc_df['True Positives'].sum(), calc_df['False Positives'].sum(), calc_df['False Negatives'].sum(), 0, metric='accuracy')
            recall = calc_metric(calc_df['True Positives'].sum(), calc_df['False Positives'].sum(), calc_df['False Negatives'].sum(), 0, metric='recall')

            # Add values to lists
            accuracies.append(accuracy)
            recalls.append(recall)
        print('') # End print stream from loop

        # Write accuracies, recalls, and thresholds to CSV
        print('Writing to file...')
        with open(os.path.join(ARGS['COMPARE_READS_FOLDER'], 'perf_across_iou.csv'), 'w') as out_file:
            for thresh, acc, rec in zip(iou_threshold, accuracies, recalls):
                out_str = ','.join([str(thresh), str(acc), str(rec)]) + '\n'
                out_file.write(out_str)

    else:
        # Create an empty DataFrame to add calculations per image
        calc_df = pd.DataFrame(columns=(['Patient', 'BBoxes Read 1', 'BBoxes Read 2', 'True Positives', 'False Positives', 'False Negatives']))

        read1_box_areas = []
        read2_box_areas = []
        for i, patient in enumerate(match_annos):
            print_iter(len(match_annos), i, print_type='image')

            # Get first- and second-read bounding boxes for patient
            read1_bboxes = get_bounding_boxes(patient, anno_df=first_reads)
            if model:
                read2_bboxes, _ = get_bounding_boxes(patient, anno_df=second_reads, has_probs=True, conf_threshold=model_conf)
            else:
                read2_bboxes = get_bounding_boxes(patient, anno_df=second_reads)

            # Add bounding box areas for both reads to lists
            [read1_box_areas.append(calc_bbox_area(box)) for box in read1_bboxes]
            [read2_box_areas.append(calc_bbox_area(box)) for box in read2_bboxes]

            # Calculate performance between bounding boxes
            true_pos, false_pos, false_neg, _ = calc_performance(read2_bboxes, read1_bboxes, iou_threshold=iou_threshold)

            # Add values to calc_df
            calc_df = calc_df.append({'Patient' : patient,
                                      'BBoxes Read 1' : len(read1_bboxes),
                                      'BBoxes Read 2' : len(read2_bboxes),
                                      'True Positives' : true_pos,
                                      'False Positives' : false_pos,
                                      'False Negatives' : false_neg}, ignore_index=True)
        print('') # End print stream from loop

        # Calculate bounding box areas for each read
        read1_areas = np.array(read1_box_areas)
        read2_areas = np.array(read2_box_areas)

        # Pull out confusion matrix values and calculate metrics
        true_pos, false_pos, false_neg, true_neg = calc_df['True Positives'].sum(), calc_df['False Positives'].sum(), calc_df['False Negatives'].sum(), 0

        accuracy = calc_metric(true_pos, false_pos, false_neg, true_neg, metric='accuracy')
        precision = calc_metric(true_pos, false_pos, false_neg, true_neg, metric='precision')
        recall = calc_metric(true_pos, false_pos, false_neg, true_neg, metric='recall')
        f1_score = calc_metric(true_pos, false_pos, false_neg, true_neg, metric='f1_score')

        # Calculate Cohen's Kappa on the two annotation reads
        coh_kappa = calc_metric(true_pos, false_pos, false_neg, true_neg, metric='cohens_kappa')
        # Calculate Free-Response Kappa on the two annotation reads
        fr_kappa = calc_metric(true_pos, false_pos, false_neg, true_neg, metric='kappa_fr')

    if verbose:
        # Print out misc. confusion matrix stats
        print('')
        print('|{:^24}|{:^10}|{:^10}|'.format('METRIC', 'Read 1', 'RetinaNet'))
        print('|{}|'.format('-'*46))
        print('|{:^24}|{:^21}|'.format('Total Images', str(len(match_annos))))
        print('|{:^24}|{:^10}|{:^10}|'.format('Total Ribs Labeled', calc_df['BBoxes Read 1'].sum(), calc_df['BBoxes Read 2'].sum()))
        print('|{:^24}|{:^10.5}|{:^10.5}|'.format('Avg. Ribs/Image', calc_df['BBoxes Read 1'].mean(), calc_df['BBoxes Read 2'].mean()))
        print('|{}|'.format('-'*46))
        print('|{:^24}|{:^10.5}|{:^10.5}|'.format('Avg. Bounding Box Area', read1_areas.mean(), read2_areas.mean()))
        print('|{:^24}|{:^10.5}|{:^10.5}|'.format('Bounding Box Std. Dev', read1_areas.std(), read2_areas.std()))
        print('|{}|'.format('-'*46))
        print('|{:^24}|{:^21}|'.format('True Positives', true_pos))
        print('|{:^24}|{:^21}|'.format('False Positives', false_pos))
        print('|{:^24}|{:^21}|'.format('False Negatives', false_neg))
        print('|{:^24}|{:^21.3}|'.format('IOU Threshold', iou_threshold))
        print('|{:^24}|{:^21.3}|'.format('Model Confidence', model_conf if model else ''))
        print('|{:^24}|{:^21.5}|'.format('Accuracy', accuracy))
        print('|{:^24}|{:^21.5}|'.format('Precision', precision))
        print('|{:^24}|{:^21.5}|'.format('Recall/TPR/Sens', recall))
        print('|{:^24}|{:^21.5}|'.format('F1 Score', f1_score))
        print('|{:^24}|{:^21.5}|'.format('Cohen\'s Kappa', coh_kappa))
        print('|{:^24}|{:^21.5}|'.format('Free-Response Kappa', fr_kappa))
        print('')

def create_bbox_dataframe(first_reads, second_reads, iou_threshold):
    """
    Create a DataFrame with each bounding box for each patient for both reads.

    Parameters
    ----------
    first_reads : DataFrame
        contains image and bounding box locations from the first radiologist reads
    second_reads : DataFrame
        contains image and bounding box locations from the second radiologist reads
    """
    # Pull out unique PatientID.png from ID column of both reads
    read1_names = np.unique([name[name.rfind('/')+1:] for name in first_reads.ID])
    read2_names = np.unique([name[name.rfind('/')+1:] for name in second_reads.ID])

    # Find matching PatientIDs
    match_annos = np.intersect1d(read1_names, read2_names)

    # Create an empty DataFrame to add calculations per image
    bbox_df = pd.DataFrame(columns=(['Patient', 'Read1 Box', 'Read1 Area', 'Read2 Box', 'Read2 Area', 'Result', 'Max IOU']))

    for i, patient in enumerate(match_annos):
        print_iter(len(match_annos), i, print_type='image')

        # Get first- and second-read bounding boxes for patient
        read1_bboxes = get_bounding_boxes(patient, anno_df=first_reads)
        read2_bboxes = get_bounding_boxes(patient, anno_df=second_reads)

        # Loop through each read 1 box and find overlapping boxes from read 2
        for box1 in read1_bboxes:
            if len(read2_bboxes) > 0:
                # Loop through each read 2 box and calculate the IOU with the current read 1 box
                temp_ious = np.array([])
                for box2 in read2_bboxes:
                    temp_ious = np.append(temp_ious, intersection_over_union(box2, box1))

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
        # as false positrives
        if len(read2_bboxes) > 0:
            for box2 in read2_bboxes:
                bbox_df = bbox_df.append({'Patient' : patient,
                                          'Read1 Box' : None,
                                          'Read1 Area' : None,
                                          'Read2 Box' : box2,
                                          'Read2 Area' : calc_bbox_area(box2),
                                          'Result' : 'false_positive',
                                          'Max IOU' : 0}, ignore_index=True)
    print('') # End print stream from loop

    # Output bbox_df to a file
    print('Writing to file...')
    bbox_df.to_csv(os.path.join(ARGS['COMPARE_READS_FOLDER'], 'read1_vs_read2_bboxes.csv'), index=False)

def compute_afroc(ground_truths, model_predictions):
    """
    Calculates LLF and FPR for each threshold to plot an AFROC curve of the model performance

    Parameters
    ----------
    ground_truths : DataFrame
        contains image and bounding box locations from radiologist reads
    model_predictions : DataFrame
        contains image and bounding box locations with probabilities from the neural network
    """
    # Sort all the probabilities from the model
    sorted_scores = model_predictions.Prob.sort_values()

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

    # Loop through sorted_scores and calculate LLF and FPR for each score
    for i, threshold in enumerate(sorted_scores):
        print_iter(sorted_scores.size, i, print_type='threshold')

        total_true_pos = 0
        total_false_pos = 0
        # Loop through each patient and compare ground truth and predicted bounding boxes
        for _, patient in enumerate(match_annos):
            # Get ground truths and box predictions for current patient
            truth_boxes = get_bounding_boxes(patient, anno_df=ground_truths)
            pred_boxes, prob_scores = get_bounding_boxes(patient, anno_df=model_predictions, has_probs=True)

            good_preds = []
            for box, score in zip(pred_boxes, prob_scores):
                if score < threshold:
                    continue
                b = np.array(box, dtype=int)
                good_preds.append(b)

            # Calculate number of true and false positives for current patient
            true_pos, false_pos, _, _ = calc_performance(good_preds, truth_boxes)

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

    # Output values to a file
    print('Writing to file...')
    with open(os.path.join(ARGS['COMPARE_READS_FOLDER'], 'read1_afroc.csv'), 'w') as out_file:
        for threshold, llf, fpr in zip(sorted_scores, llf_list, fpr_list):
            out_file.write('{},{},{}\n'.format(threshold, llf, fpr))

def main(parse_args):
    """Main Function"""
    # Import first and second reads
    first_reads = pd.read_csv(parse_args.first_read_csv, names=('ID', 'Height', 'Width', 'x1', 'y1', 'x2', 'y2'))
    if parse_args.model:
        second_reads = pd.read_csv(parse_args.second_read_csv, names=('ID', 'Height', 'Width', 'x1', 'y1', 'x2', 'y2', 'Prob'))
    else:
        second_reads = pd.read_csv(parse_args.second_read_csv, names=('ID', 'Height', 'Width', 'x1', 'y1', 'x2', 'y2'))

    # Drop Height and Width columns
    first_reads = first_reads.drop(columns=['Height', 'Width'])
    second_reads = second_reads.drop(columns=['Height', 'Width'])

    if parse_args.images:
        make_images(first_reads, second_reads)

    if parse_args.color_images:
        make_images_from_file(parse_args.filename, first_reads)

    if parse_args.metrics:
        calculate_metrics(first_reads, second_reads, iou_threshold=parse_args.iou_thresh,
                          verbose=True, model=parse_args.model, model_conf=parse_args.model_conf)

    if parse_args.bboxes:
        create_bbox_dataframe(first_reads, second_reads, iou_threshold=parse_args.iou_thresh)

    if parse_args.plot:
        if parse_args.model:
            iou_threshold = second_reads.Prob.sort_values()
        else:
            iou_threshold = np.arange(0, 101, 1) / 100
        calculate_metrics(first_reads, second_reads, iou_threshold=iou_threshold,
                          verbose=False, model=parse_args.model, model_conf=parse_args.model_conf)

    if parse_args.afroc:
        compute_afroc(first_reads, second_reads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create images with annotations from two sets of reads, or calculate metrics of inter-reader relatability.')

    parser.add_argument('--images', action='store_true',
                        help='Create images with drawn-on annotations from both reads.')

    parser.add_argument('--color_images', action='store_true',
                        help='Create images with drawn-on annotations from both reads; boxes colored based on TP/FP/FN.')

    parser.add_argument('--metrics', action='store_true',
                        help='Calculate various performance metrics and print them out to console.')

    parser.add_argument('--bboxes', action='store_true',
                        help='Calculate statistics on bounding boxes between two reads and save to a CSV.')

    parser.add_argument('--plot', action='store_true',
                        help='Compare first and second radiologist reads across IOU thresholds and save to a CSV')

    parser.add_argument('--afroc', action='store_true',
                        help='Calculate LLF/FPR values for an AFROC plot to show model performance.')

    parser.add_argument('--model', action='store_true',
                        help='Boolean for whether model predictions are being used')
    
    parser.add_argument('--first_read_csv', type=str, default=os.path.join(ARGS['COMPARE_READS_FOLDER'], 'read1_annotations_crop.csv'),
                        help='Filename to CSV containing first read annotations. Also used a ground truth annotations in --afroc.')

    parser.add_argument('--second_read_csv', type=str, default=os.path.join(ARGS['COMPARE_READS_FOLDER'], 'read2_annotations_crop.csv'),
                        help='Filename to CSV containing second read annotations. Also used as model predictions.')

    parser.add_argument('--iou_thresh', type=float, default=0.30,
                        help='The threshold to use for determining if IOU counts toward a True Positive.')

    parser.add_argument('--model_conf', type=float, default=0.50,
                        help='The threshold to keep model bounding box predictions.')

    parser.add_argument('--filename', type=str, default=os.path.join(ARGS['COMPARE_READS_FOLDER'], 'read1_vs_read2_bboxes.csv'),
                        help='Filename of the CSV to output bounding box comparisons between reads.')

    parser_args = parser.parse_args()

    if not any([parser_args.images, parser_args.color_images, parser_args.metrics, parser_args.bboxes, parser_args.plot, parser_args.afroc]):
        parser.error('Please choose one of --images, --color_images, --metrics, --bboxes, --plot, or --afroc.')

    if parser_args.afroc and not parser_args.model:
        parser.error('Use --model flag when using --afroc.')

    # Print out start of execution
    print('Starting execution...')
    start_time = time.perf_counter()

    # Run main function
    main(parser_args)

    # Print out time to complete
    print('Done!')
    end_time = time.perf_counter()
    print('Execution finished in {} seconds.'.format(round(end_time - start_time, 3)))
