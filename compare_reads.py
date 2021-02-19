'''
Filename: compare_reads.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 02/10/2021
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
from eval_utils import draw_box, draw_caption, get_bounding_boxes, calc_performance, calc_cohens, calc_kappa_fr, calc_bbox_area, intersection_over_union

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
    img_list = [os.path.join(root, file) for root, dirs, files in os.walk(ARGS['8_BIT_OG_IMAGE_FOLDER']) for file in files]

    # Loop through original/cropped images, draw annotations, and save JPEGs
    for i, img_nm in enumerate(img_list):
        print_iter(len(img_list), i, 'image')

        # Pull the Patient ID from the annotations file
        patient_id = img_nm[img_nm.rfind('Anon_'):-4]

        # Set temp DataFrames to pull annotations from
        patient_read1 = first_reads[first_reads['ID'].str.contains(patient_id)]
        patient_read2 = second_reads[second_reads['ID'].str.contains(patient_id)]

        # Set box colors
        box_color1 = [255, 255, 0] # cv2 saves JPG as BGR -> this is teal
        box_color2 = [0, 255, 255] # cv2 saves JPG as BGR -> this is yellow

        # Import image
        img = cv2.imread(img_nm)

        # Loop through first read annotations and draw boxes
        for _, row in patient_read1.iterrows():
            box = [row[1], row[2], row[3], row[4]]
            draw_box(img, box, box_color1)

        # Loop through second read annotations and draw boxes
        for _, row in patient_read2.iterrows():
            box = [row[1], row[2], row[3], row[4]]
            draw_box(img, box, box_color2)

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
    img_list = [os.path.join(root, file) for root, dirs, files in os.walk(ARGS['8_BIT_OG_IMAGE_FOLDER']) for file in files]

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

def calculate_metrics(first_reads, second_reads, iou_threshold):
    """
    Calculates Cohen's Kappa across the dataset annotated in both reads.

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
    calc_df = pd.DataFrame(columns=(['Patient', 'BBoxes Read 1', 'BBoxes Read 2', 'True Positives', 'False Positives', 'False Negatives']))

    for i, patient in enumerate(match_annos):
        print_iter(len(match_annos), i, print_type='image')

        # Get first- and second-read bounding boxes for patient
        read1_bboxes = get_bounding_boxes(patient, anno_df=first_reads)
        read2_bboxes = get_bounding_boxes(patient, anno_df=second_reads)

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

    accuracy = calc_df['True Positives'].sum() / (calc_df['True Positives'].sum() + calc_df['False Positives'].sum() + calc_df['False Negatives'].sum())
    precision = calc_df['True Positives'].sum() / (calc_df['True Positives'].sum() + calc_df['False Positives'].sum())
    recall = calc_df['True Positives'].sum() / (calc_df['True Positives'].sum() + calc_df['False Negatives'].sum())
    f1_score = 2 * (recall * precision) / (recall + precision)
    # Calculate Cohen's Kappa on the two annotation reads
    coh_kappa = calc_cohens(calc_df['True Positives'].sum(), calc_df['False Positives'].sum(), calc_df['False Negatives'].sum(), 0)
    # Calculate Free-Response Kappa on the two annotation reads
    fr_kappa = calc_kappa_fr(calc_df['True Positives'].sum(), calc_df['False Positives'].sum(), calc_df['False Negatives'].sum())

    # Print out misc. confusion matrix stats
    print('{:^20}|{:^10}|{:^10}|'.format('METRIC', '1', '2'))
    print('-'*43)
    print('{:20}|{:^21}|'.format('Total Images', str(len(match_annos))))
    print('{:20}|{:^10}|{:^10}|'.format('Total Ribs Labeled', calc_df['BBoxes Read 1'].sum(), calc_df['BBoxes Read 2'].sum()))
    print('{:20}|{:^10.5}|{:^10.5}|'.format('Avg. Ribs/Image', calc_df['BBoxes Read 1'].mean(), calc_df['BBoxes Read 2'].mean()))
    print('{:20}|{:^21.5}|'.format('Accuracy', accuracy))
    print('{:20}|{:^21.5}|'.format('Precision', precision))
    print('{:20}|{:^21.5}|'.format('Recall/TPR/Sens', recall))
    print('{:20}|{:^21.5}|'.format('F1 Score', f1_score))
    print('{:20}|{:^21.5}|'.format('Cohen\'s Kappa', coh_kappa))
    print('{:20}|{:^21.5}|'.format('Free-Response Kappa', fr_kappa))

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

    return bbox_df

def main(parse_args):
    """Main Function"""
    # Import first and second reads
    first_reads = pd.read_csv(parse_args.first_read_csv, names=('ID', 'Height', 'Width', 'x1', 'y1', 'x2', 'y2'))
    second_reads = pd.read_csv(parse_args.second_read_csv, names=('ID', 'Height', 'Width', 'x1', 'y1', 'x2', 'y2'))

    # Drop Height and Width columns
    first_reads = first_reads.drop(columns=['Height', 'Width'])
    second_reads = second_reads.drop(columns=['Height', 'Width'])

    if parse_args.type == 'images':
        make_images(first_reads, second_reads)

    if parse_args.type == 'color_images':
        make_images_from_file(parse_args.filename, first_reads)

    if parse_args.type == 'metrics':
        calculate_metrics(first_reads, second_reads, iou_threshold=parse_args.iou_thresh)

    if parse_args.type == 'file':
        bbox_df = create_bbox_dataframe(first_reads, second_reads, iou_threshold=parse_args.iou_thresh)

        # Output bbox_df to a file
        print('Writing to file...')
        bbox_df.to_csv(parse_args.filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create images with annotations from two sets of reads, or calculate metrics of inter-reader relatability.')

    parser.add_argument('--type', required=True, choices=['images', 'color_images', 'metrics', 'file'],
                        help='Determine whether to create annotated images or calculate metrics across the dataset.')

    parser.add_argument('--first_read_csv', type=str, default=os.path.join(ARGS['COMPARE_READS_FOLDER'], 'read1_annotations.csv'),
                        help='Filename to CSV containing first read annotations.')

    parser.add_argument('--second_read_csv', type=str, default=os.path.join(ARGS['COMPARE_READS_FOLDER'], 'read2_annotations.csv'),
                        help='Filename to CSV containing second read annotations.')

    parser.add_argument('--iou_thresh', type=float, default=0.3,
                        help='The threshold to use for determining if IOU counts toward a True Positive.')

    parser.add_argument('--filename', type=str, default=os.path.join(ARGS['COMPARE_READS_FOLDER'], 'read1_vs_read2.csv'),
                        help='Filename of the CSV to output bounding box comparisons between reads.')

    parser_args = parser.parse_args()

    # Print out start of execution
    print('Starting execution...')
    start_time = time.perf_counter()

    # Run main function
    main(parser_args)

    # Print out time to complete
    print('Done!')
    end_time = time.perf_counter()
    print('Execution finished in {} seconds.'.format(round(end_time - start_time, 3)))
