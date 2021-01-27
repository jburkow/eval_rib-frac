'''
Filename: compare_reads.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 01/26/2021
Description: Goes through first and second read annotation files, draws
    boxes on the corresponding image, and saves them as JPEG files.
'''

import os
import time
import numpy as np
import pandas as pd
import cv2
import args
from general_utils import print_iter

def main():
    """Main Function"""
    # Import first and second reads
    first_reads = pd.read_csv(os.path.join(args.ARGS['COMPARE_READS_FOLDER'], 'read1_annotations.csv'), names=('ID', 'Height', 'Width', 'x1', 'y1', 'x2', 'y2'))
    second_reads = pd.read_csv(os.path.join(args.ARGS['COMPARE_READS_FOLDER'], 'read2_annotations.csv'), names=('ID', 'Height', 'Width', 'x1', 'y1', 'x2', 'y2'))

    # Drop Height and Width columns
    first_reads = first_reads.drop(columns=['Height', 'Width'])
    second_reads = second_reads.drop(columns=['Height', 'Width'])

    # Define list of images
    img_list = [os.path.join(root, file) for root, dirs, files in os.walk(os.path.join(args.ARGS['8_BIT_FOLDER'], args.ARGS['ORIGINAL_IMAGE_FOLDER'])) for file in files]

    # Loop through original/cropped images, draw annotations, and save JPEGs
    for i, img_nm in enumerate(img_list):
        print_iter(len(img_list), i, 'image')

        # Pull the Patient ID from the annotations file
        patient_id = img_nm[img_nm.rfind('Anon_'):-4]

        # Set temp DataFrames to pull annotations from
        patient_read1 = first_reads[first_reads['ID'].str.contains(patient_id)]
        patient_read2 = second_reads[second_reads['ID'].str.contains(patient_id)]

        # Set box colors and thickness
        box_color1 = [255, 255, 0] # cv2 saves JPG as BGR -> this is teal
        box_color2 = [0, 255, 255] # cv2 saves JPG as BGR -> this is yellow
        box_thick = 2

        # Import image
        img = cv2.imread(img_nm)

        # Loop through first read annotations and draw boxes
        for _, row in patient_read1.iterrows():
            b = np.array([row[1], row[2], row[3], row[4]]).astype(int)
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), box_color1, box_thick, cv2.LINE_AA)

        # Loop through second read annotations and draw boxes
        for _, row in patient_read2.iterrows():
            b = np.array([row[1], row[2], row[3], row[4]]).astype(int)
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), box_color2, box_thick, cv2.LINE_AA)

        # Save image to file
        cv2.imwrite(os.path.join(args.ARGS['COMPARE_READS_IMAGES_FOLDER'], patient_id + '.jpg'), img)
    print('') # End print stream from loop


if __name__ == "__main__":
    # Print out start of execution
    print('Starting execution...')
    start_time = time.perf_counter()

    # Run main function
    main()

    # Print out time to complete
    print('Done!')
    end_time = time.perf_counter()
    print('Execution finished in {} seconds.'.format(round(end_time - start_time, 3)))
