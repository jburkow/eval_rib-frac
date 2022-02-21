'''
Filename: ensemble.py
Author(s): Gregory Holste, giholste@gmail.com, UT Austin
           Jonathan Burkow, burkowjo@msu.edu, Michigan State University
Last Updated: 02/20/2022
Description: Takes in a list of model prediction CSVs and creates a new ensemble CSV using Non-
    Maximum Suppression and outputs as a single new prediction CSV.
'''

import argparse
import os
import time
from typing import List

import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms


def make_ensemble(model_pred_list: str, images_path: str, df_names: List[str] = None) -> pd.DataFrame:
    """
    Combine multiple prediction CSVs into a single ensemble DataFrame.

    Parameters
    ----------
    model_pred_list  : list of all model prediction CSVs
    images_path      : path to directory of images to add back in at the end
    df_names         : column headers for the DataFrame

    Returns
    -------
    ens_df : DataFrame of ensembled predictions
    """
    if not df_names:
        df_names = ['img_path', 'x1', 'y1', 'x2', 'y2', 'score']

    list_of_pred_dfs = []
    for model_pred_path in model_pred_list:
        temp_df = pd.read_csv(model_pred_path, names=df_names,
                              dtype={'x1': pd.Int64Dtype(), 'y1': pd.Int64Dtype(), 'x2': pd.Int64Dtype(), 'y2': pd.Int64Dtype()})
        list_of_pred_dfs.append(temp_df)

    pred_df = pd.concat(list_of_pred_dfs, ignore_index=True).sort_values(by=df_names[0], ignore_index=True)

    # Remove paths from images so that potentially different image locations in each ensemble
    # member doesn't cause duplicate boxes later.
    pred_df[df_names[0]] = pred_df.apply(lambda x: x[df_names[0]].split('/')[-1], axis=1)

    # For each image: (1) gather all detections across ensemble members + (2) perform NMS on that set of detections
    ens_img_paths = []
    ens_x1s, ens_y1s, ens_x2s, ens_y2s = [], [], [], []
    ens_scores = []
    for img_path in list(set(pred_df[df_names[0]])):
        # Subset data frame for all detections on image
        df = pred_df[pred_df[df_names[0]] == img_path]

        # Remove NAs... if nothing remains (no detections from any model), continue
        df = df.dropna(subset=df_names[1:])

        if df.shape[0] == 0:
            continue

        # Convert to float tensors for PyTorch NMS usage
        boxes = torch.from_numpy(df[['x1', 'y1', 'x2', 'y2']].values.astype(np.float64))
        scores = torch.from_numpy(df[df_names[-1]].values.astype(np.float64))

        # Run NMS and collect results
        keep = nms(boxes, scores, iou_threshold=0.55)

        ens_img_paths.extend([img_path]*keep.shape[0])
        ens_x1s.extend(boxes[keep, 0].int().tolist())
        ens_y1s.extend(boxes[keep, 1].int().tolist())
        ens_x2s.extend(boxes[keep, 2].int().tolist())
        ens_y2s.extend(boxes[keep, 3].int().tolist())
        ens_scores.extend(scores[keep].tolist())

    ens_df = pd.DataFrame({df_names[0]: ens_img_paths, 'x1': ens_x1s, 'y1': ens_y1s, 'x2': ens_x2s, 'y2': ens_y2s, df_names[-1]: ens_scores})
    ens_df[df_names[0]] = ens_df.apply(lambda x: images_path + x[df_names[0]], axis=1)

    # Sort by paths/names (ascending) then by probability score (descending)
    ens_df.sort_values(by=[df_names[0], df_names[-1]], inplace=True, ignore_index=True, ascending=[True, False])

    return ens_df


def main(args):
    """Main Function""" 
    ens_df = make_ensemble(args.preds, args.image_path)
    ens_df.to_csv(os.path.join(args.save_dir, f"{args.ensemble_name}.csv"), index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble predictions from multiple detection models into a single csv file.')

    parser.add_argument('--preds', nargs='+', type=str, required=True,
                        help='Space-delimited list of prediction csv files.')

    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to directory where ensembled csv will be saved.')

    parser.add_argument('--ensemble_name', type=str, required=True,
                        help='String describing the name of the final ensemble (dictates csv filename).')

    parser.add_argument('--image_path', type=str, required=True,
                        help='Path location to images to add to all rows before exporting to CSV.')

    args = parser.parse_args()

    # Print out start of execution
    print('Starting execution...')
    start_time = time.perf_counter()

    # Run main function
    main(args)

    # Print out time to complete
    print('Done!')
    end_time = time.perf_counter()
    print(f'Execution finished in {end_time - start_time:.3f} seconds.')
