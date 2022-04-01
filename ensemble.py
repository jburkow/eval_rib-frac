'''
Filename: ensemble.py
Author(s): Jonathan Burkow, burkowjo@msu.edu, Michigan State University
           Gregory Holste, giholste@gmail.com, UT Austin
Last Updated: 03/21/2022
Description: Takes in a list of model prediction CSVs and creates a new ensemble CSV using Non-
    Maximum Suppression and outputs as a single new prediction CSV.
'''

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms

from general_utils import convert_seconds


def non_max_suppression(
        data_df: pd.DataFrame,
        df_names: List[str] = None,
        iou_thresh=0.55
    ) -> Tuple[List[int], List[int], List[int], List[int], List[float], List[bool]]:
    """
    Applies non-maximum suppression (NMS) to the provided data_df and outputs all remaining bounding
    boxes and the kept indices.

    Parameters
    ----------
    data_df    : DataFrame of current patient's bounding box predictions
    df_names   : column headers for the DataFrame
    iou_thresh : IOU threshold for discarded boxes during non-max suppression

    Returns
    -------
    x1_list    : list of all top left x-values of NMS-accepted bounding boxes
    y1_list    : list of all top left y-values of NMS-accepted bounding boxes
    x2_list    : list of all bottom left x-values of NMS-accepted bounding boxes
    y2_list    : list of all bottom left y-values of NMS-accepted bounding boxes
    score_list : list of all corresponding scores of bounding boxes
    keep       : boolean list of accepted bounding boxes from NMS
    """
    # Convert to float tensors for PyTorch NMS usage
    boxes = torch.from_numpy(data_df[['x1', 'y1', 'x2', 'y2']].values.astype(np.float64))
    scores = torch.from_numpy(data_df[df_names[-1]].values.astype(np.float64))

    # Run NMS and collect results
    keep = nms(boxes, scores, iou_threshold=iou_thresh)

    x1_list = boxes[keep, 0].int().tolist()
    y1_list = boxes[keep, 1].int().tolist()
    x2_list = boxes[keep, 2].int().tolist()
    y2_list = boxes[keep, 3].int().tolist()
    score_list = scores[keep].tolist()

    return x1_list, y1_list, x2_list, y2_list, score_list, keep


def apply_nms(
        data_df: pd.DataFrame,
        df_names: List[str] = None,
        iou_thresh: float = 0.55
    ) -> pd.DataFrame:
    # For each image: (1) gather all detections across ensemble members + (2) perform NMS on that set of detections
    img_paths = []
    x1_list, y1_list, x2_list, y2_list = [], [], [], []
    all_scores = []
    for img_path in list(set(data_df[df_names[0]])):
        # Subset data frame for all detections on image
        df = data_df[data_df[df_names[0]] == img_path]

        # Remove NAs... if nothing remains (no detections from any model), continue
        df = df.dropna(subset=df_names[1:])

        if df.shape[0] == 0:
            continue

        patient_x1s, patient_y1s, patient_x2s, patient_y2s, score_list, keep = non_max_suppression(df, df_names, iou_thresh)

        img_paths.extend([img_path]*keep.shape[0])
        x1_list.extend(patient_x1s)
        y1_list.extend(patient_y1s)
        x2_list.extend(patient_x2s)
        y2_list.extend(patient_y2s)
        all_scores.extend(score_list)

    return pd.DataFrame({df_names[0]: img_paths, 'x1': x1_list, 'y1': y1_list, 'x2': x2_list, 'y2': y2_list, df_names[-1]: all_scores})


def make_ensemble(
        model_pred_list: str,
        images_path: str,
        df_names: List[str] = None,
        nms_iou_thresh: float = 0.55
    ) -> pd.DataFrame:
    """
    Combine multiple prediction CSVs into a single ensemble DataFrame.

    Parameters
    ----------
    model_pred_list : list of all model prediction CSVs
    images_path     : path to directory of images to add back in at the end
    df_names        : column headers for the DataFrame
    nms_iou_thresh  : iou threshold for removing overlapping boxes during NMS

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

    # Apply Non-Max Suppression to the data
    ens_df = apply_nms(pred_df, df_names, iou_thresh=nms_iou_thresh)

    # Apply new image paths, then sort by paths/names (ascending) then by prob. score (descending)
    ens_df[df_names[0]] = ens_df.apply(lambda x: images_path + x[df_names[0]], axis=1)
    ens_df.sort_values(by=[df_names[0], df_names[-1]], inplace=True, ignore_index=True, ascending=[True, False])

    return ens_df


def main(args) -> None:
    """Main Function""" 
    ens_df = make_ensemble(args.preds, args.image_path)
    ens_df.to_csv(os.path.join(args.save_dir, f"{args.ensemble_name}.csv"), index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--preds', nargs='+', type=str, required=True,
                        help='Space-delimited list of prediction csv files.')

    parser.add_argument('--save_dir', type=str, required=True,
                        help='Path to directory where ensembled csv will be saved.')

    parser.add_argument('--ensemble_name', type=str, required=True,
                        help='String describing the name of the final ensemble (dictates csv filename).')

    parser.add_argument('--image_path', type=str, required=True,
                        help='Path location to images to add to all rows before exporting to CSV.')

    args = parser.parse_args()

    print('\nStarting execution...')
    start_time = time.perf_counter()
    main(args)
    elapsed = time.perf_counter() - start_time
    print('Done!')
    print(f'Execution finished in {elapsed:.3f} seconds ({convert_seconds(elapsed)}).\n')
