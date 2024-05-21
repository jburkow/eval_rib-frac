'''
Filename: ensemble.py
Author(s): Jonathan Burkow, burkowjo@msu.edu, Michigan State University
           Gregory Holste, giholste@gmail.com, UT Austin
Last Updated: 05/15/2024
Description: Takes in a list of model prediction CSVs and creates a new ensemble CSV using Non-
    Maximum Suppression and outputs as a single new prediction CSV.
'''

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from avalanche_predictions import AVALANCHE_CONFIGS, get_avalanche_df
from eval_utils import apply_nms


def parse_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--preds', nargs='+', type=str, required=True, help='Space-delimited list of prediction csv files.')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to directory where ensembled csv will be saved.')
    parser.add_argument('--ensemble_name', type=str, required=True, help='String describing the name of the final ensemble (dictates csv filename).')
    parser.add_argument('--image_path', type=str, required=True, help='Path location to images to add to all rows before exporting to CSV.')
    parser.add_argument('--avalanche', type=str, choices=['standard', 'posterior', 'conservative', 'gamma15', 'gamma20'], help='Avalanche decision scheme to apply to the ensemble.')

    return parser.parse_args()


def make_ensemble(
        model_pred_list: str,
        images_path: str,
        df_names: list[str] = None,
        nms_iou_thresh: float = 0.55) -> pd.DataFrame:
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


def main() -> None:
    """Main Function"""
    args = parse_cmd_args()

    ens_df = make_ensemble(args.preds, args.image_path)
    if args.avalanche:
        base_value = AVALANCHE_CONFIGS[args.avalanche]['base_val']
        rate = AVALANCHE_CONFIGS[args.avalanche]['rate']
        ens_df = get_avalanche_df(ens_df, 'avalanche' if 'gamma' in args.avalanche else args.avalanche, base_value, rate, add_nms=True, df_cols=['img_path', 'x1', 'y1', 'x2', 'y2', 'score'])
    ens_df.to_csv(os.path.join(args.save_dir, f"{args.ensemble_name}.csv"), index=False, header=False)


if __name__ == "__main__":
    print(f"\n{'Starting execution: ' + Path(__file__).name:-^80}\n")
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    print(f"\n{'Done!':-^80}")
    print(f'Execution finished in {elapsed:.3f} seconds ({time.strftime("%-H hr, %-M min, %-S sec", time.gmtime(elapsed))}).\n')
