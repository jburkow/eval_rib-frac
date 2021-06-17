import os
import numpy as np
import pandas as pd
import cv2
import argparse

import torch
from torchvision.ops import nms

def main(args):
    # Take union of all models' predictions (concatenate dataframes along row axis) and sort by ID
    pred_dfs = [pd.read_csv(pred_path, names=['img_path', 'x1', 'y1', 'x2', 'y2', 'score'], dtype={'x1': pd.Int64Dtype(), 'y1': pd.Int64Dtype(), 'x2': pd.Int64Dtype(), 'y2': pd.Int64Dtype()}) for pred_path in args.preds]
    pred_df = pd.concat(pred_dfs, ignore_index=True).sort_values(by='img_path', ignore_index=True)

    # For each image: (1) gather all detections across ensemble members + (2) perform NMS on that set of detections
    ens_img_paths = []
    ens_x1s, ens_y1s, ens_x2s, ens_y2s = [], [], [], []
    ens_scores = []
    for img_path in list(set(pred_df['img_path'])):
        # Subset data frame for all detections on image
        df = pred_df[pred_df['img_path'] == img_path]

        # Remove NAs... if nothing remains (no detections from any model), continue
        df = df.dropna(subset=['x1', 'y1', 'x2', 'y2', 'score'])

        if df.shape[0] == 0:
            continue

        # Convert to float tensors for PyTorch NMS usage
        boxes = torch.from_numpy(df[['x1', 'y1', 'x2', 'y2']].values.astype(np.float64))
        scores = torch.from_numpy(df['score'].values.astype(np.float64))

        # Run NMS and collect results
        keep = nms(boxes, scores, iou_threshold=0.55)

        ens_img_paths.extend([img_path]*keep.shape[0])
        ens_x1s.extend(boxes[keep, 0].int().tolist())
        ens_y1s.extend(boxes[keep, 1].int().tolist())
        ens_x2s.extend(boxes[keep, 2].int().tolist())
        ens_y2s.extend(boxes[keep, 3].int().tolist())
        ens_scores.extend(scores[keep].tolist())

    # Create resulting ensembled prediction data frame and save as csv
    ens_df = pd.DataFrame({'img_path': ens_img_paths, 'x1': ens_x1s, 'y1': ens_y1s, 'x2': ens_x2s, 'y2': ens_y2s, 'score': ens_scores})
    ens_df.to_csv(os.path.join(args.save_dir, args.ensemble_name + '.csv'), index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble predictions from multiple detection models into a single csv file.')

    parser.add_argument('--preds', nargs='+', type=str, required=True, help='Space-delimited list of prediction csv files.')
    parser.add_argument('--save_dir', type=str, help='Path to directory where ensembled csv will be saved.',
                        default='/mnt/research/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/ensemble_preds/')
    parser.add_argument('--ensemble_name', type=str, required=True, help='String descrbing the name of your ensemble.')

    args = parser.parse_args()

    main(args)
