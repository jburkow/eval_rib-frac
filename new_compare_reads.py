'''
Filename: new_compare_reads.py
Author(s): Jonathan Burkow, burkowjo@msu.edu, Michigan State University
Last Updated: 06/22/2022
Description: Updated version of script that either compare two radiologist reads, or one set of
    reads with model predictions.
'''

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from eval_utils import MetricsConfMatrix, calc_conf_matrix
from tqdm import tqdm


def parse_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--first_read_csv', type=str, help='Filename to CSV containing first read annotations; i.e., ground truth annotations')
    parser.add_argument('--second_read_csv', type=str, help='Filename to CSV containing second read annotations; i.e., model predictions.')
    parser.add_argument('--model_conf', type=float, default=0.50, help='The threshold to keep model bounding box predictions.')
    parser.add_argument('--iou_thresh', type=float, default=0.30, help='The threshold to use for determining if IOU counts toward a True Positive.')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset used for training.')
    parser.add_argument('--bootstrap_iters', type=int, default=5000, help='Number of iterations to use for bootstrapping to obtain 95% CIs for precision, recall, and F2 score.')
    parser.add_argument('--read_1_name', type=str, default='GT', help='Name for the title in the read 1 column of output table.')
    parser.add_argument('--read_2_name', type=str, default='YOLOv5', help='Name for the title in the read 2 column of output table.')

    return parser.parse_args()


def main():
    """Main Function"""
    parse_args = parse_cmd_args()

    ground_truth_df = pd.read_csv(parse_args.first_read_csv, names=['image', 'x1', 'y1', 'x2', 'y2', 'class'],
                                  dtype={'x1': pd.Int64Dtype(), 'y1': pd.Int64Dtype(), 'x2': pd.Int64Dtype(), 'y2': pd.Int64Dtype()})
    ground_truth_df.image = ground_truth_df.image.apply(lambda x: Path(x).stem)  # Remove image paths
    pred_df = pd.read_csv(parse_args.second_read_csv, names=['image', 'x1', 'y1', 'x2', 'y2', 'prob'],
                          dtype={'x1': pd.Int64Dtype(), 'y1': pd.Int64Dtype(), 'x2': pd.Int64Dtype(), 'y2': pd.Int64Dtype()})
    pred_df.image = pred_df.image.apply(lambda x: Path(x).stem)  # Remove image paths

    # Print out the number of unique images in each set
    print(f"{ground_truth_df.image.unique().size} unique ground truth images")
    print(f"{pred_df.image.unique().size} unique prediction images")

    # Print out the number of matching images in both sets
    matching_images = set(ground_truth_df.image.unique()) & set(pred_df.image.unique())
    print(f"{len(matching_images)} matching images in both ground truth and predictions")

    # Calculate number of fracture present and absent images in prediction set
    pred_frac_present = 0
    pred_frac_absent = 0
    unique_pred_images = pred_df.image.unique().tolist()
    for im in unique_pred_images:
        temp_df = pred_df[pred_df.image == im]
        temp_df = temp_df[temp_df.prob >= parse_args.model_conf]
        if len(temp_df) > 0:
            pred_frac_present += 1
        else:
            pred_frac_absent += 1
    print(f"{pred_frac_present} predicted fracture present images")
    print(f"{pred_frac_absent} predicted fracture absent images")

    info_df = pd.DataFrame(columns=['image', 'gt_is_frac_present', 'gt_is_frac_absent', 'gt_num_fractures', 'pred_is_frac_present', 'pred_is_frac_absent', 'pred_num_fractures', 'true_pos', 'false_pos', 'false_neg', 'true_neg', 'ious', 'overlaps'])

    for im in unique_pred_images:
        gt_image_df = ground_truth_df[ground_truth_df['image'] == im]
        gt_image_df = gt_image_df.dropna()
        gt_image_bounding_boxes = [[row.x1, row.y1, row.x2, row.y2] for row in gt_image_df.itertuples()]
        pred_image_df = pred_df[pred_df['image'] == im]
        pred_image_df = pred_image_df.dropna()
        pred_image_df = pred_image_df[pred_image_df['prob'] >= parse_args.model_conf]
        pred_image_bounding_boxes = [[row.x1, row.y1, row.x2, row.y2] for row in pred_image_df.itertuples()]

        true_pos, false_pos, false_neg, true_neg, ious, overlaps = calc_conf_matrix(pred_image_bounding_boxes, gt_image_bounding_boxes, iou_threshold=parse_args.iou_thresh)

        new_row_info = pd.DataFrame({'image': [im],
                                     'gt_is_frac_present': [1 if len(gt_image_df) > 0 else 0],
                                     'gt_is_frac_absent': [1 if len(gt_image_df) == 0 else 0],
                                     'gt_num_fractures': [len(gt_image_df)],
                                     'pred_is_frac_present': [1 if len(pred_image_df) > 0 else 0],
                                     'pred_is_frac_absent': [1 if len(pred_image_df) == 0 else 0],
                                     'pred_num_fractures': [len(pred_image_df)],
                                     'true_pos': [true_pos],
                                     'false_pos': [false_pos],
                                     'false_neg': [false_neg],
                                     'true_neg': [true_neg],
                                     'ious': [ious],
                                     'overlaps': [overlaps]})
        info_df = pd.concat([info_df, new_row_info], ignore_index=True)

    metric_calc = MetricsConfMatrix(info_df.true_pos.sum(), info_df.false_pos.sum(), info_df.false_neg.sum(), info_df.true_neg.sum())

    all_ious = np.array(list(itertools.chain(*info_df.ious)))
    all_overlaps = np.array(list(itertools.chain(*info_df.overlaps)))

    # Print out all computed metrics
    dataset_name = parse_args.dataset_name
    col_1_width = 25
    col_2a_width = max(10, len(parse_args.read_1_name) + 2)
    col_2b_width = max(10, len(parse_args.read_2_name) + 2)
    col_2_width = col_2a_width + col_2b_width + 1
    dash_width = col_1_width + col_2_width + 1
    print('')
    if dataset_name:
        print(f"|{'Dataset: ' + dataset_name:^{dash_width}}|")
        print(f"|{'-' * dash_width}|")
    print(f"|{'METRIC':^{col_1_width}}|{parse_args.read_1_name:^{col_2a_width}}|{parse_args.read_2_name:^{col_2b_width}}|")
    print(f"|{'-' * dash_width}|")
    print(f"|{'Total Images':^{col_1_width}}|{ground_truth_df.image.unique().size:^{col_2_width}}|")
    print(f"|{'Fracture Present Images':^{col_1_width}}|{info_df.gt_is_frac_present.sum():^{col_2a_width}}|{info_df.pred_is_frac_present.sum():^{col_2b_width}}|")
    print(f"|{'Fracture Absent Images':^{col_1_width}}|{info_df.gt_is_frac_absent.sum():^{col_2a_width}}|{info_df.pred_is_frac_absent.sum():^{col_2b_width}}|")
    print(f"|{'Total Ribs Labeled':^{col_1_width}}|{info_df.gt_num_fractures.sum():^{col_2a_width}}|{info_df.pred_num_fractures.sum():^{col_2b_width}}|")
    print(f"|{'Avg. Ribs/Image':^{col_1_width}}|{info_df.gt_num_fractures.mean():^{col_2a_width}.3}|{info_df.pred_num_fractures.mean():^{col_2b_width}.3}|")
    print(f"|{'StdDev. Ribs/Image':^{col_1_width}}|{info_df.gt_num_fractures.std():^{col_2a_width}.3}|{info_df.pred_num_fractures.std():^{col_2b_width}.3}|")
    print(f"|{'Min Ribs/Image':^{col_1_width}}|{info_df.gt_num_fractures.min():^{col_2a_width}}|{info_df.pred_num_fractures.min():^{col_2b_width}}|")
    print(f"|{'Max Ribs/Image':^{col_1_width}}|{info_df.gt_num_fractures.max():^{col_2a_width}}|{info_df.pred_num_fractures.max():^{col_2b_width}}|")
    print(f"|{'Median Ribs/Image':^{col_1_width}}|{info_df.gt_num_fractures.median():^{col_2a_width}}|{info_df.pred_num_fractures.median():^{col_2b_width}}|")
    print(f"|{'Q1 Ribs/Image':^{col_1_width}}|{info_df.gt_num_fractures.quantile(0.25):^{col_2a_width}.2}|{info_df.pred_num_fractures.quantile(0.25):^{col_2b_width}.2}|")
    print(f"|{'Q3 Ribs/Image':^{col_1_width}}|{info_df.gt_num_fractures.quantile(0.75):^{col_2a_width}.2}|{info_df.pred_num_fractures.quantile(0.75):^{col_2b_width}.2}|")
    print(f"|{'IQR Ribs/Image':^{col_1_width}}|{info_df.gt_num_fractures.quantile(0.75) - info_df.gt_num_fractures.quantile(0.25):^{col_2a_width}.2}|{info_df.pred_num_fractures.quantile(0.75) - info_df.pred_num_fractures.quantile(0.25):^{col_2b_width}.2}|")
    print(f"|{'-' * dash_width}|")
    print(f"|{'IOU Threshold':^{col_1_width}}|{parse_args.iou_thresh:^{col_2_width}.3}|")
    print(f"|{'Avg. Percent Overlap':^{col_1_width}}|{(all_overlaps.mean() if all_overlaps.size > 0 else 0.0):^{col_2_width}.3}|")
    print(f"|{'Avg. IOU':^{col_1_width}}|{(all_ious.mean() if all_ious.size > 0 else 0.0):^{col_2_width}.3}|")
    print(f"|{'Model Confidence':^{col_1_width}}|{parse_args.model_conf:^{col_2_width}}|")
    print(f"|{'-' * dash_width}|")
    print(f"|{'True Positives':^{col_1_width}}|{metric_calc.true_pos:^{col_2_width}}|")
    print(f"|{'False Positives':^{col_1_width}}|{metric_calc.false_pos:^{col_2_width}}|")
    print(f"|{'False Negatives':^{col_1_width}}|{metric_calc.false_neg:^{col_2_width}}|")
    print(f"|{'True Negatives':^{col_1_width}}|{metric_calc.true_neg:^{col_2_width}}|")
    print(f"|{'Accuracy':^{col_1_width}}|{metric_calc.accuracy():^{col_2_width}.3}|")
    print(f"|{'Precision':^{col_1_width}}|{metric_calc.precision():^{col_2_width}.3}|")
    print(f"|{'Recall/TPR/Sens':^{col_1_width}}|{metric_calc.recall():^{col_2_width}.3}|")
    print(f"|{'F1 Score':^{col_1_width}}|{float(metric_calc.f1_score()):^{col_2_width}.3}|")
    print(f"|{'F2 Score':^{col_1_width}}|{float(metric_calc.f2_score()):^{col_2_width}.3}|")
    print(f"|{'Cohens Kappa':^{col_1_width}}|{metric_calc.cohens_kappa():^{col_2_width}.3}|")
    print(f"|{'Free-Response Kappa':^{col_1_width}}|{metric_calc.free_kappa():^{col_2_width}.3}|")
    print('')

    # Bootstrapping to obtain 95% CIs and measures of variation in precision, recall, and F2 score
    if parse_args.bootstrap_iters > 0:
        boot_precisions, boot_recalls, boot_f2s = [], [], []
        for i in tqdm(range(parse_args.bootstrap_iters), desc='Bootstrapping...'):
            # Take stratified bootstrap sample: n_pos fracture-present cases w/ replacement + n_neg fracture-absent cases w/ replacement
            bootstrap_df_pos = info_df[info_df.gt_is_frac_present == 1].sample(frac=1, replace=True, random_state=i)
            bootstrap_df_neg = info_df[info_df.gt_is_frac_absent == 1].sample(frac=1, replace=True, random_state=i)
            bootstrap_df = pd.concat([bootstrap_df_pos, bootstrap_df_neg])

            true_pos, false_pos, false_neg, true_neg = bootstrap_df.true_pos.sum(), bootstrap_df.false_pos.sum(), bootstrap_df.false_neg.sum(), bootstrap_df.true_neg.sum()

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

        boot_precision_lb, boot_precision_ub = sorted_boot_precisions[int(0.05 * len(sorted_boot_precisions))], sorted_boot_precisions[int(0.95 * len(sorted_boot_precisions))]
        boot_recall_lb, boot_recall_ub = sorted_boot_recalls[int(0.05 * len(sorted_boot_recalls))], sorted_boot_recalls[int(0.95 * len(sorted_boot_recalls))]
        boot_f2_lb, boot_f2_ub = sorted_boot_f2s[int(0.05 * len(sorted_boot_f2s))], sorted_boot_f2s[int(0.95 * len(sorted_boot_f2s))]

        print("")
        print(f"{'':^23} | {'Precision':^15} | {'Recall':^15} | {'F2 Score':^15} |")
        print(f"{'Mean +/- StDev':>23} | {np.mean(boot_precisions):.3f} +/- {np.std(boot_precisions):.3f} | {np.mean(boot_recalls):.3f} +/- {np.std(boot_recalls):.3f} | {np.mean(boot_f2s):.3f} +/- {np.std(boot_f2s):.3f} |")
        print(f"{'95% CI':>23} | ({boot_precision_lb:.3f}, {boot_precision_ub:.3f})  | ({boot_recall_lb:.3f}, {boot_recall_ub:.3f})  | ({boot_f2_lb:.3f}, {boot_f2_ub:.3f})  |")
        print(f"{'95% CI Range':>23} | {boot_precision_ub - boot_precision_lb:^15.3f} | {boot_recall_ub - boot_recall_lb:^15.3f} | {boot_f2_ub - boot_f2_lb:^15.3f} |")
        print(f"{'Coeff. of Variation':>23} | {np.std(boot_precisions) / np.mean(boot_precisions):^15.3f} | {np.std(boot_recalls) / np.mean(boot_recalls):^15.3f} | {np.std(boot_f2s) / np.mean(boot_f2s):^15.3f} |")
        print("")


if __name__ == "__main__":
    print(f"\n{'Starting execution: ' + Path(__file__).name:-^80}\n")
    start_time = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start_time
    print(f"\n{'Done!':-^80}")
    print(f'Execution finished in {elapsed:.3f} seconds ({time.strftime("%-H hr, %-M min, %-S sec", time.gmtime(elapsed))}).\n')
