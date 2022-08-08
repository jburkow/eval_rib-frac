'''
Filename: multi_model_metrics.py
Author(s): Jonathan Burkow, burkowjo@msu.edu, Michigan State University
Last Updated: 04/18/2022
Description: Combines multiple model predictions and automatically calculates the avg +/- std.dev
    of precision, recall, and F2 scores from them all.
'''

import argparse
import random
import time
from itertools import combinations
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from args import ARGS
from avalanche_predictions import get_avalanche_df
from compare_reads import compute_afroc
from ensemble import make_ensemble
from eval_utils import (MetricsConfMatrix, calc_conf_matrix, calc_mAP,
                        get_bounding_boxes)
from general_utils import print_elapsed

RETINANET_MODEL_PREDS = {
    "model_0" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-02-07_6fold-splits/20220207_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split0/20220207_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split0_predictions.csv",
    "model_1" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-02-07_6fold-splits/20220207_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split1/20220207_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split1_predictions.csv",
    "model_2" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-02-07_6fold-splits/20220207_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split2/20220207_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split2_predictions.csv",
    "model_3" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-02-07_6fold-splits/20220207_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split3/20220207_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split3_predictions.csv",
    "model_4" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-02-07_6fold-splits/20220207_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split4/20220207_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split4_predictions.csv",
    "model_5" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-02-07_6fold-splits/20220207_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split5/20220207_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split5_predictions.csv"
}

RETINANET_BIN_BIN_BIN_MODEL_PREDS = {
    "model_0" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_bin-bin-bin/20220304_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split0/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split0_bin-bin-bin_predictions.csv",
    "model_1" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_bin-bin-bin/20220304_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split1/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split1_bin-bin-bin_predictions.csv",
    "model_2" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_bin-bin-bin/20220304_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split2/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split2_bin-bin-bin_predictions.csv",
    "model_3" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_bin-bin-bin/20220304_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split3/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split3_bin-bin-bin_predictions.csv",
    "model_4" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_bin-bin-bin/20220304_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split4/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split4_bin-bin-bin_predictions.csv",
    "model_5" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_bin-bin-bin/20220304_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split5/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split5_bin-bin-bin_predictions.csv"
}

RETINANET_RAW_HISTEQ_BI_MODEL_PREDS = {
    "model_0" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_raw-hist-bi/20220304_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split0/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split0_raw-hist-bi_predictions.csv",
    "model_1" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_raw-hist-bi/20220304_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split1/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split1_raw-hist-bi_predictions.csv",
    "model_2" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_raw-hist-bi/20220304_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split2/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split2_raw-hist-bi_predictions.csv",
    "model_3" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_raw-hist-bi/20220304_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split3/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split3_raw-hist-bi_predictions.csv",
    "model_4" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_raw-hist-bi/20220304_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split4/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split4_raw-hist-bi_predictions.csv",
    "model_5" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_raw-hist-bi/20220304_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split5/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split5_raw-hist-bi_predictions.csv"
}

## THESE ARE OLD WITH NMS IOU_THRESH=0.3
# YOLO_MODEL_PREDS = {
#     "model_0" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-02-07_yolov5l6_pretrained_300epoch_bs8_split0/2022-02-07_yolov5l6_pretrained_300epoch_bs8_split0_model_predictions.csv",
#     "model_1" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-02-07_yolov5l6_pretrained_300epoch_bs8_split1/2022-02-07_yolov5l6_pretrained_300epoch_bs8_split1_model_predictions.csv",
#     "model_2" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-02-07_yolov5l6_pretrained_300epoch_bs8_split2/2022-02-07_yolov5l6_pretrained_300epoch_bs8_split2_model_predictions.csv",
#     "model_3" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-02-07_yolov5l6_pretrained_300epoch_bs8_split3/2022-02-07_yolov5l6_pretrained_300epoch_bs8_split3_model_predictions.csv",
#     "model_4" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-02-07_yolov5l6_pretrained_300epoch_bs8_split4/2022-02-07_yolov5l6_pretrained_300epoch_bs8_split4_model_predictions.csv",
#     "model_5" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-02-07_yolov5l6_pretrained_300epoch_bs8_split5/2022-02-07_yolov5l6_pretrained_300epoch_bs8_split5_model_predictions.csv"
# }

## NMS IOU_THRESH=0.45
YOLO_MODEL_PREDS = {
    "model_0" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split0/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split0_model_predictions.csv",
    "model_1" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split1/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split1_model_predictions.csv",
    "model_2" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split2/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split2_model_predictions.csv",
    "model_3" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split3/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split3_model_predictions.csv",
    "model_4" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split4/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split4_model_predictions.csv",
    "model_5" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split5/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split5_model_predictions.csv"
}

# ## NMS IOU_THRESH=0.95 CONF=0.01
# YOLO_MODEL_PREDS = {
#     "model_0" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-04-11_yolov5l6_pretrained_300epoch_bs8_split0_nms-iou0.95_conf0.01/2022-04-11_yolov5l6_pretrained_300epoch_bs8_split0_model-conf0_01_nms-iou0_95_model_predictions.csv",
#     "model_1" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-04-11_yolov5l6_pretrained_300epoch_bs8_split1_nms-iou0.95_conf0.01/2022-04-11_yolov5l6_pretrained_300epoch_bs8_split1_model-conf0_01_nms-iou0_95_model_predictions.csv",
#     "model_2" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-04-11_yolov5l6_pretrained_300epoch_bs8_split2_nms-iou0.95_conf0.01/2022-04-11_yolov5l6_pretrained_300epoch_bs8_split2_model-conf0_01_nms-iou0_95_model_predictions.csv",
#     "model_3" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-04-11_yolov5l6_pretrained_300epoch_bs8_split3_nms-iou0.95_conf0.01/2022-04-11_yolov5l6_pretrained_300epoch_bs8_split3_model-conf0_01_nms-iou0_95_model_predictions.csv",
#     "model_4" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-04-11_yolov5l6_pretrained_300epoch_bs8_split4_nms-iou0.95_conf0.01/2022-04-11_yolov5l6_pretrained_300epoch_bs8_split4_model-conf0_01_nms-iou0_95_model_predictions.csv",
#     "model_5" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-04-11_yolov5l6_pretrained_300epoch_bs8_split5_nms-iou0.95_conf0.01/2022-04-11_yolov5l6_pretrained_300epoch_bs8_split5_model-conf0_01_nms-iou0_95_model_predictions.csv"
# }

YOLO_BIN_BIN_BIN_MODEL_PREDS = {
    "model_0" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split0_bin-bin-bin/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split0_bin-bin-bin_model_predictions.csv",
    "model_1" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split1_bin-bin-bin/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split1_bin-bin-bin_model_predictions.csv",
    "model_2" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split2_bin-bin-bin/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split2_bin-bin-bin_model_predictions.csv",
    "model_3" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split3_bin-bin-bin/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split3_bin-bin-bin_model_predictions.csv",
    "model_4" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split4_bin-bin-bin/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split4_bin-bin-bin_model_predictions.csv",
    "model_5" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split5_bin-bin-bin/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split5_bin-bin-bin_model_predictions.csv",
}

YOLO_RAW_HISTEQ_BI_MODEL_PREDS = {
    "model_0" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split0_raw-hist-bi/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split0_raw-hist-bi_model_predictions.csv",
    "model_1" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split1_raw-hist-bi/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split1_raw-hist-bi_model_predictions.csv",
    "model_2" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split2_raw-hist-bi/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split2_raw-hist-bi_model_predictions.csv",
    "model_3" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split3_raw-hist-bi/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split3_raw-hist-bi_model_predictions.csv",
    "model_4" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split4_raw-hist-bi/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split4_raw-hist-bi_model_predictions.csv",
    "model_5" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split5_raw-hist-bi/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split5_raw-hist-bi_model_predictions.csv"
}

AVALANCHE_CONFIGS ={
    "standard" : {
        "model_conf" : 0.50,
        "base_val" : 0.50,
        "rate" : 0.00
    },
    "posterior" : {
        "model_conf" : 0.00,
        "base_val" : 0.50,
        "rate" : 0.00
    },
    "conservative" : {
        "model_conf" : 0.00,
        "base_val" : 0.45,
        "rate" : 0.00
    },
    "gamma15" : {
        "model_conf" : 0.00,
        "base_val" : 0.40,
        "rate" : 0.15
    },
    "gamma20" : {
        "model_conf" : 0.00,
        "base_val" : 0.60,
        "rate" : 0.20
    }
}


def calc_metrics(first_reads: pd.DataFrame,
                 second_reads: pd.DataFrame,
                 iou_threshold: Union[float, Iterable[float]] = None,
                 model: bool = False,
                 model_conf: Optional[float] = None) -> None:
    # Pull out unique PatientID.png from ID column of both reads
    read1_names = np.unique([name.split('/')[-1].upper() for name in first_reads.ID])
    read2_names = np.unique([name.split('/')[-1].upper() for name in second_reads.ID])

    # Find matching PatientIDs
    match_annos = np.intersect1d(read1_names, read2_names)
    # print(f'{len(match_annos)} MATCHING IDs -- TEST SET SIZE {len(read1_names)}')
    # if len(match_annos) < len(read1_names):
    #     for gt in read1_names:
    #         if gt not in match_annos:
    #             print(gt)

    # Create an empty DataFrame to add calculations per image
    calc_df = pd.DataFrame(columns=(['Patient', 'BBoxes Read 1', 'BBoxes Read 2', 'True Positives', 'False Positives', 'False Negatives', 'True Negatives']))

    all_overlaps = []
    all_ious = []
    frac_abs_imgs = 0
    for _, patient in tqdm(enumerate(match_annos), desc='Calculating Metrics', total=len(match_annos), disable=True):
        # Get first- and second-read bounding boxes for patient
        read1_bboxes = get_bounding_boxes(patient, anno_df=first_reads)
        frac_abs_imgs += 1 if len(read1_bboxes) == 0 else 0
        if not model:
            read2_bboxes = get_bounding_boxes(patient, anno_df=second_reads)
        else:
            read2_bboxes, _ = get_bounding_boxes(patient, anno_df=second_reads, has_probs=True, conf_threshold=model_conf)

        # Calculate performance between bounding boxes
        true_pos, false_pos, false_neg, true_neg, ious, overlaps = calc_conf_matrix(read2_bboxes, read1_bboxes, iou_threshold=iou_threshold)

        # Add percent overlaps to all_overlaps and ious to all_ious
        all_overlaps.extend(overlaps)
        all_ious.extend(ious)

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

    # mAP = calc_mAP(preds=second_reads, annots=first_reads, iou_threshold=iou_threshold)
    mAP = 0
    # curr_auc = compute_afroc(first_reads, second_reads, iou_threshold, no_save=True)

    return calc_df, metric_calc, mAP


# def single_model_metrics(
#         ground_truth_csv: pd.DataFrame,
#         model_det_dict: Dict[str, str],
#         avalanche_scheme: str,
#         add_nms: bool = False,
#         calc_auc: bool = False
#     ) -> None:
#     first_reads = pd.read_csv(ground_truth_csv, names=('ID', 'x1', 'y1', 'x2', 'y2', 'class'))

#     all_precisions = np.array([])
#     all_recalls = np.array([])
#     all_f2 = np.array([])
#     all_auc = np.array([])
#     for model in model_det_dict.values():
#         second_reads = pd.read_csv(model, names=('ID', 'x1', 'y1', 'x2', 'y2', 'Prob'))
#         auc = compute_afroc(first_reads, second_reads, iou_threshold=0.30, no_save=True) if avalanche_scheme == 'standard' and calc_auc else 0.0
#         second_reads = get_avalanche_df(second_reads, 'avalanche' if 'gamma' in avalanche_scheme else avalanche_scheme, AVALANCHE_CONFIGS[avalanche_scheme]["base_val"], AVALANCHE_CONFIGS[avalanche_scheme]["rate"], add_nms=add_nms)
#         # Convert necessary columns to float for comparisons to work
#         for col in ['x1', 'y1', 'x2', 'y2', 'Prob']:
#             second_reads[col] = pd.to_numeric(second_reads[col], errors='coerce')

#         # calc_df, metric_calc, mAP = calc_metrics(first_reads, second_reads, iou_threshold=0.30, model=True, model_conf=0.50)
#         _, metric_calc, _ = calc_metrics(first_reads, second_reads, iou_threshold=0.30, model=True, model_conf=AVALANCHE_CONFIGS[avalanche_scheme]["model_conf"])

#         all_precisions = np.append(all_precisions, metric_calc.precision())
#         all_recalls = np.append(all_recalls, metric_calc.recall())
#         all_f2 = np.append(all_f2, metric_calc.f2_score())
#         all_auc = np.append(all_auc, auc)

#     print(f"{avalanche_scheme.title()} Avalanche Scheme\nConfig: {AVALANCHE_CONFIGS[avalanche_scheme]}")

#     for i, (precision, recall, f2, auc) in enumerate(zip(all_precisions, all_recalls, all_f2, all_auc)):
#         print(f"Model {i} | Precision: {precision:.3f} -- Recall: {recall:.3f} -- F2: {f2:.3f} -- AUC: {auc:.3f}")

#     print(f"Avg. Precision: {all_precisions.mean():.3f} +/- {all_precisions.std():.3f}")
#     print(f"Avg. Recall: {all_recalls.mean():.3f} +/- {all_recalls.std():.3f}")
#     print(f"Avg. F2 Score: {all_f2.mean():.3f} +/- {all_f2.std():.3f}")
#     print(f"Avg. AUC: {all_auc.mean():.3f} +/- {all_auc.std():.3f}")


def single_model_metrics(
        ground_truth_csv: pd.DataFrame,
        model_det_dict: Dict[str, str],
        avalanche_scheme: str,
        add_nms: bool = False,
        calc_auc: bool = False,
        calc_max_f2: bool = False
    ) -> None:
    first_reads = pd.read_csv(ground_truth_csv, names=('ID', 'x1', 'y1', 'x2', 'y2', 'class'))

    all_precisions = np.array([])
    all_recalls = np.array([])
    all_f2 = np.array([])
    all_auc = np.array([])
    all_max_f2 = np.array([])
    all_max_f2_thresh = np.array([])
    for model in tqdm(model_det_dict.values(), desc="Iterating through splits"):
        second_reads = pd.read_csv(model, names=('ID', 'x1', 'y1', 'x2', 'y2', 'Prob'))
        auc = compute_afroc(first_reads, second_reads, iou_threshold=0.30, no_save=True) if avalanche_scheme == 'standard' and calc_auc else 0.0

        max_f2_score = 0
        max_f2_thresh = 0
        for thresh in (pbar2 := tqdm(np.linspace(0.05, 1.0, 20))):
            thresh = round(thresh, 2)
            pbar2.set_description(f"Calculating {thresh=}")
            if not calc_max_f2 and thresh != AVALANCHE_CONFIGS[avalanche_scheme]["base_val"]:
                continue
            if avalanche_scheme != "standard":
                second_reads = get_avalanche_df(second_reads, 'avalanche' if 'gamma' in avalanche_scheme else avalanche_scheme, thresh, AVALANCHE_CONFIGS[avalanche_scheme]["rate"], add_nms=add_nms)

            # Convert necessary columns to float for comparisons to work
            for col in ['x1', 'y1', 'x2', 'y2', 'Prob']:
                second_reads[col] = pd.to_numeric(second_reads[col], errors='coerce')

            conf = thresh if avalanche_scheme == "standard" else 0.0
            # calc_df, metric_calc, mAP = calc_metrics(first_reads, second_reads, iou_threshold=0.30, model=True, model_conf=0.50)
            _, metric_calc, _ = calc_metrics(first_reads, second_reads, iou_threshold=0.30, model=True, model_conf=conf)
            if metric_calc.f2_score() > max_f2_score:
                max_f2_score = metric_calc.f2_score()
                max_f2_thresh = thresh

            if conf == AVALANCHE_CONFIGS[avalanche_scheme]["model_conf"]:
                all_precisions = np.append(all_precisions, metric_calc.precision())
                all_recalls = np.append(all_recalls, metric_calc.recall())
                all_f2 = np.append(all_f2, metric_calc.f2_score())
                all_auc = np.append(all_auc, auc)
        all_max_f2 = np.append(all_max_f2, max_f2_score)
        all_max_f2_thresh = np.append(all_max_f2_thresh, max_f2_thresh)

    print(f"{avalanche_scheme.title()} Avalanche Scheme\nConfig: {AVALANCHE_CONFIGS[avalanche_scheme]}")

    for i, (precision, recall, f2, auc, max_f2, max_f2_thresh) in enumerate(zip(all_precisions, all_recalls, all_f2, all_auc, all_max_f2, all_max_f2_thresh)):
        out_str = f"Model {i} | Precision: {precision:.3f} -- Recall: {recall:.3f} -- F2: {f2:.3f}"
        out_str += f" -- AUC: {auc:.3f}" if calc_auc else ""
        out_str += f" | Max F2: {max_f2:.3f} (thresh={max_f2_thresh:.2f})" if calc_max_f2 else ""
        print(out_str)

    print(f"Avg. Precision: {all_precisions.mean():.3f} +/- {all_precisions.std():.3f}")
    print(f"Avg. Recall: {all_recalls.mean():.3f} +/- {all_recalls.std():.3f}")
    print(f"Avg. F2 Score: {all_f2.mean():.3f} +/- {all_f2.std():.3f}")
    if calc_auc:
        print(f"Avg. AUC: {all_auc.mean():.3f} +/- {all_auc.std():.3f}")
    if calc_max_f2:
        print(f"Avg. Max F2: {all_max_f2.mean():.3f} +/- {all_max_f2.std():.3f}")


def multiple_model_metrics(
        ground_truth_csv: pd.DataFrame,
        images_path: str,
        num: int,
        model_det_dict: Dict[str, str],
        avalanche_scheme: str,
        add_nms: bool = False,
        calc_auc: bool = False,
        calc_max_f2: bool = False
    ) -> None:
    first_reads = pd.read_csv(ground_truth_csv, names=('ID', 'x1', 'y1', 'x2', 'y2', 'class'))

    all_combos = combinations(model_det_dict.keys(), num)

    all_precisions = np.array([])
    all_recalls = np.array([])
    all_f2 = np.array([])
    all_auc = np.array([])
    all_max_f2 = np.array([])
    all_max_f2_thresh = np.array([])
    all_combo_list = []
    for combo in all_combos:
        print(f"CURRENT MODELS: {combo}")
        combo_list = [model_det_dict[model] for model in combo]
        all_combo_list.append(combo)
        second_reads = make_ensemble(combo_list, images_path, df_names=['ID', 'x1', 'y1', 'x2', 'y2', 'Prob'], nms_iou_thresh=0.45)
        auc = compute_afroc(first_reads, second_reads, iou_threshold=0.30, no_save=True) if avalanche_scheme == 'standard' and calc_auc else 0.0

        max_f2_score = 0
        max_f2_thresh = 0
        pbar = tqdm(np.linspace(0.05, 1.0, 20))
        for thresh in pbar:
            thresh = round(thresh, 2)
            pbar.set_description(f"Calculating {thresh=}")
            if not calc_max_f2 and thresh != AVALANCHE_CONFIGS[avalanche_scheme]["base_val"]:
                continue
            if avalanche_scheme != "standard":
                second_reads = get_avalanche_df(second_reads, 'avalanche' if 'gamma' in avalanche_scheme else avalanche_scheme, thresh, AVALANCHE_CONFIGS[avalanche_scheme]["rate"], add_nms=add_nms)

            # Convert necessary columns to float for comparisons to work
            for col in ['x1', 'y1', 'x2', 'y2', 'Prob']:
                second_reads[col] = pd.to_numeric(second_reads[col], errors='coerce')

            conf = thresh if avalanche_scheme == "standard" else 0.0
            # calc_df, metric_calc, mAP = calc_metrics(first_reads, second_reads, iou_threshold=0.30, model=True, model_conf=0.50)
            _, metric_calc, _ = calc_metrics(first_reads, second_reads, iou_threshold=0.30, model=True, model_conf=conf)
            if metric_calc.f2_score() > max_f2_score:
                max_f2_score = metric_calc.f2_score()
                max_f2_thresh = thresh
            if conf == AVALANCHE_CONFIGS[avalanche_scheme]["model_conf"]:
                all_precisions = np.append(all_precisions, metric_calc.precision())
                all_recalls = np.append(all_recalls, metric_calc.recall())
                all_f2 = np.append(all_f2, metric_calc.f2_score())
                all_auc = np.append(all_auc, auc)
        all_max_f2 = np.append(all_max_f2, max_f2_score)
        all_max_f2_thresh = np.append(all_max_f2_thresh, max_f2_thresh)

    print(f"\n{avalanche_scheme.title()} Avalanche Scheme\nConfig: {AVALANCHE_CONFIGS[avalanche_scheme]}")
    for i, (combo, precision, recall, f2, auc, max_f2, max_f2_thresh) in enumerate(zip(all_combo_list, all_precisions, all_recalls, all_f2, all_auc, all_max_f2, all_max_f2_thresh)):
        out_str = f"Combo {i:>2} {combo} | Precision: {precision:.3f} -- Recall: {recall:.3f} -- F2: {f2:.3f}"
        out_str += f" -- AUC: {auc:.3f}" if calc_auc else ""
        out_str += f" | Max F2: {max_f2:.3f} (thresh={max_f2_thresh:.2f})" if calc_max_f2 else ""
        print(out_str)

    print(f"Avg. Precision: {all_precisions.mean():.3f} +/- {all_precisions.std():.3f}")
    print(f"Avg. Recall: {all_recalls.mean():.3f} +/- {all_recalls.std():.3f}")
    print(f"Avg. F2 Score: {all_f2.mean():.3f} +/- {all_f2.std():.3f}")
    if calc_auc:
        print(f"Avg. AUC: {all_auc.mean():.3f} +/- {all_auc.std():.3f}")
    if calc_max_f2:
        print(f"Avg. Max F2: {all_max_f2.mean():.3f} +/- {all_max_f2.std():.3f}")


def hybrid_model_metrics(
        ground_truth_csv: pd.DataFrame,
        images_path: str,
        avalanche_scheme: str,
        all_combos=None,
        model_list=None,
        add_nms: bool = False,
        calc_auc: bool = False,
        calc_max_f2: bool = False
    ) -> None:
    first_reads = pd.read_csv(ground_truth_csv, names=('ID', 'x1', 'y1', 'x2', 'y2', 'class'))

    all_precisions = np.array([])
    all_recalls = np.array([])
    all_f2 = np.array([])
    all_auc = np.array([])
    all_max_f2 = np.array([])
    all_max_f2_thresh = np.array([])
    all_combo_list = []
    for i, combo in enumerate(all_combos):
        print(f"CURRENT MODELS: {combo}")
        combo_list = model_list[i]
        all_combo_list.append(combo)
        second_reads = make_ensemble(combo_list, images_path, df_names=['ID', 'x1', 'y1', 'x2', 'y2', 'Prob'], nms_iou_thresh=0.45)
        auc = compute_afroc(first_reads, second_reads, iou_threshold=0.30, no_save=True) if avalanche_scheme == 'standard' and calc_auc else 0.0

        max_f2_score = 0
        max_f2_thresh = 0
        pbar = tqdm(np.linspace(0.05, 1.0, 20))
        for thresh in pbar:
            thresh = round(thresh, 2)
            pbar.set_description(f"Calculating {thresh=}")
            if not calc_max_f2 and thresh != AVALANCHE_CONFIGS[avalanche_scheme]["base_val"]:
                continue
            if avalanche_scheme != "standard":
                second_reads = get_avalanche_df(second_reads, 'avalanche' if 'gamma' in avalanche_scheme else avalanche_scheme, thresh, AVALANCHE_CONFIGS[avalanche_scheme]["rate"], add_nms=add_nms)

            # Convert necessary columns to float for comparisons to work
            for col in ['x1', 'y1', 'x2', 'y2', 'Prob']:
                second_reads[col] = pd.to_numeric(second_reads[col], errors='coerce')

            conf = thresh if avalanche_scheme == "standard" else 0.0
            # calc_df, metric_calc, mAP = calc_metrics(first_reads, second_reads, iou_threshold=0.30, model=True, model_conf=0.50)
            _, metric_calc, _ = calc_metrics(first_reads, second_reads, iou_threshold=0.30, model=True, model_conf=conf)
            if metric_calc.f2_score() > max_f2_score:
                max_f2_score = metric_calc.f2_score()
                max_f2_thresh = thresh
            if conf == AVALANCHE_CONFIGS[avalanche_scheme]["model_conf"]:
                all_precisions = np.append(all_precisions, metric_calc.precision())
                all_recalls = np.append(all_recalls, metric_calc.recall())
                all_f2 = np.append(all_f2, metric_calc.f2_score())
                all_auc = np.append(all_auc, auc)
        all_max_f2 = np.append(all_max_f2, max_f2_score)
        all_max_f2_thresh = np.append(all_max_f2_thresh, max_f2_thresh)

    print(f"\n{avalanche_scheme.title()} Avalanche Scheme\nConfig: {AVALANCHE_CONFIGS[avalanche_scheme]}")
    for i, (combo, precision, recall, f2, auc, max_f2, max_f2_thresh) in enumerate(zip(all_combo_list, all_precisions, all_recalls, all_f2, all_auc, all_max_f2, all_max_f2_thresh)):
        out_str = f"Combo {i:>2} {combo} | Precision: {precision:.3f} -- Recall: {recall:.3f} -- F2: {f2:.3f}"
        out_str += f" -- AUC: {auc:.3f}" if calc_auc else ""
        out_str += f" | Max F2: {max_f2:.3f} (thresh={max_f2_thresh:.2f})" if calc_max_f2 else ""
        print(out_str)
    print(f"Avg. Precision: {all_precisions.mean():.3f} +/- {all_precisions.std():.3f}")
    print(f"Avg. Recall: {all_recalls.mean():.3f} +/- {all_recalls.std():.3f}")
    print(f"Avg. F2 Score: {all_f2.mean():.3f} +/- {all_f2.std():.3f}")
    if calc_auc:
        print(f"Avg. AUC: {all_auc.mean():.3f} +/- {all_auc.std():.3f}")
    if calc_max_f2:
        print(f"Avg. Max F2: {all_max_f2.mean():.3f} +/- {all_max_f2.std():.3f}")


def randomly_choose_ensemble(num_iter: int, architecture: str):
    model_list = []
    combo_list = []
    while len(model_list) < num_iter:
        if architecture == 'retinanet':
            normal_key = random.choice(list(RETINANET_MODEL_PREDS.keys()))
            model_0 = RETINANET_MODEL_PREDS[normal_key]

            bin_bin_bin_key = random.choice(list(RETINANET_BIN_BIN_BIN_MODEL_PREDS.keys()))
            model_1 = RETINANET_BIN_BIN_BIN_MODEL_PREDS[bin_bin_bin_key]

            raw_hist_bi_key = random.choice(list(RETINANET_RAW_HISTEQ_BI_MODEL_PREDS.keys()))
            model_2 = RETINANET_RAW_HISTEQ_BI_MODEL_PREDS[raw_hist_bi_key]

            if (normal_key, bin_bin_bin_key, raw_hist_bi_key) in model_list:
                continue
            model_list.append((normal_key, bin_bin_bin_key, raw_hist_bi_key))
            combo_list.append((model_0, model_1, model_2))
        elif architecture == 'yolo':
            normal_key = random.choice(list(YOLO_MODEL_PREDS.keys()))
            model_0 = YOLO_MODEL_PREDS[normal_key]

            bin_bin_bin_key = random.choice(list(YOLO_BIN_BIN_BIN_MODEL_PREDS.keys()))
            model_1 = YOLO_BIN_BIN_BIN_MODEL_PREDS[bin_bin_bin_key]

            raw_hist_bi_key = random.choice(list(YOLO_RAW_HISTEQ_BI_MODEL_PREDS.keys()))
            model_2 = YOLO_RAW_HISTEQ_BI_MODEL_PREDS[raw_hist_bi_key]
            
            if (normal_key, bin_bin_bin_key, raw_hist_bi_key) in model_list:
                continue
            model_list.append((normal_key, bin_bin_bin_key, raw_hist_bi_key))
            combo_list.append((model_0, model_1, model_2))
        elif architecture == 'both':
            ret_normal_key = random.choice(list(RETINANET_MODEL_PREDS.keys()))
            ret_model_0 = RETINANET_MODEL_PREDS[ret_normal_key]

            ret_bin_bin_bin_key = random.choice(list(RETINANET_BIN_BIN_BIN_MODEL_PREDS.keys()))
            ret_model_1 = RETINANET_BIN_BIN_BIN_MODEL_PREDS[ret_bin_bin_bin_key]

            ret_raw_hist_bi_key = random.choice(list(RETINANET_RAW_HISTEQ_BI_MODEL_PREDS.keys()))
            ret_model_2 = RETINANET_RAW_HISTEQ_BI_MODEL_PREDS[ret_raw_hist_bi_key]

            yolo_normal_key = random.choice(list(YOLO_MODEL_PREDS.keys()))
            yolo_model_0 = YOLO_MODEL_PREDS[yolo_normal_key]

            yolo_bin_bin_bin_key = random.choice(list(YOLO_BIN_BIN_BIN_MODEL_PREDS.keys()))
            yolo_model_1 = YOLO_BIN_BIN_BIN_MODEL_PREDS[yolo_bin_bin_bin_key]

            yolo_raw_hist_bi_key = random.choice(list(YOLO_RAW_HISTEQ_BI_MODEL_PREDS.keys()))
            yolo_model_2 = YOLO_RAW_HISTEQ_BI_MODEL_PREDS[yolo_raw_hist_bi_key]
            
            if (ret_normal_key, ret_bin_bin_bin_key, ret_raw_hist_bi_key, yolo_normal_key, yolo_bin_bin_bin_key, yolo_raw_hist_bi_key) in model_list:
                continue
            model_list.append((ret_normal_key, ret_bin_bin_bin_key, ret_raw_hist_bi_key, yolo_normal_key, yolo_bin_bin_bin_key, yolo_raw_hist_bi_key))
            combo_list.append((ret_model_0, ret_model_1, ret_model_2, yolo_model_0, yolo_model_1, yolo_model_2))


    return model_list, combo_list


def main(parse_args):
    """Main Function"""
    # Set the random module seed
    random.seed(parse_args.seed)

    if parse_args.hybrid_input:
        combos, csvs = randomly_choose_ensemble(parse_args.hybrid_combo_count, parse_args.hybrid_input)
        hybrid_model_metrics(parse_args.ground_truth_csv, parse_args.img_path, all_combos=combos, model_list=csvs, avalanche_scheme=parse_args.avalanche, add_nms=parse_args.avalanche_nms, calc_auc=parse_args.auc, calc_max_f2=parse_args.max_f2)
        return


    if parse_args.model == "retinanet":
        if parse_args.preprocessing == "normal":
            model_dict = RETINANET_MODEL_PREDS
        elif parse_args.preprocessing == "bin-bin-bin":
            model_dict = RETINANET_BIN_BIN_BIN_MODEL_PREDS
        elif parse_args.preprocessing == "raw-hist-bi":
            model_dict = RETINANET_RAW_HISTEQ_BI_MODEL_PREDS
    if parse_args.model == "yolo":
        if parse_args.preprocessing == "normal":
            model_dict = YOLO_MODEL_PREDS
        elif parse_args.preprocessing == "bin-bin-bin":
            model_dict = YOLO_BIN_BIN_BIN_MODEL_PREDS
        elif parse_args.preprocessing == "raw-hist-bi":
            model_dict = YOLO_RAW_HISTEQ_BI_MODEL_PREDS


    if parse_args.num_models > 1:
        multiple_model_metrics(parse_args.ground_truth_csv, parse_args.img_path, parse_args.num_models, model_dict, avalanche_scheme=parse_args.avalanche, add_nms=parse_args.avalanche_nms, calc_auc=parse_args.auc, calc_max_f2=parse_args.max_f2)
    else:
        single_model_metrics(parse_args.ground_truth_csv, model_dict, avalanche_scheme=parse_args.avalanche, add_nms=parse_args.avalanche_nms, calc_auc=parse_args.auc, calc_max_f2=parse_args.max_f2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=ARGS['RANDOM_SEED'],
                        help='Set seed for randomizations.')

    parser.add_argument('--ground_truth_csv', type=str, required=True,
                        help='Path to CSV containing ground truth annotations for the test set.')

    parser.add_argument('--model', type=str, required=True,
                        help='Choice whether to calculate metrics for RetinaNet or YOLO predictions')

    parser.add_argument('--preprocessing', choices=['normal', 'bin-bin-bin', 'raw-hist-bi'], required=True,
                        help='Choice of what type of preprocessing for single-model-type metrics.')

    parser.add_argument('--num_models', type=int, required=True, choices=[1, 2, 3, 6],
                        help='If 1, averages all cross-fold models. If 2/3/6, goes through all \
                             combinations of models and ensembles predictions before calculating metrics.')

    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to pre-processed images to add in during ensembling.')

    parser.add_argument('--avalanche', choices=['standard', 'posterior', 'conservative', 'gamma15', 'gamma20'],
                        help='Which type of avalanche scheme to use.')

    parser.add_argument('--hybrid_input', choices=['retinanet', 'yolo', 'both'],
                        help='Combines the three different input processing types into ensembles based on architecture choice.')

    parser.add_argument('--hybrid_combo_count', type=int,
                        help='Number of hybrid input combos to create.')

    parser.add_argument('--avalanche_nms', action='store_true',
                        help='Applies non-max suppression to avalanche methods if True.')

    parser.add_argument('--auc', action='store_true',
                        help='If true, calculate AUC.')

    parser.add_argument('--max_f2', action='store_true',
                        help='If true, calculate the max F2 score across all thresholds.')

    args = parser.parse_args()

    print('\nStarting execution...')
    start_time = time.perf_counter()
    main(args)
    elapsed = time.perf_counter() - start_time
    print('Done!')
    print_elapsed(elapsed)
