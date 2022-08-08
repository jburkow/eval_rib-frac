'''
Filename: parallel_multi_model_metrics.py
Author(s): Jonathan Burkow, burkowjo@msu.edu, Michigan State University
Last Updated: 05/09/2022
Description: Combines multiple model predictions and automatically calculates the avg +/- std.dev
    of precision, recall, and F2 scores from them all. Uses multiprocessing to speed up calcualtions.
'''

import argparse
import concurrent.futures
import random
import time
from itertools import combinations
from typing import Iterable, Optional, Union

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
    "model_5" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-02-07_6fold-splits/20220207_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split5/20220207_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split5_predictions.csv",
    "model_6" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-05-09_10fold_splits/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split6/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split6_predictions.csv",
    "model_7" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-05-09_10fold_splits/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split7/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split7_predictions.csv",
    "model_8" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-05-09_10fold_splits/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split8/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split8_predictions.csv",
    "model_9" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-05-09_10fold_splits/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split9/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split9_predictions.csv"
}

RETINANET_BIN_BIN_BIN_MODEL_PREDS = {
    "model_0" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_bin-bin-bin/20220304_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split0/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split0_bin-bin-bin_predictions.csv",
    "model_1" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_bin-bin-bin/20220304_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split1/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split1_bin-bin-bin_predictions.csv",
    "model_2" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_bin-bin-bin/20220304_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split2/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split2_bin-bin-bin_predictions.csv",
    "model_3" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_bin-bin-bin/20220304_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split3/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split3_bin-bin-bin_predictions.csv",
    "model_4" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_bin-bin-bin/20220304_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split4/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split4_bin-bin-bin_predictions.csv",
    "model_5" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_bin-bin-bin/20220304_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split5/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split5_bin-bin-bin_predictions.csv",
    "model_6" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-05-09_10fold_splits_bin-bin-bin/20220509_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split6/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split6_bin-bin-bin_predictions.csv",
    "model_7" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-05-09_10fold_splits_bin-bin-bin/20220509_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split7/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split7_bin-bin-bin_predictions.csv",
    "model_8" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-05-09_10fold_splits_bin-bin-bin/20220509_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split8/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split8_bin-bin-bin_predictions.csv",
    "model_9" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-05-09_10fold_splits_bin-bin-bin/20220509_bin-bin-bin_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split9/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split9_bin-bin-bin_predictions.csv"
}

RETINANET_RAW_HISTEQ_BI_MODEL_PREDS = {
    "model_0" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_raw-hist-bi/20220304_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split0/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split0_raw-hist-bi_predictions.csv",
    "model_1" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_raw-hist-bi/20220304_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split1/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split1_raw-hist-bi_predictions.csv",
    "model_2" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_raw-hist-bi/20220304_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split2/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split2_raw-hist-bi_predictions.csv",
    "model_3" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_raw-hist-bi/20220304_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split3/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split3_raw-hist-bi_predictions.csv",
    "model_4" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_raw-hist-bi/20220304_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split4/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split4_raw-hist-bi_predictions.csv",
    "model_5" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-03-04_6fold-splits_raw-hist-bi/20220304_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split5/20220304_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split5_raw-hist-bi_predictions.csv",
    "model_6" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-05-09_10fold_splits_raw-hist-bi/20220509_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split6/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split6_raw-hist-bi_predictions.csv",
    "model_7" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-05-09_10fold_splits_raw-hist-bi/20220509_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split7/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split7_raw-hist-bi_predictions.csv",
    "model_8" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-05-09_10fold_splits_raw-hist-bi/20220509_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split8/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split8_raw-hist-bi_predictions.csv",
    "model_9" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/model_training_and_eval/pytorch-retinanet/2022-05-09_10fold_splits_raw-hist-bi/20220509_raw-hist-bi_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split9/20220509_resnet50_pretrained_300epoch_no-norm_aug_lr0.0001_bs8_patience5-30_seed0_split9_raw-hist-bi_predictions.csv"
}

YOLO_MODEL_PREDS = {
    "model_0" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split0/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split0_model_predictions.csv",
    "model_1" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split1/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split1_model_predictions.csv",
    "model_2" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split2/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split2_model_predictions.csv",
    "model_3" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split3/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split3_model_predictions.csv",
    "model_4" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split4/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split4_model_predictions.csv",
    "model_5" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split5/2022-03-29_yolov5l6_pretrained_300epoch_bs8_split5_model_predictions.csv",
    "model_6" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split6/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split6_model_predictions.csv",
    "model_7" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split7/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split7_model_predictions.csv",
    "model_8" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split8/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split8_model_predictions.csv",
    "model_9" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split9/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split9_model_predictions.csv"
}

YOLO_BIN_BIN_BIN_MODEL_PREDS = {
    "model_0" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split0_bin-bin-bin/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split0_bin-bin-bin_model_predictions.csv",
    "model_1" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split1_bin-bin-bin/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split1_bin-bin-bin_model_predictions.csv",
    "model_2" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split2_bin-bin-bin/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split2_bin-bin-bin_model_predictions.csv",
    "model_3" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split3_bin-bin-bin/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split3_bin-bin-bin_model_predictions.csv",
    "model_4" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split4_bin-bin-bin/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split4_bin-bin-bin_model_predictions.csv",
    "model_5" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split5_bin-bin-bin/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split5_bin-bin-bin_model_predictions.csv",
    "model_6" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split6_bin-bin-bin/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split6_bin-bin-bin_model_predictions.csv",
    "model_7" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split7_bin-bin-bin/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split7_bin-bin-bin_model_predictions.csv",
    "model_8" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split8_bin-bin-bin/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split8_bin-bin-bin_model_predictions.csv",
    "model_9" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split9_bin-bin-bin/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split9_bin-bin-bin_model_predictions.csv"
}

YOLO_RAW_HISTEQ_BI_MODEL_PREDS = {
    "model_0" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split0_raw-hist-bi/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split0_raw-hist-bi_model_predictions.csv",
    "model_1" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split1_raw-hist-bi/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split1_raw-hist-bi_model_predictions.csv",
    "model_2" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split2_raw-hist-bi/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split2_raw-hist-bi_model_predictions.csv",
    "model_3" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split3_raw-hist-bi/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split3_raw-hist-bi_model_predictions.csv",
    "model_4" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split4_raw-hist-bi/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split4_raw-hist-bi_model_predictions.csv",
    "model_5" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split5_raw-hist-bi/2022-03-13_yolov5l6_pretrained_300epoch_bs8_split5_raw-hist-bi_model_predictions.csv",
    "model_6" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split6_raw-hist-bi/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split6_raw-hist-bi_model_predictions.csv",
    "model_7" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split7_raw-hist-bi/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split7_raw-hist-bi_model_predictions.csv",
    "model_8" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split8_raw-hist-bi/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split8_raw-hist-bi_model_predictions.csv",
    "model_9" : "/mnt/home/burkowjo/midi_lab/burkowjo_data/architectures/yolov5-ultralytics/yolov5/runs/detect/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split9_raw-hist-bi/2022-05-09_yolov5l6_pretrained_300epoch_bs8_split9_raw-hist-bi_model_predictions.csv"
}

AVALANCHE_CONFIGS = {
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


def calc_metrics(
        first_reads: pd.DataFrame,
        second_reads: pd.DataFrame,
        iou_threshold: Union[float, Iterable[float]] = None,
        model: bool = False,
        model_conf: Optional[float] = None
    ) -> None:
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


def get_values(thresh, avalanche_scheme, rate, ground_truth_csv, second_read_csv, images_path):
    first_reads = pd.read_csv(ground_truth_csv, names=('ID', 'x1', 'y1', 'x2', 'y2', 'class'))
    if isinstance(second_read_csv, str):
        second_reads = pd.read_csv(second_read_csv, names=('ID', 'x1', 'y1', 'x2', 'y2', 'Prob'))
    else:
        second_reads = make_ensemble(second_read_csv, images_path, df_names=['ID', 'x1', 'y1', 'x2', 'y2', 'Prob'], nms_iou_thresh=0.45)
    
    if avalanche_scheme != "standard":
        second_reads = get_avalanche_df(second_reads, 'avalanche' if 'gamma' in avalanche_scheme else avalanche_scheme, thresh, rate, add_nms=True)

    # Convert necessary columns to float for comparisons to work
    for col in ['x1', 'y1', 'x2', 'y2', 'Prob']:
        second_reads[col] = pd.to_numeric(second_reads[col], errors='coerce')

    conf = thresh if avalanche_scheme == "standard" else 0.0
    _, metric_calc, _ = calc_metrics(first_reads, second_reads, iou_threshold=0.30, model=True, model_conf=conf)

    return metric_calc


def get_array_of_metrics(model_det_dict, num, avalanche_scheme, avalanche_config, ground_truth_csv, images_path, hybrid=None, max_combos=20):
    all_model_arrays = []

    if hybrid is None:
        *all_combos, = combinations(model_det_dict.keys(), num) if num > 1 else model_det_dict.values()
    else:
        all_combos = hybrid[0]

    num_combos = sum(1 for _ in all_combos)
    # print(f"number of combos: {sum(1 for _ in all_combos)}")
    # # print(all_combos)
    # for combo in all_combos:
    #     print(combo)

    if num_combos > max_combos:
        all_combos = random.sample(all_combos, max_combos)

        # print(f"new number of combos: {sum(1 for _ in all_combos_new)}")
        # for combo in all_combos_new:
        #     print(combo)

    for i, combo in enumerate(all_combos):
        if hybrid is None:
            combo_list = combo if isinstance(combo, str) else [model_det_dict[model] for model in combo]
        else:
            combo_list = hybrid[1][i]
        # threshes = np.round(np.linspace(0.05, 1.0, 20), 2)
        threshes = np.round(np.linspace(0.01, 1.0, 100), 2)
        all_precisions = []
        all_recalls = []
        all_f2 = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            first_reads_csvs = [ground_truth_csv for _ in range(threshes.size)]
            second_reads_csvs = [combo_list for _ in range(threshes.size)]
            rates = [avalanche_config[avalanche_scheme]["rate"] for _ in range(threshes.size)]
            schemes = [avalanche_scheme for _ in range(threshes.size)]
            img_paths = [images_path for _ in range(threshes.size)]
            
            results = executor.map(get_values, threshes, schemes, rates, first_reads_csvs, second_reads_csvs, img_paths)

            for result in results:
                all_precisions.append(result.precision())
                all_recalls.append(result.recall())
                all_f2.append(result.f2_score())

        all_model_arrays.append(np.column_stack([threshes, all_precisions, all_recalls, all_f2]))

    return np.stack(all_model_arrays)


def print_avgs(array_stack, avalanche_scheme, configs, model_combos=None) -> None:
    idx = np.argwhere(array_stack[0,:,0] == configs[avalanche_scheme]["base_val"])[0][0]
    print(f"{avalanche_scheme.capitalize()} Avalanche Scheme")
    print(f"Config: {configs[avalanche_scheme]}")
    for i, metrics_array in enumerate(array_stack):
        out_str = 'Combo' if len(model_combos[0]) > 1 else 'Model'
        out_str += f" {i:>2}" if len(array_stack) > 9 else f" {i}"
        out_str += f" {model_combos[i]} " if len(model_combos[0]) > 1 else ' '
        out_str += f"| Precision: {metrics_array[idx,1]:.3f} -- Recall: {metrics_array[idx,2]:.3f} -- F2: {metrics_array[idx,3]:.3f}"
        out_str += f" | Max F2: {max(metrics_array[:,3]):.3f} (thresh={metrics_array[np.argmax(metrics_array[:,3]), 0]:.2f})"
        print(out_str)
    print(f"Avg. Precision:  {np.mean(array_stack[:,idx,1]):.3f} +/- {np.std(array_stack[:,idx,1]):.3f}")
    print(f"Avg. Recall:     {np.mean(array_stack[:,idx,2]):.3f} +/- {np.std(array_stack[:,idx,2]):.3f}")
    print(f"Avg. F2 Score:   {np.mean(array_stack[:,idx,3]):.3f} +/- {np.std(array_stack[:,idx,3]):.3f}")
    print(f"Avg. Max F2:     {np.mean([max(metrics_array[:,3]) for metrics_array in array_stack]):.3f} +/- {np.std([max(metrics_array[:,3]) for metrics_array in array_stack]):.3f}")


def main(parse_args) -> None:
    """Main Function"""
    # Set the random module seed
    random.seed(parse_args.seed)

    if parse_args.hybrid_input:
        # combos, csvs = randomly_choose_ensemble(parse_args.hybrid_combo_count, parse_args.hybrid_input)
        combos, csvs = randomly_choose_ensemble(parse_args.max_combos, parse_args.hybrid_input)

        array_stack = get_array_of_metrics(model_det_dict=None, num=None, avalanche_scheme=parse_args.avalanche, avalanche_config=AVALANCHE_CONFIGS, ground_truth_csv=parse_args.ground_truth_csv, images_path=parse_args.img_path, hybrid=(combos, csvs), max_combos=parse_args.max_combos)
        print_avgs(array_stack, parse_args.avalanche, AVALANCHE_CONFIGS, combos)
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

    array_stack = get_array_of_metrics(model_det_dict=model_dict, num=parse_args.num_models, avalanche_scheme=parse_args.avalanche, avalanche_config=AVALANCHE_CONFIGS, ground_truth_csv=parse_args.ground_truth_csv, images_path=parse_args.img_path, hybrid=None, max_combos=parse_args.max_combos)
    print_avgs(array_stack, parse_args.avalanche, AVALANCHE_CONFIGS, list(combinations(model_dict.keys(), parse_args.num_models)))


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

    parser.add_argument('--max_combos', type=int,
                        help='Maximum number of combinations to calculate.')

    parser.add_argument('--avalanche_nms', action='store_true',
                        help='Applies non-max suppression to avalanche methods if True.')

    parser.add_argument('--auc', action='store_true',
                        help='If true, calculate AUC.')

    args = parser.parse_args()

    print('\nStarting execution...')
    start_time = time.perf_counter()
    main(args)
    elapsed = time.perf_counter() - start_time
    print('Done!')
    print(print_elapsed(elapsed))
