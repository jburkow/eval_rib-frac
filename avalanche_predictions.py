'''
Filename: avalanche_predictions.py
Author: Jonathan Burkow, burkowjo@msu.edu
        Michigan State University
Last Updated: 02/28/2022
Description: Takes a provided model predictions CSV file and applies an avalanche decision scheme
    to it. A new CSV file is created with the predictions that are kept after the scheme is applied.
'''

import argparse
import os
import time

import numpy as np
import pandas as pd


def apply_scheme(patient_df: pd.DataFrame, scheme: str, base_value: float, rate: float) -> pd.DataFrame:
    """
    Applies one of the avalanche schemes to the patient_df and outputs a DataFrame of all
    predictions accepted for the current patient.

    Parameters
    ----------
    patient_df : DataFrame of model predictions on a single patient
    scheme     : the chosen avalanche scheme to apply to patient_df
    base_value : the starting value, alpha_0, to apply across all threshold values
    rate       : if scheme='avalanche', uses rate for constant rate decrease between thresholds
    """
    # Define avalanche percent drops for different schemes
    if scheme == "posterior":
        # avalanche_percentages = [1, 1-0.7643979057591623, 1-0.8424657534246576, 1-0.7723577235772358, 1-0.7842105263157895]  # 20210420 data
        avalanche_percentages = [1, 1-0.7364864864864865, 1-0.8134556574923547, 1-0.7669172932330827, 1-0.7696078431372549]  # 20210902 data
    elif scheme == "conservative":
        # avalanche_percentages = [1, 0.7643979057591623, 0.8424657534246576, 0.7723577235772358, 0.7842105263157895]  # 20210420 data
        avalanche_percentages = [1, 0.7364864864864865, 0.8134556574923547, 0.7669172932330827, 0.7696078431372549]  # 20210902 data
    elif scheme == "avalanche":
        avalanche_percentages = [(1 - rate) for _ in range(10)]
        avalanche_percentages.insert(0, 1)
    else:
        avalanche_percentages = []

    # Create the test thresholds list to use when determining what model predictions are kept
    if scheme == "standard":
        thresholds = None
    else:
        thresholds = [base_value * val for val in [np.prod(np.array(avalanche_percentages[:k])) for k in range(1, len(avalanche_percentages)+1)]]

    start_df = patient_df[patient_df['Prob'] >= thresholds[0]]
    num_model_boxes = len(start_df)

    prior_num_model_boxes = 0
    thresh = None
    while num_model_boxes > prior_num_model_boxes:
        prior_num_model_boxes = num_model_boxes
        # Set threshold level based on number of already accepted fractures
        test_thresh_ind = min(len(thresholds)-1, num_model_boxes)
        thresh = thresholds[test_thresh_ind]
        step_df = patient_df[patient_df['Prob'] >= thresh]
        num_model_boxes = len(step_df)

        if test_thresh_ind == len(thresholds):
            break

    return start_df if thresh is None else step_df


def get_avalanche_df(data_df: pd.DataFrame, scheme: str, base_value: float, rate: float) -> pd.DataFrame:
    """
    Apply an avalanche scheme to the model predictions DataFrame and output the final DataFrame of
    model predictions.

    Parameters
    ----------
    data_df    : DataFrame of model predictions on entire test set
    scheme     : the chosen avalanche scheme to apply to patient_df
    base_value : the starting value, alpha_0, to apply across all threshold values
    rate       : if scheme='avalanche', uses rate for constant rate decrease between thresholds
    """
    out_df = pd.DataFrame(columns=data_df.columns)

    grouped_df = data_df.groupby('ID')

    for group_name, df_group in grouped_df:
        if scheme == "standard":
            scheme_df = df_group[df_group['Prob'] >= base_value]
        else:
            scheme_df = apply_scheme(df_group, scheme, base_value, rate)

        if len(scheme_df) == 0:
            out_df = out_df.append({'ID'   : group_name,
                                    'x1'   : "",
                                    'y1'   : "",
                                    'x2'   : "",
                                    'y2'   : "",
                                    'Prob' : ""}, ignore_index=True)
        else:
            out_df = out_df.append(scheme_df, ignore_index=True)

    return out_df


def main(pargs):
    """Main Function"""
    data_df = pd.read_csv(pargs.csv_preds, names=('ID', 'x1', 'y1', 'x2', 'y2', 'Prob'),
                          dtype={'x1': pd.Int64Dtype(), 'y1': pd.Int64Dtype(), 'x2': pd.Int64Dtype(), 'y2': pd.Int64Dtype()})

    out_df = get_avalanche_df(data_df, pargs.method, pargs.base_val, pargs.rate)

    if not pargs.no_save:
        save_name = f'{pargs.filename}_{pargs.method}_base{pargs.base_val}'
        save_name += f"_rate{pargs.rate}.csv" if pargs.method == "avalanche" else '.csv'
        out_df.to_csv(os.path.join(pargs.save_dir, save_name), header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv_preds', type=str, required=True,
                        help='Filename to CSV containing model predictions.')

    parser.add_argument('--method', choices=['standard', 'posterior', 'conservative', 'avalanche'],
                        help='Which type of avalanche scheme to use.')

    parser.add_argument('--base_val', type=float, default=0.50,
                        help='Starting model confidence to begin the avalanche scheme.')

    parser.add_argument('--rate', type=float, default=0.05,
                        help='The percentage that each successive avalanche scheme step drops by.')

    parser.add_argument('--save_dir', default='./out_files',
                        help='Default folder to save files to.')

    parser.add_argument('--filename',
                        help='Name to save new CSV files as.')

    parser.add_argument('--no_save', action='store_true',
                        help='If true, does not save resulting DataFrame to a CSV.')

    parser_args = parser.parse_args()

    # Print out start of execution
    print('Starting execution...')
    start_time = time.perf_counter()

    # Run main function
    main(parser_args)

    # Print out time to complete
    print('Done!')
    end_time = time.perf_counter()
    print(f'Execution finished in {end_time - start_time:.3f} seconds.')
