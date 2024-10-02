"""
This file takes all_delirium_chart_events_mimic_iv_copy.csv from the first step and saves off 'MIMICIV_chart_events_delirium_labels_copy.csv', which is needed in the next step.
Takes some time depending on computational capacities!


@author: nikolausschreiber
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from tqdm import tqdm

# File paths
input_file = '~/all_delirium_chart_events_mimic_iv_copy.csv'
output_file = 'MIMICIV_chart_events_delirium_labels_copy.csv'

# Define the chunk size (adjust based on your system's memory)
chunk_size = 100000  # Number of rows per chunk

# Initialize an empty list to store results
results_list = []

# Define the function for processing delirium testing
def get_delirium_testing(subject_id, chart_time, ids_and_results):
    # Get the testing done at this subject and time
    temp_rows = ids_and_results[(ids_and_results['subject_id'] == subject_id) & 
                                (ids_and_results['charttime'] == chart_time)]
    
    # Check if delirium assessment is present
    del_assess = temp_rows[temp_rows['label'] == 'delirium assessment']
    if not del_assess.empty:
        value = del_assess['value'].iloc[0]
        if value == 'Negative':
            return 0
        elif value == 'Positive':
            return 1
        elif value == 'UTA':
            return np.nan

    # If no delirium assessment, check CAM-ICU components
    ms_change = temp_rows[temp_rows['label'] == 'cam-icu ms change']['value'].iloc[0] if not temp_rows[temp_rows['label'] == 'cam-icu ms change'].empty else 'Unable to Assess'
    inattention = temp_rows[temp_rows['label'] == 'cam-icu inattention']['value'].iloc[0] if not temp_rows[temp_rows['label'] == 'cam-icu inattention'].empty else 'Unable to Assess'
    rass_loc = temp_rows[temp_rows['label'] == 'cam-icu rass loc']['value'].iloc[0] if not temp_rows[temp_rows['label'] == 'cam-icu rass loc'].empty else 'Unable to Assess'
    altered_loc = temp_rows[temp_rows['label'] == 'cam-icu altered loc']['value'].iloc[0] if not temp_rows[temp_rows['label'] == 'cam-icu altered loc'].empty else 'Unable to Assess'
    disorganized = temp_rows[temp_rows['label'] == 'cam-icu disorganized thinking']['value'].iloc[0] if not temp_rows[temp_rows['label'] == 'cam-icu disorganized thinking'].empty else 'Unable to Assess'
    
    # Determine delirium status based on these parts
    if 'No' in ms_change:
        return 0
    elif 'Unable to Assess' in ms_change:
        return np.nan
    elif 'No' in inattention:
        return 0
    elif 'Unable to Assess' in inattention:
        return np.nan
    elif rass_loc == 'Yes' or altered_loc == 'Yes':
        return 1
    elif disorganized == 'No':
        return 0
    elif disorganized == 'Unable to Assess':
        return np.nan
    elif disorganized == 'Yes':
        return 1

# Function to process each chunk
def process_chunk(chunk):
    # Keep only relevant columns
    ids_and_results = chunk[['subject_id', 'hadm_id', 'stay_id', 'charttime', 'label', 'value']]
    
    # Get unique subjects/times for processing
    subjects_and_times = ids_and_results[['subject_id', 'hadm_id', 'stay_id', 'charttime']].drop_duplicates()
    subjects_and_times.sort_values(['subject_id', 'charttime'], inplace=True)
    
    # Apply the function to each subject and time, with progress tracking
    subjects_and_times['delirium_positive'] = subjects_and_times.progress_apply(
        lambda row: get_delirium_testing(row['subject_id'], row['charttime'], ids_and_results), axis=1)
    
    return subjects_and_times

# Initialize progress bar
tqdm.pandas()

# Read and process the input file in chunks
for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    # Process each chunk and append the results
    chunk_results = process_chunk(chunk)
    results_list.append(chunk_results)

# Concatenate the results from all chunks
final_results = pd.concat(results_list)

# Save the final results to a CSV file
final_results.to_csv(output_file, index=False)
