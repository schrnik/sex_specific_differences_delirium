"""
This file takes the d_items.csv file and the chartevent.csv file from https://physionet.org/files/mimiciv/3.0/ 
and saves off any rows with delirium testing information from into a csv (all_delirium_chart_events_mimic_iv_copy.csv).
This file is needed in the next step. 

@author: Nikolaus Schreiber
"""

# Import necessary libraries
import pandas as pd
import numpy as np

# Load D_ITEMS file to get the ITEMID and LABEL columns for MIMIC-IV
items = pd.read_csv("~/d_items.csv", usecols=['itemid', 'label'])

# Make LABEL column lowercase
items['label'] = items['label'].apply(lambda s: s.lower() if isinstance(s, str) else s)

# Find relevant terms in LABELs
cam_ids = items[items['label'].str.contains('cam-icu', na=False)]
delirium_ids = items[items['label'].str.contains('delirium', na=False)]
# Combine CAM-ICU and delirium IDs
relevant_ids = pd.concat([cam_ids, delirium_ids])

# Initialize a DataFrame to store the delirium-related chart events
delirium_testing_rows = pd.read_csv("~/Desktop/mimic-iv-3.0/Manuscript/Code/chartevents.csv", nrows=0)

# Load chartevents.csv in chunks to handle large file size
for chunk in pd.read_csv("~/Desktop/mimic-iv-3.0/icu/chartevents.csv", 
                         chunksize=1000000, 
                         dtype={'itemid': 'int', 'subject_id': 'int', 'charttime': 'str', 'value': 'str'}, 
                         low_memory=False):
    # Filter rows where ITEMID is in the relevant IDs list
    temp_rows = chunk[chunk['itemid'].isin(relevant_ids['itemid'])]
    # Concatenate the filtered rows to the main DataFrame
    delirium_testing_rows = pd.concat([delirium_testing_rows, temp_rows], ignore_index=True)

# Sort values by subject_id and charttime for easier analysis
delirium_testing_rows.sort_values(['subject_id', 'charttime'], inplace=True)

# Merge with relevant_ids to get LABEL from D_ITEMS
delirium_testing_rows = delirium_testing_rows.merge(relevant_ids, how='left', on='itemid')

# Extract relevant columns and sort by subject_id and charttime
del_values = delirium_testing_rows[['subject_id', 'charttime', 'label', 'value']]
del_values.sort_values(['subject_id', 'charttime'], inplace=True)

# Get unique subjects/time stamps for delirium assessments
assessments = del_values[del_values['label'] == 'delirium assessment']
assessment_times = assessments[['subject_id', 'charttime']].drop_duplicates()

# Get unique subjects/time stamps for parts of CAM-ICU
parts = del_values[del_values['label'] != 'delirium assessment']
parts_times = parts[['subject_id', 'charttime']].drop_duplicates()

# Find where assessment times and part times do not overlap
non_overlap = parts_times.merge(assessment_times, how='outer', on=['subject_id', 'charttime'], indicator=True)
non_overlap = non_overlap[non_overlap['_merge'] != 'both']

# Save the results to CSV files
delirium_testing_rows.to_csv('all_delirium_chart_events_mimic_iv_copy.csv', index=False)

# Extract unique subject_id, hadm_id, stay_id combinations for delirium patients (MIMIC-IV uses stay_id)
delirium_IDs = delirium_testing_rows[['subject_id', 'hadm_id', 'stay_id']].drop_duplicates()
delirium_IDs.to_csv('MIMICIV_delirium_chart_event_IDs_copy.csv', index=False)
