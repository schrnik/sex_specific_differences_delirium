"""
This file creates 'MIMICIV_complete_dataset.csv' from MIMICIV_chart_events_delirium_labels_copy.csv (created in the last script) and the icusstays.csv file from https://physionet.org/files/mimiciv/3.0/.

@author: nikolausschreiber
"""

#%% Package setup and loading in data.
import pandas as pd

# Load delirium labels for MIMIC-IV
del_chart_events = pd.read_csv('~/MIMICIV_chart_events_delirium_labels_copy.csv')
# Load ICU stays data for MIMIC-IV
icustays = pd.read_csv("~/icustays.csv", usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'los'])

#%% Find if patients had delirium, and when they got it.
# Drop any rows with missing data
del_chart_events = del_chart_events.dropna()

# Group by subject_id, hadm_id, and stay_id to find if the patient had delirium during ICU stay
dataset = del_chart_events[['subject_id', 'hadm_id', 'stay_id', 'delirium_positive']].groupby(['subject_id', 'hadm_id', 'stay_id']).max()

# Get delirium onset times
del_onset = del_chart_events[['stay_id', 'charttime', 'delirium_positive']]
del_onset = del_onset[del_onset['delirium_positive'] == 1]
del_onset['charttime'] = pd.to_datetime(del_onset['charttime'])

# Group by stay_id to get the earliest delirium onset time
del_onset = del_onset[['stay_id', 'charttime']]
del_onset = del_onset.groupby('stay_id').min()
del_onset.reset_index(inplace=True)

# Merge the delirium onset times back to the dataset
dataset = dataset.merge(del_onset, on='stay_id', how='left')

#%% Remove ICU stays where the patient wasn't in the ICU for at least 12 hours.
# Merge with ICU stay information
dataset = dataset.merge(icustays, on='stay_id', how='left')

# Keep ICU stays of at least 12 hours (0.5 days)
dataset = dataset[dataset['los'] >= 0.5]

# Calculate delirium onset time from ICU admission
dataset['intime'] = pd.to_datetime(dataset['intime'])
dataset['del_onset'] = dataset['charttime'] - dataset['intime']

# Convert the time difference to minutes
dataset['del_onset'] = dataset.apply(lambda row: row['del_onset'].total_seconds() / 60, axis=1)

#%% Rename columns and save the results.
dataset.rename(columns={'charttime': 'del_onset_time'}, inplace=True)
dataset.sort_values('stay_id', inplace=True)

# Save the final dataset to a CSV file
dataset.to_csv('MIMICIV_complete_dataset.csv', index=False)

#%% Find class balance
# Check class balance of delirium positive and negative cases
class_bal = dataset[['stay_id', 'delirium_positive']]
class_bal = class_bal.groupby('delirium_positive').count()

# Output: class balance
print(class_bal)
