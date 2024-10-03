"""
This script produces the main analyses and plots of the paper. 
It uses MIMICIV_complete_dataset.csv as well as the dataframes extracted using SQL queries provided by the official MIMIC GitHub repository (see vairables.sql).

@author: nikolausschreiber
"""



# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt



# Load the datasets
delir = pd.read_csv('~/MIMICIV_complete_dataset.csv') #make sure that the path to the file is correctly specified
demographics = pd.read_csv('~/demographics_with_death_copy.csv') #make sure that the path to the file is correctly specified

# Filter adelirium
delir = delir[delir['delirium_positive'] == 1]  # Only delirium-positive patients
delir = delir.drop_duplicates(subset='subject_id', keep='first')  # Remove duplicates
delir['del_onset_time'] = pd.to_datetime(delir['del_onset_time'])  # Convert to datetime

# Merge with demographics
delir_complete_df = pd.merge(delir, demographics, on='subject_id', how='inner')
delir_complete_df = delir_complete_df.drop_duplicates(subset='subject_id', keep='first')

# Convert datetime columns
delir_complete_df['deathtime'] = pd.to_datetime(delir_complete_df['deathtime'], errors='coerce')
delir_complete_df['admittime'] = pd.to_datetime(delir_complete_df['admittime'], errors='coerce')
delir_complete_df['dischtime'] = pd.to_datetime(delir_complete_df['dischtime'])

# Create 30-day mortality outcome variable 
delir_complete_df['mortality_within_30_days'] = np.where(
    (delir_complete_df['deathtime'].notnull()) & ((delir_complete_df['deathtime'] - delir_complete_df['del_onset_time']).dt.days <= 30),
    1,
    0
)

# Load Charlson Comorbidity, APSIII and SAPS II datasets, and merge with the main dataset
charlson = pd.read_csv('~/charlson.csv') #make sure that the path to the file is correctly specified
sapsii = pd.read_csv('~/sapsii.csv') #make sure that the path to the file is correctly specified
apsiii = pd.read_csv('~/apsiii.csv') #make sure that the path to the file is correctly specified

# Merge Charlson, SAPS II, and APSIII data
delir_complete_df = pd.merge(delir_complete_df, charlson, on='subject_id', how='inner')
delir_complete_df = pd.merge(delir_complete_df, sapsii, on='subject_id', how='inner')
apsiii.drop(['hadm_id', 'stay_id'], axis=1, inplace=True)  # Drop unnecessary columns
delir_complete_df = pd.merge(delir_complete_df, apsiii, on='subject_id', how='inner')
delir_complete_df = delir_complete_df.drop_duplicates(subset='subject_id', keep='first')

#Create a follow-up time column (capped at 30 days)
delir_complete_df['followup_time'] = np.where(
    delir_complete_df['deathtime'].notnull(),
    (delir_complete_df['deathtime'] - delir_complete_df['del_onset_time']).dt.total_seconds() / (60 * 60 * 24),
    (delir_complete_df['dischtime'] - delir_complete_df['del_onset_time']).dt.total_seconds() / (60 * 60 * 24)
)
delir_complete_df['followup_time'] = delir_complete_df['followup_time'].clip(upper=30)
delir_complete_df = delir_complete_df[(delir_complete_df['followup_time'] > 0) & (delir_complete_df['followup_time'] < 30)]

#recode gender for modeling 
delir_complete_df['gender'] = delir_complete_df['gender'].apply(lambda x: 0 if x == 'M' else 1)  # Encode gender to float: Male=0, Female=1


#import and merge mechanical ventilation dataset 

mv = pd.read_csv('~/ventilation.csv') #make sure that the path to the file is correctly specified
mv_merged = pd.merge(sapsii, mv, on='stay_id', how='inner')
mv_merged = mv_merged.drop_duplicates(subset='subject_id', keep='first')
mv_ids = mv_merged[['subject_id', 'starttime_x', 'endtime_x', 'ventilation_status']]
mv_ids = mv_ids.drop_duplicates(subset='subject_id', keep='first')

delir_complete_df = pd.merge(delir_complete_df, mv_ids, on='subject_id', how='inner')

# create variable for minvasive ventilation befor delirium onset
delir_complete_df['starttime_x'] = pd.to_datetime(delir_complete_df['starttime_x'], errors='coerce')

delir_complete_df['mechanical_ventilation_before'] = delir_complete_df.apply(
    lambda row: 1 if (row['ventilation_status'] == 'InvasiveVent') and 
                        (row['starttime_x'] >= row['admittime']) and 
                        (row['starttime_x'] <= row['del_onset_time']) 
                 else 0, 
    axis=1
)

#load sepsis data and merge with main dataframe 
sepsis = pd.read_csv('~/sepsis.csv') #make sure that the path to the file is correctly specified

delir_complete_df = pd.merge(delir_complete_df, sepsis[['subject_id', 'sepsis3']], on='subject_id', how='left')

# Create the covariate column sepsis_at_admission

delir_complete_df['sepsis_at_admission'] = delir_complete_df['sepsis3'].apply(lambda x: 1 if x == 't' else 0)

# If there are any NaNs in sepsis3 (because some subject_ids are not in the sepsis dataframe), set them to 0
delir_complete_df['sepsis_at_admission'].fillna(0, inplace=True)
delir_complete_df = delir_complete_df.drop_duplicates(subset='subject_id', keep='first')

# Create covariate column 'admission_type'
delir_complete_df['admission_type'] = delir_complete_df['admissiontype_score'].apply(lambda x: 'surgical' if x == 8 else 'medical')

delir_complete_df['admission_type'] = delir_complete_df['admission_type'].apply(lambda x: 0 if x == 'medical' else 1)  # Encode admission type to float 


#rename variables for better readability 
# Renaming columns
delir_complete_df = delir_complete_df.rename(columns={'myocardial_infarct': 'coronary_artery_disease', 'anchor_age': 'age'})


import pandas as pd
from tableone import TableOne 



# Filter baseline variables 
baseline_vars = ['subject_id', 'gender', 'age', 'charlson_comorbidity_index', 'mechanical_ventilation_before', 'sepsis_at_admission', 'admission_type', 'peripheral_vascular_disease', 'cerebrovascular_disease', 
                 'mortality_within_30_days', 'apsiii', 'sapsii', 
                 'coronary_artery_disease', 'congestive_heart_failure', 
                 'renal_disease', 'malignant_cancer', 'dementia',
                 'chronic_pulmonary_disease', 'rheumatic_disease',
                 'peptic_ulcer_disease', 'mild_liver_disease', 'diabetes_without_cc',
                 'diabetes_with_cc', 'paraplegia',
                 'severe_liver_disease', 'metastatic_solid_tumor', 'aids']


df_baseline = delir_complete_df[baseline_vars]

# make sure no duplicates are in the dataframe 
df_baseline = df_baseline.drop_duplicates(subset='subject_id')

# Recode gender to string 
df_baseline['gender'] = df_baseline['gender'].apply(lambda x: 'Male' if x == 0 else 'Female')

# Drop rows with NaN values (if any)
df_baseline = df_baseline.dropna()

# Define categorical and continuous variables
categorical_vars = ['gender', 'mortality_within_30_days', 'coronary_artery_disease', 'mechanical_ventilation_before', 'sepsis_at_admission', 'admission_type', 'peripheral_vascular_disease', 'cerebrovascular_disease', 
                    'congestive_heart_failure', 'diabetes_without_cc', 
                    'renal_disease', 'malignant_cancer', 'dementia',
                    'chronic_pulmonary_disease', 'rheumatic_disease',
                    'peptic_ulcer_disease', 'mild_liver_disease',
                    'diabetes_with_cc', 'paraplegia',
                    'severe_liver_disease', 'metastatic_solid_tumor', 'aids']

continuous_vars = ['age', 'charlson_comorbidity_index', 'apsiii', 'sapsii']

#  Create Table One, with 'Female' vs 'Male' grouping for gender
table1 = TableOne(df_baseline, 
                  columns=baseline_vars[1:],  # Exclude 'subject_id'
                  categorical=categorical_vars, 
                  groupby='gender',  # Compare 'Female' vs 'Male'
                  nonnormal=continuous_vars,  # Non-normal distribution for continuous variables
                  label_suffix=True,
                  pval=True,  # Display p-values
                  pval_test_name=True)

# Display Table One (you can tabulate it nicely with different formats)
print(table1.tabulate(tablefmt="fancy_grid"))

# Convert the table to a pandas DataFrame and save it as an HTML file
html_table = table1.to_html()  # Convert to HTML string
with open("table_one.html", "w") as file:
    file.write(html_table)

############univariable analysis - 
#chi square test for 30 day mortality 

import pandas as pd
from scipy.stats import chi2_contingency

# Create a contingency table
contingency_table = pd.crosstab(delir_complete_df['gender'], delir_complete_df['mortality_within_30_days'])

print("Contingency Table:")
print(contingency_table)

# Perform the Chi-squared test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print("\nChi-squared Test Results:")
print(f"Chi-squared statistic: {chi2}")
print(f"Degrees of freedom: {dof}")
print(f"P-value: {p_value}")

# cox proportional hazard regression
# Prepare data for Cox regression
df_cox = delir_complete_df[['subject_id', 'followup_time', 'mortality_within_30_days', 'gender']]
df_cox = df_cox.dropna(subset=['followup_time', 'mortality_within_30_days', 'gender'])
df_cox = df_cox.drop_duplicates(subset='subject_id')


# Initialize and fit Cox proportional hazards model
cph = CoxPHFitter()
cph.fit(
    df_cox.drop(columns=['subject_id']),
    duration_col='followup_time',
    event_col='mortality_within_30_days'
)

# Display the Cox regression summary
cph.print_summary()

# Extract the summary dataframe
summary_df = cph.summary

# Calculate the hazard ratios and confidence intervals
summary_df['HR'] = summary_df['exp(coef)']
summary_df['HR_lower_95%'] = summary_df['exp(coef) lower 95%']
summary_df['HR_upper_95%'] = summary_df['exp(coef) upper 95%']

# Select the relevant columns
hr_results = summary_df[['HR', 'HR_lower_95%', 'HR_upper_95%', 'p']]

# Print the hazard ratios with 95% confidence intervals
print(hr_results)

# Check proportional hazards assumption
cph.check_assumptions(df_cox.drop(columns=['subject_id']), show_plots=True)

#kaplan maier univariable whole cohort
# Initialize the Kaplan-Meier fitter
kmf = KaplanMeierFitter()

# Separate data by gender
male = df_cox[df_cox['gender'] == 0]
female = df_cox[df_cox['gender'] == 1]

# Calculate the number of males, females, and events
num_males = male.shape[0]
num_female_events = female[female['mortality_within_30_days'] == 1].shape[0]
num_males_events = male[male['mortality_within_30_days'] == 1].shape[0]
num_females = female.shape[0]

# Plot both survival curves on the same plot
plt.figure(figsize=(10, 6))
ax = plt.subplot(111)

# Fit for males and plot
kmf.fit(
    male['followup_time'],
    event_observed=male['mortality_within_30_days'],
    label=f'Male (n={num_males}, events={num_males_events})'
)
kmf.plot_survival_function(ax=ax)

# Fit for females and plot
kmf.fit(
    female['followup_time'],
    event_observed=female['mortality_within_30_days'],
    label=f'Female (n={num_females}, events={num_female_events})'
)
kmf.plot_survival_function(ax=ax)

# Customize the plot
plt.title("Kaplan-Meier Survival Curve by Sex")
plt.xlabel("Days")
plt.xlim(0, 30)
plt.ylabel("Survival Probability")
plt.legend(loc='best')

# Save the plot as a high-quality image
plt.savefig('Kaplan_Meier_Survival_Curve_by_Sex.png', dpi=300)

plt.show()

# Perform log-rank test
results = logrank_test(
    male['followup_time'], female['followup_time'],
    event_observed_A=male['mortality_within_30_days'],
    event_observed_B=female['mortality_within_30_days']
)

# Print the result of the log-rank test
print(f"Log-Rank Test p-value: {results.p_value:.4f}")



#########Propensity score matching 


#  Define covariates to match on
match_vars = ['age', 'charlson_comorbidity_index', 'sapsii', 'mechanical_ventilation_before', 'sepsis_at_admission', 'admission_type',
              'coronary_artery_disease', 'congestive_heart_failure',
              'peripheral_vascular_disease', 'cerebrovascular_disease', 'dementia',
              'chronic_pulmonary_disease', 'rheumatic_disease',
              'peptic_ulcer_disease', 'mild_liver_disease', 'diabetes_without_cc',
              'diabetes_with_cc', 'paraplegia', 'renal_disease', 'malignant_cancer',
              'severe_liver_disease', 'metastatic_solid_tumor', 'aids']

# Create DataFrame with covariates for matching and outcome variables
df_match = delir_complete_df[['gender', 'followup_time', 'mortality_within_30_days'] + match_vars]

# Drop missing values
df_match = df_match.dropna()

# Fit logistic regression model to estimate propensity scores
X = df_match[match_vars]
y = df_match['gender']

propensity_model = LogisticRegression()
propensity_model.fit(X, y)

# Get the predicted propensity scores
df_match['propensity_score'] = propensity_model.predict_proba(X)[:, 1]

#  Perform Nearest Neighbor Matching (1:1 matching)
females = df_match[df_match['gender'] == 1]
males = df_match[df_match['gender'] == 0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(males[['propensity_score']])
distances, indices = nn.kneighbors(females[['propensity_score']])

# Get matched males based on the nearest neighbors
matched_males = males.iloc[indices.flatten()]

# Combine matched males and females into one DataFrame
matched_df = pd.concat([matched_males, females])

#  Fit Cox Proportional Hazards Model on Matched Data
cph_matched = CoxPHFitter()
cph_matched.fit(matched_df[['mortality_within_30_days', 'followup_time', 'gender']], 
                duration_col='followup_time', 
                event_col='mortality_within_30_days')

# Display the Cox regression summary
cph_matched.print_summary()

# Step 8: Extract the summary dataframe
summary_df = cph_matched.summary

# Step 9: Calculate the hazard ratios and confidence intervals
summary_df['HR'] = summary_df['exp(coef)']
summary_df['HR_lower_95%'] = summary_df['exp(coef) lower 95%']
summary_df['HR_upper_95%'] = summary_df['exp(coef) upper 95%']

# Select the relevant columns and round to 4 decimal places
hr_results = summary_df[['HR', 'HR_lower_95%', 'HR_upper_95%', 'p']].round(4)

# Print the hazard ratios with 95% confidence intervals
print(hr_results)

import seaborn as sns
import matplotlib.pyplot as plt

#  Plot propensity score density for the unmatched cohort
plt.figure(figsize=(10, 6))
sns.kdeplot(df_match[df_match['gender'] == 1]['propensity_score'], label='Women (Unmatched)', shade=True, color='blue')
sns.kdeplot(df_match[df_match['gender'] == 0]['propensity_score'], label='Men (Unmatched)', shade=True, color='red')
plt.title('Propensity Score Distribution (Unmatched Cohort)')
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.legend(loc='best')
plt.savefig('UnmatchedPlot.png', dpi=300)
plt.show()

# Plot propensity score density for the matched cohort
plt.figure(figsize=(10, 6))
sns.kdeplot(matched_df[matched_df['gender'] == 1]['propensity_score'], label='Women (Matched)', shade=True, color='blue')
sns.kdeplot(matched_df[matched_df['gender'] == 0]['propensity_score'], label='Men (Matched)', shade=True, color='red')
plt.title('Propensity Score Distribution (Matched Cohort)')
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.legend(loc='best')
plt.savefig('MatchedPlot.png', dpi=300)
plt.show()




# Initialize the Kaplan-Meier fitter
kmf = KaplanMeierFitter()

# Separate data by gender
male = matched_df[matched_df['gender'] == 0]
female = matched_df[matched_df['gender'] == 1]

# Calculate the number of males, females, and events
num_males = male.shape[0]
num_female_events = female[female['mortality_within_30_days'] == 1].shape[0]
num_males_events = male[male['mortality_within_30_days'] == 1].shape[0]
num_females = female.shape[0]

# Plot both survival curves on the same plot
plt.figure(figsize=(10, 6))
ax = plt.subplot(111)

# Fit for males and plot
kmf.fit(
    male['followup_time'],
    event_observed=male['mortality_within_30_days'],
    label=f'Male (n={num_males}, events={num_males_events})'
)
kmf.plot_survival_function(ax=ax)

# Fit for females and plot
kmf.fit(
    female['followup_time'],
    event_observed=female['mortality_within_30_days'],
    label=f'Female (n={num_females}, events={num_female_events})'
)
kmf.plot_survival_function(ax=ax)

# Customize the plot
plt.title("Kaplan-Meier Survival Curve by Sex")
plt.xlabel("Days")
plt.xlim(0, 30)
plt.ylabel("Survival Probability")
plt.legend(loc='best')
plt.show()

# Perform log-rank test
results = logrank_test(
    male['followup_time'], female['followup_time'],
    event_observed_A=male['mortality_within_30_days'],
    event_observed_B=female['mortality_within_30_days']
)

# Print the result of the log-rank test
print(f"Log-Rank Test p-value: {results.p_value:.4f}")

# Define a function to calculate SMD (Standardized Mean Difference)
def compute_smd(group1, group2):
    mean_diff = group1.mean() - group2.mean()
    pooled_std = np.sqrt((group1.std() ** 2 + group2.std() ** 2) / 2)
    return np.abs(mean_diff / pooled_std)

# Initialize an empty list to collect comparison data
comparison_data = []

#  Loop through each covariate and calculate the mean, std, and SMD
for var in match_vars:
    male_mean = matched_df[matched_df['gender'] == 0][var].mean()
    female_mean = matched_df[matched_df['gender'] == 1][var].mean()
    male_std = matched_df[matched_df['gender'] == 0][var].std()
    female_std = matched_df[matched_df['gender'] == 1][var].std()
    smd = compute_smd(matched_df[matched_df['gender'] == 0][var], matched_df[matched_df['gender'] == 1][var])
    
    # Collect the results for this variable
    comparison_data.append({
        'Variable': var,
        'Mean Male': round(male_mean, 4),
        'Mean Female': round(female_mean, 4),
        'Std Male': round(male_std, 4),
        'Std Female': round(female_std, 4),
        'SMD': round(smd, 4)
    })

# Create the comparison table DataFrame from the list
comparison_table = pd.DataFrame(comparison_data)


# Optionally print the table for terminal output
print(comparison_table)

import pandas as pd
import numpy as np

# Step 1: Define a function to calculate SMD (Standardized Mean Difference)
def compute_smd(group1, group2):
    mean_diff = group1.mean() - group2.mean()
    pooled_std = np.sqrt((group1.std() ** 2 + group2.std() ** 2) / 2)
    return np.abs(mean_diff / pooled_std)

# Initialize an empty list to collect comparison data for both unmatched and matched data
comparison_data = []

# Step 3: Loop through each covariate and calculate the mean, std, and SMD for unmatched and matched data
for var in match_vars:
    # Unmatched data calculations
    male_mean_unmatched = delir_complete_df[delir_complete_df['gender'] == 0][var].mean()
    female_mean_unmatched = delir_complete_df[delir_complete_df['gender'] == 1][var].mean()
    male_std_unmatched = delir_complete_df[delir_complete_df['gender'] == 0][var].std()
    female_std_unmatched = delir_complete_df[delir_complete_df['gender'] == 1][var].std()
    smd_unmatched = compute_smd(delir_complete_df[delir_complete_df['gender'] == 0][var], 
                                delir_complete_df[delir_complete_df['gender'] == 1][var])
    
    # Matched data calculations
    male_mean_matched = matched_df[matched_df['gender'] == 0][var].mean()
    female_mean_matched = matched_df[matched_df['gender'] == 1][var].mean()
    male_std_matched = matched_df[matched_df['gender'] == 0][var].std()
    female_std_matched = matched_df[matched_df['gender'] == 1][var].std()
    smd_matched = compute_smd(matched_df[matched_df['gender'] == 0][var], 
                              matched_df[matched_df['gender'] == 1][var])

    # Collect the results for this variable
    comparison_data.append({
        'Variable': var,
        'Mean Male (Unmatched)': round(male_mean_unmatched, 4),
        'Mean Female (Unmatched)': round(female_mean_unmatched, 4),
        'Std Male (Unmatched)': round(male_std_unmatched, 4),
        'Std Female (Unmatched)': round(female_std_unmatched, 4),
        'SMD (Unmatched)': round(smd_unmatched, 4),
        'Mean Male (Matched)': round(male_mean_matched, 4),
        'Mean Female (Matched)': round(female_mean_matched, 4),
        'Std Male (Matched)': round(male_std_matched, 4),
        'Std Female (Matched)': round(female_std_matched, 4),
        'SMD (Matched)': round(smd_matched, 4)
    })

# Create the comparison table DataFrame from the list
comparison_table = pd.DataFrame(comparison_data)

#  Display the comparison table
print(comparison_table)
pd.set_option('display.max_columns', None)
print(comparison_table)

# Print only the columns with the SMDs for unmatched and matched data
smd_columns = comparison_table[['Variable', 'SMD (Unmatched)', 'SMD (Matched)']]

# Display the table with only the relevant columns
print(smd_columns)


# Optionally save the table to a CSV file
import matplotlib.pyplot as plt

#  Sort the comparison table by 'SMD (Unmatched)' in descending order
comparison_table_sorted = comparison_table.sort_values(by='SMD (Unmatched)', ascending=True)

#  Create a Love Plot
plt.figure(figsize=(10, 8))

# Plot SMDs for unmatched data (red) in the sorted order
plt.scatter(comparison_table_sorted['SMD (Unmatched)'], comparison_table_sorted['Variable'], color='red', label='Unmatched', s=100)

# Plot SMDs for matched data (blue) in the sorted order
plt.scatter(comparison_table_sorted['SMD (Matched)'], comparison_table_sorted['Variable'], color='blue', label='Matched', s=100)

# Add a vertical reference line at SMD = 0.1 and -0.1 (threshold for good balance)
plt.axvline(x=0.1, color='gray', linestyle='--', label='SMD Threshold (0.1)', linewidth=2)
plt.axvline(x=-0.1, color='gray', linestyle='--', linewidth=2)

# Add a "fat" vertical line at SMD = 0 (bold line for zero reference)
plt.axvline(x=0, color='black', linestyle='-', linewidth=5, label='SMD = 0')

# Customize the plot
plt.xlabel('Standardized Mean Difference (SMD)')
plt.ylabel('Covariates')
plt.title('Love Plot: Standardized Mean Differences Before and After Matching')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()


# Save the plot as a high-quality image
plt.savefig('LovePlot.png', dpi=300)
# Display the plot
plt.show()



