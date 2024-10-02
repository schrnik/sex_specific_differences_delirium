"This File assumes that MIMIC-IV is installed on a local PostgreSQL server, and buildmimic files and concepts files from the official MIMIC code repository (https://github.com/MIT-LCP/mimic-iv) are installed.

author: @nikolausschreiber"

  
#Demographics table
\COPY (SELECT 
    p.subject_id, 
    p.gender, 
    p.anchor_age, 
    p.anchor_year, 
    p.anchor_year_group,
    a.admission_type, 
    a.admittime, 
    a.dischtime, 
    a.deathtime, 
    a.insurance
FROM mimiciv_hosp.patients p
JOIN mimiciv_hosp.admissions a ON p.subject_id = a.subject_id) 
TO '/path_to_your_file/demographics_with_death.csv' WITH CSV HEADER;


SET search_path TO mimiciv_derived

#APSIII table
\copy apsiii TO 'apsiii.csv' WITH CSV HEADER

#SAPSII
\copy sapsii TO 'sapsii_copy.csv' WITH CSV HEADER

#Sepsis
\copy sepsis3 TO 'sepsis_copy.csv' WITH CSV HEADER

#Ventilation
\copy ventilation TO 'ventilation_copy.csv' WITH CSV HEADER

#Charlson Comorbidity Index
\copy charlson TO 'charlson_copy.csv' WITH CSV HEADER

