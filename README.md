Ugochukwu Fred Odimba
Project title:  Predicting COPD Exacerbations


1.	Introduction

COPD is a chronic respiratory disease affecting middle-aged and older adults. It is characterized by progressive airflow limitation and respiratory symptoms. 
COPD exacerbations are acute events in the course of the disease that lead to worsened respiratory symptoms, increased healthcare utilization, reduced quality of life, and mortality. 
Identifying individuals at higher risk of COPD exacerbations is crucial for implementing timely interventions to improve their outcomes and reduce the healthcare burden. 
This project aims to develop a predictive model for COPD exacerbations.

2.	Selection of dataset

The dataset selected for this ISP is sourced from a public open-access repository. This dataset can be found at this link: https://zenodo.org/records/6400146. 
It includes anonymized patient data such as demographic information,  body composition measurements,  lifestyle and clinical characteristics, medication use, 
lung function tests,  various comorbidities,  and exacerbation status of COPD patients. The dataset was chosen for its comprehensive nature and relevance to 
the COPD exacerbations study, enabling thorough analysis and model development.

3.	Plan for analysis

The analysis will involve pre-processing the data to handle missing values, outliers, and normalization or standardization. 
The data will be split into training and test set percentages of 75/25, respectively. Supervised Machine learning algorithms such as 
logistic regression, decision trees, and random forests will be utilized to build predictive models. 
The performance of these models will be evaluated using appropriate classification metrics such as accuracy, recall, precision, F1 score, and area under the ROC curve (AUC-ROC). 
However, the recall and the area under the ROC  will be the most important metrics to be used for selecting the best predictive model. 
The analysis will be done in Python programming language and libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn will be utilized.

Variables in the dataset
* sex				- Women & Men
* age_cat2		- Age in three categories 
bodycomp		- body composition was assessed with bioelectrical impedance measurements using a Bodystat 1500 to calculate fat mass index (FMI) and fat free mass index (FFMI). Cachexia was defined as having a FFMI < 14 kg/m2 in women and < 17 kg/m2 in men which corresponds to the lower 95% CI in a normal population (19). Similarly, obesity was defined as FMI > 13.5 kg/m2 in women and > 9.3 kg/m2 in men.
smok_habits		- defined as never, ex, or current (daily)
packyrs_10		- Pack years smoked (defined for ex and daily only) where 20 cigarettes/day for one year equals one pack year divided by ten for regression purposes 
diabetes		- self-reported, and assessed with aid of medical journal by a physician l ( Yes vs No)
statin			- self-reported, and assessed with aid of medical journal by a physician l (Yes vs No)
ARB_ACE_all		- use of either AII blockers or angiotensin receptor blockers
sign_CACS		- having a calcium score > 100 = "yes", else is "no"
cor_stenosis	-  (Yes vs No) having significant stenosis on CT angiography assessed by one of two experienced cardiology radiologists. Confirmed coronary stenosis was defined as presence of stenosis (lumen reduction > 50%)
COPD_control	- either COPD patient or control without lung disease
gold			- In COPD patients either GOLD stage I/II or stage III/IV
copd_exacerb_cat	- having had a moderate or serious COPD exacerbation the last 12 months or not. Serious = hospitalization, moderate = treatment with systemic antiobiotics and/or steroids (Yes vs No)	
resp_failure	-	Having arterial blood gass oxygen tension (pO2) < 8 kPa = yes
eosinophilic_COPD	- Having blood eosinophilia defined as ≥0.3*10^9 cells/L = yes
wholelung950	- % lung area of density < 950 Hounsfield units
crp_cat			- C-reactive protein below 5 or 5 and above (No vs Yes)
