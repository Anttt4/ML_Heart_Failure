# ML_Heart_Failure

This machine learning project aims to predict whether a person would be prone to heart disease, taking into account a series of health characteristics.

The dataset used is public and comes from Kaggle (user FEDESORIANO - https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). The dataset has 918 rows and 12 columns. The columns are as follows:

- ***Age***: Age of the patient (years)
- ***Sex***: Sex of the patient (M: Male, F: Female)
- ***ChestPainType***: Chest pain type (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)
- ***RestingBP***: Resting blood pressure (mm Hg)
- ***Cholesterol***: Serum cholesterol (mm/dl)
- ***FastingBS***: Fasting blood sugar (1: if FastingBS > 120 mg/dl, 0: otherwise)
- ***RestingECG***: Resting electrocardiogram results (Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria)
- ***MaxHR***: Maximum heart rate achieved (Numeric value between 60 and 202)
- ***ExerciseAngina***:  Exercise-induced angina (Y: Yes, N: No)
- ***Oldpeak***: Oldpeak = ST (Numeric value measured in depression)
- ***ST_Slope***: The slope of the peak exercise ST segment (Up: upsloping, Flat: flat, Down: downsloping)

The project is broken down into an initial data visualization and then, through a series of pipelines, preprocessing, selection of relevant features, creation, and optimization of machine learning models. Finally, the optimized model is evaluated against the test dataset.

The repository is divided into a 'main' folder with this README and a 'src' folder, which in turn has the following subfolders:

- data_sample: dataset used
- img: images that may have been used
- models: best model obtained
- notebooks: notebooks used
- results_notebook: final notebook
- utils: functions that may have been used

