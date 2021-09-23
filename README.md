# Azure ML Heart Failure Outcome Prediction Project

This project was intended to explore and prove conceptual and practical knowledge of the Azure ML Studio environment and specifically to train and deploy the best machine learning model possible into a full production environment. The approaches used were the Hyperdrive Parameter Tuning and AutoML features of Azure ML Studio. This project was written to be plug-and-play for anyone who wishes to run this in their own environment. To do so, simply replace the config file with your own from Azure ML Studio and run the cells in the order they are presented.

## Project Set Up and Installation
To run this project:
  - Create compute resouce
  - Create compute cluster resource that match the names provided in the notebooks, or alter the name in the notebook to a cluster you already have created.
  - Replace the config file with a config file from your own Azure ML Studio subscription.
  - Upload the files to a notebook folder in the Notebooks section oof Azure ML Studio.
  - Run the notebooks from top to bottom.
  - Preparing stage while the model is training may take between 5-10 minutes depending on compute resources because the .yml environment used for training is the same that is used for model deployment to cut down on folder size.

## Dataset

### Overview
The data that was obtained from kaggle and the raw csv file was pulled from the UCI Machine Learning Repository at https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records (Davide Chicco, Giuseppe Jurman: "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone". BMC Medical Informatics and Decision Making 20, 16 (2020)). The dataset contains heart failure clinical records from patients that experienced heart failure after receiving 1 of 2 treatments. 

### Task
The DEATH_EVENT column is the outcome event that we are trying to predict and indicates if the patient had heart failure in the time between follow up appointments. The features we are using to predict are health related data points such as if the patient has high blood pressure, age, the presence of diabetes, the amount of each serum in the blood, smoking status, as well as a few others. All of the data has already been 1 hot encoded and there was not need for transformation. The only cleaning of the data that was done was the replacing of age values of 60.667 with 60 and the removal of a platelets value that seemed erroneous in the data. This resulted in a total of 274 rows of data for the model.

### Access
The data was pulled into the workspace directly from the csv link provided here https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv using a Tabular Dataset object from_csv_files() function.
```
ds = Dataset.Tabular.from_delimited_files(path="https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")
```

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

For the AutoML experiment I used the config to set the task to classification and also to split the dataset into it's validation set and testing set during the experiment. Because I used a more robust compute resource than my Hyperdrive run my quota would only permit 2 nodes and so there are only 2 concurrent iterations at a time for the AutoML run.

```
automl_config = AutoMLConfig(
    experiment_timeout_minutes=60,
    task='classification',
    primary_metric='accuracy',
    training_data=data,
    label_column_name='DEATH_EVENT',
    validation_size=.1,
    test_size=.1,
    compute_target=compute_target,
    max_concurrent_iterations=2)
```

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
The best model that was produced from the AutoML run was the MaxAbsScaler, XGBoostClassifier. The accuracy of this model was only 80%. This type of model trains many decision trees using each of the previous trees to train the next and reduce the prediction error. The first thing I would do to improve this model would be to write out a .py file for hyperparameter tuning to change the learning rate and max depth parameters at the very least to see if a better result could be acheived.
*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.
<img width="1475" alt="capstone_automl_rundetails" src="https://user-images.githubusercontent.com/28558135/134451414-dcd10448-1826-4a22-b5fc-2578966bfbc4.png">

<img width="2018" alt="capstone_automl_bestmodel_runID" src="https://user-images.githubusercontent.com/28558135/134451432-06ad0eef-8cbb-4432-b93b-85531a6e38b0.png">




## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
