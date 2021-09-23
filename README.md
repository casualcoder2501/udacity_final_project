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
For the AutoML experiment I used the config to set the task to classification and also to split the dataset into it's validation set and testing set during the experiment. Because I used a more robust compute resource than my Hyperdrive run my quota would only permit 2 nodes and so there are only 2 concurrent iterations at a time for the AutoML run.

```
automl_settings ={
    "task":'classification',
    "primary_metric":'accuracy',
    "training_data":data,
    "validation_size":.1,
    "test_size":.1,
    "max_concurrent_iterations":2,
    "label_column_name":'DEATH_EVENT',
    "experiment_timeout_hours":1.0
}


automl_config = AutoMLConfig(
    compute_target=compute_target,
    **automl_settings
    )
```

### Results
The best model that was produced from the AutoML run was the MaxAbsScaler, XGBoostClassifier. The accuracy of this model was only 80%. This type of model trains many decision trees using each of the previous trees to train the next and reduce the prediction error. The first thing I would do to improve this model would be to write out a .py file for hyperparameter tuning to change the learning rate and max depth parameters at the very least to see if a better result could be acheived.

<img width="1475" alt="capstone_automl_rundetails" src="https://user-images.githubusercontent.com/28558135/134451414-dcd10448-1826-4a22-b5fc-2578966bfbc4.png">

<img width="2018" alt="capstone_automl_bestmodel_runID" src="https://user-images.githubusercontent.com/28558135/134451432-06ad0eef-8cbb-4432-b93b-85531a6e38b0.png">




## Hyperparameter Tuning
For the hyperparameter tuning run I chose the decision tree classifier from sklearn because I am familiar with these types of models and understand their output. For the type of problem that I am trying to solve and the outcome that I am trying to predict I thought it would have a better outcome than a simple logistic regression. The hyper parameters that I decided to modify with each run was the criterion, splitter, and the max depth of the tree using a random parameter sampling method. With the choices I provided the script I set the config to run 100 times too ensure that each combination could be tried.

```
param_sampling = RandomParameterSampling( {
        "criterion": choice("gini","entropy"),
        "splitter": choice("best", "random"),
        "max_depth": choice(250,110,10,15,60,18,20,22,24,28,30,40,50,35,55,45,70,80,90,100,150,200,85,300)
    })
hyperdrive_run_config = HyperDriveConfig(run_config=estimator,
                             hyperparameter_sampling=param_sampling,
                             policy=early_termination_policy,
                             primary_metric_name="Accuracy",
                             primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                             max_total_runs=100,
                             max_concurrent_runs=4)
```
A bandit early termination policy was used to keep the cluster from wasting time on poor combinations.

### Results
The results of 100 runs yielded a combination that provided an accuracy of 92.7%. The parameter values were "Max_depth":40, "Splitter":random, and "Criterion":gini. To improve this model I would limit the max depth choices to values under 100 as anything over seemed to overcomplicate the model. I would also try altering other hyperparamters such as the max leaf nodes and the minimum samples split.

<img width="1592" alt="capstone_hyperdrive_rundetails" src="https://user-images.githubusercontent.com/28558135/134452149-6b2499bb-f860-427f-a848-9382ea6277d5.png">

<img width="2033" alt="capstone_hyperdrive_bestrun_runid" src="https://user-images.githubusercontent.com/28558135/134452166-b25bb184-aed5-4ee0-bed7-8157204b1c3c.png">



## Model Deployment
The model that made the cut for deployment was the hyperdrive tuned model with the best accuracy.
<img width="2029" alt="capstone_hyperdrive_model_endpoint" src="https://user-images.githubusercontent.com/28558135/134452311-2991ecbd-d0d2-4d3a-87d1-07ddf852c85a.png">

Deploying the model was possibly the most difficult part of this project since the instructor only covered how to do it for an AutoML model and through the GUI. The hyperparameter tuning model required that we compose a score.py, InferenceConfig, environment.yml, and a deployment configuration. After a lot of troubleshooting and reading documentation I decided to grab the score.py from the AutoML run and repurpose it for the hyperdrive parameter run which turned out to work just fine. The environment.yml was composed using the same dependencies as the AutoML run and this also worked for the hyperdrive parameter deployment. The inference config included both of these assets and the deployment config was set to was is standard in an AutoML deployment which is 1 cpu_core and 2gb of memory.

```
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
service_name = "hyperdrive-model-service"
deploy_env = Environment.from_conda_specification(name="project_environment",file_path="conda_env.yml")
inference_config = InferenceConfig(entry_script="./score.py", environment=deploy_env)
deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 2)
service = Model.deploy(workspace=ws,
                          name=service_name,
                          models=[model],
                          inference_config=inference_config,
                          deployment_config=deployment_config,
                          overwrite=True)
                          
service.wait_for_deployment(show_output=True)
```
To use this endpoint and receive a prediction from it you have to submit a json data object like this, with a method of "predict". Json dump this data object into a variable and include it as the body of the request. The endpoint was not secured with an API key so only a header with the conent type is needed as shown below. In fact, since I have not closed my endpoint prior to submitting this project you can run the below code and receive a response from my deployed endpoint. I will leave the endpoint open until the assignment is approved.
```
scoring_uri = 'http://b5f0f797-f0d3-4647-af72-4462aa93ff41.eastus.azurecontainer.io/score'
data = {
  "Inputs":{
      "data":
        [
          {
            "age": 50,
            "anaemia": 1,
            "creatinine": 452,
            "diabetes": 0,
            "ejection_fraction": 35,
            "high_blood_pressure": 1,
            "platelets": 196000,
            "serum_creatinine": 1.1,
            "serum_sodium": 130,
            "sex": 0,
            "smoking": 1,
            "time": 150
          },
          {
            "age": 45,
            "anaemia": 0,
            "creatinine": 352,
            "diabetes": 1,
            "ejection_fraction": 60,
            "high_blood_pressure": 0,
            "platelets": 200000,
            "serum_creatinine": 1.9,
            "serum_sodium": 123,
            "sex": 1,
            "smoking": 0,
            "time": 100
          }
      ]
  },
  "method":"predict"

    }
    input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)


headers = {'Content-Type': 'application/json'}

resp = requests.post(scoring_uri, input_data, headers=headers)
```
The response will be an array of "Results", where a 0 is a "No Heart Failure" outcome and a 1 is a "Heart Failure" outcome in the order of the data that was sent like this. These values can be changed to a no and yes respectively once received.
```
"Results": [0, 1]}
```
## Screen Recording

https://youtu.be/T1L0DTCtNfE

## Standout Suggestions
My standout was to turn the model into an onnx model to have improved portability to other platforms. I demonstrate this at the end of the hyperparameter_tuning.ipynb file, but here is the code for a quick look.
```
import onnxmltools
import joblib
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
# loading best model from outputs folder
model = joblib.load("./outputs/hyperdrive-model.joblib")
#defining intial type parameter. The model takes 12 integers and 1 float as input, workaround it to
# declare all as float since it will not matter to the model
initial_type =[('float_input', FloatTensorType([None,13]))]
#using onnxmltools function convert_sklearn to convert the decision tree to onnx model
onxmdl = onnxmltools.convert_sklearn(model, name="onxmdl", initial_types=initial_type)
print(onxmdl)
```
Cheers and happy analysis!
