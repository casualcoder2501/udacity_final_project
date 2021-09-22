import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib
import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

data_sample = PandasParameterType(pd.DataFrame({"age": pd.Series([0.0], dtype="float64"), "anaemia": pd.Series([0], dtype="int64"), "creatinine_phosphokinase": pd.Series([0], dtype="int64"), "diabetes": pd.Series([0], dtype="int64"), "ejection_fraction": pd.Series([0], dtype="int64"), "high_blood_pressure": pd.Series([0], dtype="int64"), "platelets": pd.Series([0.0], dtype="float64"), "serum_creatinine": pd.Series([0.0], dtype="float64"), "serum_sodium": pd.Series([0], dtype="int64"), "sex": pd.Series([0], dtype="int64"), "smoking": pd.Series([0], dtype="int64"), "time": pd.Series([0], dtype="int64")}))
input_sample = StandardPythonParameterType({'data': data_sample})
method_sample = StandardPythonParameterType("predict")

result_sample = NumpyParameterType(np.array([0]))
output_sample = StandardPythonParameterType({'Results':result_sample})



def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'hyperdrive-model.joblib')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)

    model = joblib.load(model_path)



@input_schema('method', method_sample, convert_to_provided_type=False)
@input_schema('Inputs', input_sample)
@output_schema(output_sample)
def run(Inputs, method="predict"):
    data = Inputs['data']
    if method == "predict_proba":
        result = model.predict_proba(data).values
    elif method == "predict":
        result = model.predict(data)
    else:
        raise Exception(f'Invalid predict method argument received ({method})')
    return {'Results':result.tolist()}