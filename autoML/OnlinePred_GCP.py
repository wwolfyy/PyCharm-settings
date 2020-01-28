# %% model reference
# dataset:            KOSPI_bNs_20191115_32
# model display name: GCP_KOSPI_bNs_20191115_32
# model name:         projects/783922927712/locations/us-central1/models/TBL1401477103181889536
#
# KOSPI_f1_bNs_FC_CC_20191127_8
# KOSPI_f1_bNs_FC_C_20200111024448
# TBL7108522993478795264
#
# KOSPI_f1_bNs_FC_CC_20191210_8
# KOSPI_f1_bNs_FC_C_20200112065906
# TBL5812049250749513728
#
# KOSPI_f1_bNs_FC_CC_20191220_8
# KOSPI_f1_bNs_FC_C_20200113035632
# TBL3092438025771155456

model_name_1 = '' # newest
model_name_2 = ''
model_name_3 = '' # oldest

# %% start mtlab engine
# matlab engine
import matlab
import matlab.engine

# # connect to esxsiting session
# # (Must run this in matlab: matlab.engine.shareEngine)
# eng = matlab.engine.start_matlab() # use this to just call matlab functions from python
# matlab.engine.find_matlab()
# eng = matlab.engine.connect_matlab('MATLAB_12792')

# start connected session
eng = matlab.engine.start_matlab('-desktop -r "format short"') # Will open matlab desktop. Work in this instance

# %% Deploy models: GCP

# authenticate for project
%env GOOGLE_APPLICATION_CREDENTIALS C:\GoogleCloudSDK\ivory-volt-263012-bc0201c2583d.json

# basic params
PROJECT_ID = "ivory-volt-263012" #@param {type:"string"}
COMPUTE_REGION = "us-central1" # Currently only supported region.
BUCKET_NAME = "automl_tables_20191225"

# import modules
from google.cloud import automl_v1beta1 as automl
import google.cloud.automl_v1beta1.proto.data_types_pb2 as data_types

# initializa modules
automl_client = automl.AutoMlClient()
tables_client = automl.TablesClient(project=PROJECT_ID, region=COMPUTE_REGION)

# get list of models & deployment state
list_models = tables_client.list_models()
models_ref = { model.display_name: model.name for model in list_models }
models_ref

list_models = tables_client.list_models()
models_deployment_states = { model.display_name: model.deployment_state for model in list_models }
models_deployment_states

# %%
# deploy model
deploy_model_response = tables_client.deploy_model(
    model_name='projects/783922927712/locations/us-central1/models/TBL3092438025771155456'
)

# %% Get prediction

# GCP
input_GCP_model_1 = eng.workspace['data4model_1'] # get from matlab

prediction_result = tables_client.predict(
    model_name=model_name_1,
    inputs=input_GCP_model_1
)

pred_GCP_model_1 = {'low': prediction_result.payload[0].tables.prediction_interval.start,
                'mid': prediction_result.payload[0].tables.value.number_value,
                'high': prediction_result.payload[0].tables.prediction_interval.end
                }

eng.workspace['pred_GCP_model_3'] = pred_GCP_model_3 # push to matlab

# %% clean up

# GCP
undeploy_model_response = tables_client.undeploy_model(model_name='projects/783922927712/locations/us-central1/models/TBL3092438025771155456')

# stop matlab engine
eng.quit()




