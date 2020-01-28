# %% groundwork

# authenticate for project
%env GOOGLE_APPLICATION_CREDENTIALS C:\GoogleCloudSDK\ivory-volt-263012-bc0201c2583d.json

# basic params
PROJECT_ID = "ivory-volt-263012" #@param {type:"string"}
COMPUTE_REGION = "us-central1" # Currently only supported region.
BUCKET_NAME = "automl_tables_20191225"

# import modules
from google.cloud import automl_v1beta1 as automl
import google.cloud.automl_v1beta1.proto.data_types_pb2 as data_types
from google.cloud import storage
import pandas as pd

# initializa modules
automl_client = automl.AutoMlClient()
tables_client = automl.TablesClient(project=PROJECT_ID, region=COMPUTE_REGION)

# %% define basics
DIR_PATH = "data_input/batch_prediction_input"
DATA_LOCATION = r'C:\Users\jp\Google Drive\MATLAB data files\Data_Matlab\Features\KOSPI_bNs\B1_F1_CC\batch_pred_data\GCP\*'

# %% upload batch prediction feature data (# QUOTE the OBJECT_LOCATION variable with $ attached)
! gsutil -m cp -r "$DATA_LOCATION" gs://$BUCKET_NAME/$DIR_PATH
! gsutil ls gs://$BUCKET_NAME/$DIR_PATH

# %% get list of uploaded files
storage_client = storage.Client()
batch_datum = storage_client.list_blobs(BUCKET_NAME, prefix=DIR_PATH+'/', delimiter='/')
batch_data_ref = { blob.name for blob in batch_datum }
batch_data_ref

# %% get list of models to use
list_models = tables_client.list_models()
models_ref = { model.display_name: model.name for model in list_models }
models_ref

# %% assign file to variables and put them in list
#batch_data_2019_1115_32 = 'gs://' + BUCKET_NAME + '/' + 'data_input/batch_prediction_input/GCP_batch_pred_data_KOSPI_f1_bNs_FC_CC_20191115_32.csv'
batch_data_2019_1127_8 = 'gs://' + BUCKET_NAME + '/' + 'data_input/batch_prediction_input/GCP_batch_pred_data_KOSPI_f1_bNs_FC_CC_20191127_8.csv'
batch_data_2019_1210_8 = 'gs://' + BUCKET_NAME + '/' + 'data_input/batch_prediction_input/GCP_batch_pred_data_KOSPI_f1_bNs_FC_CC_20191210_8.csv'
batch_data_2019_1220_8 = 'gs://' + BUCKET_NAME + '/' + 'data_input/batch_prediction_input/GCP_batch_pred_data_KOSPI_f1_bNs_FC_CC_20191220_8.csv'

batch_data = [batch_data_2019_1127_8, batch_data_2019_1210_8, batch_data_2019_1220_8]

# %% assign models to variables and put them in list
#model_2019_1115_32 = tables_client.get_model(model_name='projects/783922927712/locations/us-central1/models/TBL1401477103181889536')
model_2019_1127_8 = tables_client.get_model(model_name='projects/783922927712/locations/us-central1/models/TBL7108522993478795264')
model_2019_1210_8 = tables_client.get_model(model_name='projects/783922927712/locations/us-central1/models/TBL5812049250749513728')
model_2019_1220_8 = tables_client.get_model(model_name='projects/783922927712/locations/us-central1/models/TBL3092438025771155456')

models = [model_2019_1127_8, model_2019_1210_8, model_2019_1220_8]

# %% run batch porediction
gcs_output_folder_name = 'predictions/batch_predictions'  # no need for bucket name
GCS_BATCH_PREDICT_OUTPUT = 'gs://{}/{}/'.format(BUCKET_NAME, gcs_output_folder_name)

batch_predict_response = tables_client.batch_predict(
    model=models[3],
    gcs_input_uris=batch_data[3],
    gcs_output_uri_prefix=GCS_BATCH_PREDICT_OUTPUT,
)

# %% predict in loop

# for i in range(len(models)):
#     batch_predict_response = tables_client.batch_predict(
#         model = models[i],
#         gcs_input_uris = batch_data[i],
#         gcs_output_uri_prefix = GCS_BATCH_PREDICT_OUTPUT,
#     )

# %% get output path & download to local folder
batch_predict_result = batch_predict_response.result() # should exeute this method first
output_path = batch_predict_response.metadata\
                  .batch_predict_details.output_info\
                  .gcs_output_directory + "/tables_1.csv"
download_path = r'C:\Users\jp\Google Drive\MATLAB data files\Commercialize\AutoML forecasts\KOSPI_F1_20191115_32\batchPred.csv'
! gsutil cp $output_path "$download_path"

# %% create pandas dataframe here
batch_predict_results_location = batch_predict_response.metadata\
                                 .batch_predict_details.output_info\
                                 .gcs_output_directory
table = pd.read_csv('{}/tables_1.csv'.format(batch_predict_results_location))
y = table["date","predicted_target"]
