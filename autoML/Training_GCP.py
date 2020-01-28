# %% groundwork
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# authenticate for project
%env GOOGLE_APPLICATION_CREDENTIALS C:\GoogleCloudSDK\ivory-volt-263012-bc0201c2583d.json

# basic params
PROJECT_ID = "ivory-volt-263012" #@param {type:"string"}
COMPUTE_REGION = "us-central1" # Currently only supported region.
BUCKET_NAME = "automl_tables_20191225"
DATA_LOCATION = r'C:\Users\jp\Google Drive\MATLAB data files\Data_Matlab\Features\KOSPI_bNs\B1_F1_CC\GCP_KOSPI_f1_bNs_FC_CC_20191230_0.csv'
INPUT_CSV_NAME = 'GCP_KOSPI_f1_bNs_FC_CC_20191230_0'
MODEL_DISPLAY_NAME = 'KOSPI_f1_bNs_FC_CC_20191230_0'
DATASET_DISPLAY_NAME = 'KOSPI_f1_bNs_FC_CC_20191230_0'

# import modules
from google.cloud import automl_v1beta1 as automl
import google.cloud.automl_v1beta1.proto.data_types_pb2 as data_types
from google.cloud import storage
import matplotlib.pyplot as plt

# initialize modules
automl_client = automl.AutoMlClient()
tables_client = automl.TablesClient(project=PROJECT_ID, region=COMPUTE_REGION)

# %% create and import dataset

GCS_DATASET_URI = 'gs://{}/{}.csv'.format(BUCKET_NAME, INPUT_CSV_NAME)

# upload data to GCS
# ! gsutil ls gs://$BUCKET_NAME
! gsutil cp "$DATA_LOCATION" $GCS_DATASET_URI

# Create dataset.
dataset = tables_client.create_dataset(
          dataset_display_name=DATASET_DISPLAY_NAME)
dataset_name = dataset.name
dataset

# List the datasets.
list_datasets = tables_client.list_datasets()
datasets = { dataset.display_name: dataset.name for dataset in list_datasets }
datasets

# %% import data from GCS to project
import_data_response = tables_client.import_data(
    dataset=dataset,
    gcs_input_uris=GCS_DATASET_URI
)
print('Dataset import operation: {}'.format(import_data_response.operation))

# Synchronous check of operation status. Wait until import is done.
print('Dataset import response: {}'.format(import_data_response.result()))
print('Dataset import response: {}'.format(import_data_response.done()))

# %% review imported dataset

# # List table specs.
# list_table_specs_response = tables_client.list_table_specs(dataset=dataset)
# table_specs = [s for s in list_table_specs_response]
#
# List column specs.
list_column_specs_response = tables_client.list_column_specs(dataset=dataset)
column_specs = {s.display_name: s for s in list_column_specs_response}

# Print Features and data_type.
features = [(key, data_types.TypeCode.Name(value.data_type.type_code))
            for key, value in column_specs.items()]
print('Feature list:\n')
for feature in features:
    print(feature[0],':', feature[1])
#
# # Table schema pie chart.
# type_counts = {}
# for column_spec in column_specs.values():
#     type_name = data_types.TypeCode.Name(column_spec.data_type.type_code)
#     type_counts[type_name] = type_counts.get(type_name, 0) + 1
#
# plt.pie(x=type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
# plt.axis('equal')
# plt.show()

# %% Define training parameters -- using TablesClient

# set target
column_spec_display_name = 'target' #@param {type:'string'}
update_dataset_response = tables_client.set_target_column(
    dataset=dataset,
    column_spec_display_name=column_spec_display_name,
)
update_dataset_response

# set data split
column_spec_display_name = 'split_tag' #@param {type:'string'}
update_dataset_response = tables_client.set_test_train_column(
    dataset=dataset,
    column_spec_display_name=column_spec_display_name,
)
update_dataset_response

# set weight
column_spec_display_name = 'weight_vector' #@param {type:'string'}
update_dataset_response = tables_client.set_weight_column(
    dataset=dataset,
    column_spec_display_name=column_spec_display_name,
)
update_dataset_response

# The number of hours to train the model.
model_train_hours = 1

# %% train

create_model_response = tables_client.create_model(
    model_display_name=MODEL_DISPLAY_NAME,
    dataset=dataset,
    train_budget_milli_node_hours=model_train_hours*1000,
    exclude_column_spec_names=['date','target','split_tag','weight_vector'],
    optimization_objective='MINIMIZE_RMSE',
    disable_early_stopping=False,
)

operation_id = create_model_response.operation.name
print('Create model operation: {}'.format(create_model_response.operation))

# %% get model details

# Wait until model training is done.
model = create_model_response.result()
model_name = model.name
model_name

# %% get evaluation metrics & feature importance

metrics= [x for x in tables_client.list_model_evaluations(model=model)][-1]
metrics.regression_evaluation_metrics

feat_list = [(x.feature_importance, x.column_display_name) for x in model.tables_model_metadata.tables_model_column_info]
feat_list.sort(reverse=True)
feat_list[:15]

# %% clean up

# # Delete model resource.
# tables_client.delete_model(model_name=model_name)
#
# # Delete dataset resource.
# tables_client.delete_dataset(dataset_name=dataset_name)
#
# # Delete Cloud Storage objects that were created.
# ! gsutil - m
# rm - r
# gs: // $BUCKET_NAME
#
# # If training model is still running, cancel it.
# automl_client.transport._operations_client.cancel_operation(operation_id)
