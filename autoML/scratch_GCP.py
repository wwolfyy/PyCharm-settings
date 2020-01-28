# get authenticated
def GCP_autheticate(PROJECT_ID, COMPUTE_REGION):


# upload data to GCS
from google.cloud import storage
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

# get deployment state (either will work)
list_models = tables_client.list_models()
for model in list_models:
    print(model.display_name + ':' + str(model.deployment_state)) # 0 for deployed, 1 for deploying, 2 for undeployed

list_models = tables_client.list_models()
models_deployment_states = { model.display_name: model.deployment_state for model in list_models }
models_deployment_states

# download from GCS
def download_blob(BUCKET_NAME, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

# %% bulid and install matalb engine for python

# run this in terminal
cd C:\Program Files\MATLAB\R2019b\extern\engines\python # where matlab engine for python is located
# build nad install here; -- can build somewhere else
python setup.py build --build-base="C:\Users\jp\Anaconda3\envs\py37_cloudML" install --prefix="C:\Users\jp\Anaconda3\envs\py37_cloudML"
cd C:\Users\jp\Anaconda3\envs\py37_cloudML\ # back to current directory

# run this in matlab
pe = pyenv('Version','C:\Users\jp\Anaconda3\envs\py37_cloudML\python.exe')

# run this here
import matlab
import matlab.engine
eng = matlab.engine.start_matlab()

# %%
! gcloud config list

exec(open("C:\\test.py").read())

assert all([
    PROJECT_ID,
    COMPUTE_REGION,
    # DATASET_DISPLAY_NAME,
    # INPUT_CSV_NAME,
    # MODEL_DISPLAY_NAME,
])

import ipywidgets as widgets

# List table specs.
list_table_specs_response = tables_client.list_table_specs(dataset=dataset)
table_specs = [s for s in list_table_specs_response]

# List column specs.
list_column_specs_response = tables_client.list_column_specs(dataset=dataset)
column_specs = {s.display_name: s for s in list_column_specs_response}
column_specs

# Verify the status by checking the example_count field.
dataset_name = 'projects/783922927712/locations/us-central1/datasets/TBL1712372812028575744'
dataset = tables_client.get_dataset(dataset_name=dataset_name)
dataset

#
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# %% GCP Define training parameters -- using automl client

target_column_name = 'target' #@param {type: 'string'}
target_column_spec = column_specs[target_column_name]
target_column_id = target_column_spec.name.rsplit('/', 1)[-1]
print('Target column ID: {}'.format(target_column_id))

split_column_name = 'split_tag' #@param {type: 'string'}
split_column_spec = column_specs[split_column_name]
split_column_id = split_column_spec.name.rsplit('/', 1)[-1]
print('Split column ID: {}'.format(split_column_id))

weight_column_name = 'weight_vector' #@param {type: 'string'}
weight_column_spec = column_specs[weight_column_name]
weight_column_id = weight_column_spec.name.rsplit('/', 1)[-1]
print('Weight column ID: {}'.format(weight_column_id))

# Define the values of the fields to be updated.
update_dataset_dict = {
    'name': dataset_name,
    'tables_dataset_metadata': {
        'target_column_spec_id': target_column_id,
        'ml_use_column_spec_id': split_column_id,
        'weight_column_spec_id': weight_column_id,
    }
}
update_dataset_response = automl_client.update_dataset(update_dataset_dict)
update_dataset_response

# Delete model resource.
tables_client.delete_model(model_name=model_name)

# Delete dataset resource.
tables_client.delete_dataset(dataset_name=dataset_name)

# Delete Cloud Storage objects that were created.
! gsutil -m rm -r gs://$BUCKET_NAME

# If training model is still running, cancel it.
automl_client.transport._operations_client.cancel_operation(operation_id)