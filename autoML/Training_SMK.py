# %% import generic packages
import time
import numpy as np
np.random.seed(1)
import pandas as pd
import json
import matplotlib.pyplot as plt

# %% in stall s3fs (S3 file system interface for python)
#! conda install -y s3fs (boto3 gets installed too)
# ! pip install sagemaker

# %% import AWS packages
import boto3
import s3fs
import sagemaker
from sagemaker import get_execution_role

# %% upload data
# create S3 bucket & upload data
s3 = boto3.client('s3')
response = s3.list_buckets()
for bucket in response['Buckets']:
    print(f' {bucket["Name"]}')

# Upload a new file
input_file_path = r'C:\Users\jp\Downloads\test_boto.csv'
data = open(input_file_path, 'rb')
s3.upload_file(
    input_file_path, 'sagemaker-us-east-2-168866271170', 'test_data.csv'
)
data.close()

# %% setup and start sagemaker / docker image
sagemaker_session = sagemaker.Session()
#role = get_execution_role()
role = 'arn:aws:iam::168866271170:role/service-role/AmazonSageMaker-ExecutionRole-20200118T225895'

bucket = ' sagemaker-us-east-2-168866271170'
prefix = 'sagemaker/Test-deepar'

s3_data_path = "{}/{}/data".format(bucket, prefix)
s3_output_path = "{}/{}/output".format(bucket, prefix)

region = sagemaker_session.boto_region_name

image_name = sagemaker.amazon.amazon_estimator.get_image_uri(region, "forecasting-deepar", "latest")


