# %% import generic modules
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

# %% import azure modules
import azureml.core
from azureml.core import Experiment, Workspace, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.train.automl import AutoMLConfig

# %% connect to (or set up) workspace
import os

subscription_id = "1a653f13-bee2-4093-8c1b-97dddb149720"
resource_group = "autoML"
workspace_name = "ML_20200125"
#workspace_region = "eastus2"

try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    # write the details of the workspace to a configuration file to the notebook library
    ws.write_config()
    print("Workspace configuration succeeded. Skip the workspace creation steps below")
except:
    print("Workspace not accessible. Change your parameters or create a new workspace below")

# %% setup compute (GPU & CPU)
gpu_cluster_name = "gpu-cluster"
cpu_cluster_name = "cpu-cluster"

# GPU
try:
    gpu_cluster = ComputeTarget(workspace=ws, name=gpu_cluster_name)
    print("Found existing gpu cluster")
except ComputeTargetException:
    print("Creating new gpu-cluster")
    compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_NV6",
                                                           min_nodes=0,
                                                           max_nodes=2)
    gpu_cluster = ComputeTarget.create(ws, gpu_cluster_name, compute_config)
    gpu_cluster.wait_for_completion(show_output=True) # wait and show output

# CPU
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print("Found existing cpu-cluster")
except ComputeTargetException:
    print("Creating new cpu-cluster")
    compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_D2_V2",
                                                           min_nodes=0,
                                                           max_nodes=4)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)
    cpu_cluster.wait_for_completion(show_output=True) # wait and show output

# check if listed in ws
cts = ws.compute_targets
cts

# %% set up (or connect to) experiment
# choose name for the run history container in the workspace
experiment_name = 'automl-forecasting-energydemand'
experiment = Experiment(ws, experiment_name)

output = {}
output['SDK version'] = azureml.core.VERSION
output['Subscription ID'] = ws.subscription_id
output['Workspace'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Run History Name'] = experiment_name
pd.set_option('display.max_colwidth', -1)
outputDf = pd.DataFrame(data = output, index = [''])
outputDf.T

# %% attach compute (gpu)



