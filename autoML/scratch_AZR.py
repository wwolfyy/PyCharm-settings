# % create workspace
from azureml.core import Workspace


# Create the workspace using the specified parameters
ws = Workspace.create(name = workspace_name,
                      subscription_id = subscription_id,
                      resource_group = resource_group, 
                      location = workspace_region,
                      create_resource_group = True,
                      sku = 'basic',
                      exist_ok = True)
ws.get_details()


# write the details of the workspace to a configuration file to the notebook library
ws.write_config()


# start Jupyter notebook in browser
jupyter notebook # in terminal


# create gpu cluster
compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_NV6",
                                                        min_nodes=0,
                                                        max_nodes=4)
gpu_cluster = ComputeTarget.create(ws, gpu_cluster_name, compute_config)
gpu_cluster.wait_for_completion(show_output=True)


# create cpu cluster
cpu_cluster_name = "cpu-cluster" # define name

try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print("Found existing cpu-cluster")
except ComputeTargetException:
    print("Creating new cpu-cluster")
        
    compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_D2_V2",
                                                           min_nodes=0,
                                                           max_nodes=4)
    
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

    cpu_cluster.wait_for_completion(show_output=True)


# Squash warning messages for cleaner output in the notebook
warnings.showwarning = lambda *args, **kwargs: None


# get current workspace in Azure
ws = Workspace.from_config()