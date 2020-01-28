# check variable size in memory and output to CSV file
def list_variable_size_csv(savepath):
    import sys
    import csv
    aaa = globals()
    with open(savepath, 'w', newline='') as csvfile:
        metric_writer = csv.writer(csvfile, delimiter=',')
        metric_writer.writerow(['variable name', 'size in byte'])
    for key, value in aaa.items():
        with open(savepath, 'a', newline='') as csvfile:
            metric_writer = csv.writer(csvfile, delimiter=',')
            metric_writer.writerow([str(key), str(sys.getsizeof(value))])


def select_gpu(computer_name):
    import os
    if os.environ['COMPUTERNAME'] == computer_name:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        selected_gpu = input("Select GPU by PCI slot # (0:GTX-1070ti, 1:RTX-2080ti, enter:all): ") or "all"
        if selected_gpu == '0':
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 1070 is on first PCI slot
            print("Selected device(" + os.environ['COMPUTERNAME'] + "): GTX 1070ti")
        elif selected_gpu == '1':
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 2080 is on second PCI slot
            print("RTX 2080ti selected on " + os.environ['COMPUTERNAME'])
        else:
            print("All GPUs are visible on " + os.environ['COMPUTERNAME'])


def extract_from_dict(dict_, keys_):
    # keys_: list of keys to extract
    extracted = {k: dict_[k] for k in (keys_)}
    return extracted


def delete_from_dict(dict_, keys_):
    # keys_: list of keys to extract
    for key_value in keys_:
        del dict_[key_value]

# loop_obj = extract_from_dict(loop_obj,[0,1,2,3,8,9,10,11,20,21,22,23])