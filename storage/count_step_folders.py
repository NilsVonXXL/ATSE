import os

root_dir = 'newDataset'
step_count = 0
for domain in os.listdir(root_dir):
    domain_path = os.path.join(root_dir, domain)
    if not os.path.isdir(domain_path):
        continue
    for data_subdir in os.listdir(domain_path):
        data_path = os.path.join(domain_path, data_subdir)
        if not os.path.isdir(data_path):
            continue
        for net_num in os.listdir(data_path):
            net_path = os.path.join(data_path, net_num)
            if not os.path.isdir(net_path):
                continue
            for input_folder in os.listdir(net_path):
                input_path = os.path.join(net_path, input_folder)
                if not os.path.isdir(input_path):
                    continue
                for step_folder in os.listdir(input_path):
                    step_path = os.path.join(input_path, step_folder)
                    if os.path.isdir(step_path) and step_folder.startswith('step_'):
                        step_count += 1
print('Total step folders:', step_count)
