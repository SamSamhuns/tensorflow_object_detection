import re
import yaml
import os.path as osp

# ###################################
# training path parameters, EDIT here
# ###################################

num_classes = 1
batch_size = 64
num_steps = 10000
num_eval_steps = 1000

# having the leading './' is important
DATASET_PATH = "./data/coco_person"
MSTORE_PATH = "./model_store"
model_dir = "./training"

model_training_config_path = osp.join(MSTORE_PATH, "mobilenet_v2_train.yaml")
train_record_path = osp.join(DATASET_PATH, "train.record")
test_record_path = osp.join(DATASET_PATH, "test.record")
labelmap_path = osp.join(DATASET_PATH, "labelmap.pbtxt")

pipeline_config_path = osp.join(MSTORE_PATH, "mobilenet_v2.config")
fine_tune_checkpoint = osp.join(MSTORE_PATH, "mobilenet_v2/mobilenet_v2.ckpt-1")

##############################################
# End of param setting, Verifying params below
##############################################

param_dicts = {"train_record_path": train_record_path,
               "test_record_path": test_record_path,
               "labelmap_path": labelmap_path,
               "model_dir": model_dir,
               "pipeline_config_path": pipeline_config_path,
               "fine_tune_checkpoint": fine_tune_checkpoint,
               "num_classes": 1,
               "batch_size": 32,
               "num_steps": 7500,
               "num_eval_steps": 1000}

with open(model_training_config_path, 'w') as file:
    documents = yaml.dump(param_dicts, file)

ignore_key_chk = {"model_dir", "fine_tune_checkpoint",
                  "num_classes", "batch_size", "num_steps", "num_eval_steps"}

for key, path in param_dicts.items():
    if key not in ignore_key_chk and not osp.exists(path):
        print(f"Aborting. {key} path at {path} does not exist.")
        exit(1)
    print(f"\t {key} = {path}")

# ###############################
# config file params, DO NOT edit
# ###############################

with open(pipeline_config_path) as f:
    config = f.read()

# edit config file to new params
with open(pipeline_config_path, 'w') as f:
    # Set labelmap path
    config = re.sub('label_map_path: ".*?"',
                    'label_map_path: "{}"'.format(labelmap_path), config)

    # Set fine_tune_checkpoint path
    config = re.sub('fine_tune_checkpoint: ".*?"',
                    'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), config)

    # Set train tf-record file path
    config = re.sub('(input_path: ".*?)(train)(.*?")',
                    'input_path: "{}"'.format(train_record_path), config)

    # Set test tf-record file path
    config = re.sub('(input_path: ".*?)(val|test|eval)(.*?")',
                    'input_path: "{}"'.format(test_record_path), config)

    # Set number of classes.
    config = re.sub('num_classes: [0-9]+',
                    'num_classes: {}'.format(num_classes), config)

    # Set batch size
    config = re.sub('batch_size: [0-9]+',
                    'batch_size: {}'.format(batch_size), config)

    # Set training steps
    config = re.sub('num_steps: [0-9]+',
                    'num_steps: {}'.format(num_steps), config)

    f.write(config)

print(
    f"Completed changing configurations in the config to new parameters in {pipeline_config_path}")
print(
    f"Train configuration YAML file is saved in {model_training_config_path}")
