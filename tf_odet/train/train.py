# must be imported at the top for setting env vars
import tf_odet.set_tf_env_vars
from subprocess import call
import argparse

import yaml
import tensorflow as tf

# limit gpu memory growth to required amount only
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train model using tensorflow 2 object detection api')
    parser.add_argument("-c",
                        "--model_config_yaml",
                        default="model_store/mobilenet_v2_train.yaml",
                        help='Def: "model_store/mobilenet_v2_train.yaml". YAML file path with training configs')
    args = parser.parse_args()
    return args


def train(model_config_yaml):
    # load config files for training
    with open(model_config_yaml) as file:
        config_dict = yaml.full_load(file)

    # #####################################
    # calling tf2 obj det training function
    # #####################################

    train_args = ["python3", "models/research/object_detection/model_main_tf2.py",
                  "--pipeline_config_path", config_dict["pipeline_config_path"],
                  "--model_dir", config_dict["model_dir"],
                  "--alsologtostderr",
                  "--num_train_steps", str(config_dict["num_steps"]),
                  "--sample_1_of_n_eval_examples", "1",
                  "--num_eval_steps", str(config_dict["num_eval_steps"])]

    rtn_status = call(train_args)
    print(f"Return status of training script call {rtn_status}")
    return rtn_status


def main():
    args = parse_args()
    train(args.model_config_yaml)


if __name__ == "__main__":
    main()
