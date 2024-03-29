# must be imported at the top for setting env vars
import tf_odet.set_tf_env_vars
import argparse
from subprocess import call

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
        description='Export trained tf2 checkpoints in savedmodel fmt using tf2 object detection api')
    parser.add_argument("-c",
                        "--model_config_yaml",
                        default="model_store/mobilenet_v2_train.yaml",
                        help='Def: "model_store/mobilenet_v2_train.yaml". YAML file path with training configs')
    parser.add_argument("-o",
                        "--output_directory",
                        default="inference_graph",
                        help="Def: output_directory. Dir where model will be exported to")
    args = parser.parse_args()

    return args


def test(model_config_yaml, output_directory):
    # load config files for exporting
    with open("model_store/mobilenet_v2_train.yaml") as file:
        config_dict = yaml.full_load(file)

    # Add a line to the tf_utils.py file.
    # This is a temporary fix to a exporting issue occuring when using the API with Tensorflow 2.
    # This code will be removed as soon as the TF Team puts out a fix.
    # import site
    # site_package_path = ''.join(site.getsitepackages())
    # tf_utils_path = site_package_path + "/tensorflow/python/keras/utils/tf_utils.py"
    #
    # with open(tf_utils_path) as f:
    #     tf_utils = f.read()
    #
    # with open(tf_utils_path, 'w') as f:
    #     # Set labelmap path
    #     throw_statement = "raise TypeError('Expected Operation, Variable, or Tensor, got ' + str(x))"
    #     tf_utils = tf_utils.replace(throw_statement, "if not isinstance(x, str):" + throw_statement)
    #     f.write(tf_utils)

    # ###################################
    # calling tf2 obj det export function
    # ###################################

    export_args = ["python3", "models/research/object_detection/exporter_main_v2.py",
                   "--trained_checkpoint_dir", config_dict["model_dir"],
                   "--output_directory", output_directory,
                   "--pipeline_config_path", config_dict["pipeline_config_path"]]

    rtn_status = call(export_args)
    print(f"Return status of export script call {rtn_status}")
    return rtn_status


def main():
    args = parse_args()
    test(args.model_config_yaml, args.output_directory)


if __name__ == "__main__":
    main()
