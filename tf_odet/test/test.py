import os
import yaml
import argparse
from subprocess import call

# os environments must be set at th beginning of the file top use GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["OMP_NUM_THREADS"] = "15"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 0, 1, 2, 3, 4, 5, 6, 7
# import tf after setting ENV vars
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
        description='Test model using tensorflow 2 object detection api')
    parser.add_argument("-c",
                        "--model_config_yaml",
                        default="model_store/mobilenet_v2_train.yaml",
                        help='Def: "model_store/mobilenet_v2_train.yaml". YAML file path with training configs')
    args = parser.parse_args()
    return args


def test(model_config_yaml):
    # load config files for testing
    with open("model_store/mobilenet_v2_train.yaml") as file:
        config_dict = yaml.full_load(file)

    # #######################################
    # calling tf2 obj det validation function
    # #######################################

    val_args = ["python3", "models/research/object_detection/model_main_tf2.py",
                "--pipeline_config_path", config_dict["pipeline_config_path"],
                "--model_dir", config_dict["model_dir"],
                "--checkpoint_dir", config_dict["model_dir"]]

    rtn_status = call(val_args)
    print(f"Return status of validation script call {rtn_status}")
    return rtn_status


def main():
    args = parse_args()
    test(args.model_config_yaml)


if __name__ == "__main__":
    main()