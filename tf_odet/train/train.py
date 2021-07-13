import os
import yaml
import tensorflow as tf
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


# limit gpu memory growth to required amount only
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# load config files for training
with open("model_store/mobilenet_v2_train.yaml") as file:
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
