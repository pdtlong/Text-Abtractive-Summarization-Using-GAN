
import os 
# logging.disable(logging.WARNING) 
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


def load_ckpt(saver, sess, ckpt_dir="train"):
    while True:
        try:
            latest_filename = None if ckpt_dir == "eval" else None
            ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
            tf.logging.info('Đang load checkpoint %s', ckpt_state.model_checkpoint_path)
            print("Đang load  checkpoint {}".format(ckpt_state.model_checkpoint_path))
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except:
            tf.logging.info("Không load được checkpoint từ %s. tiếnh hành Sleeping trong %i secs...", ckpt_dir, 10)
            time.sleep(10)
