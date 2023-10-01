import tensorflow as tf
import os
import train
import preprocess
from absl import flags


# home = os.getcwd()
# record_dir = os.path.join(home, 'record')
# save_base = os.path.join(home, 'log')
# log_dir = os.path.join(save_base)
# data_dir = os.path.join(home, 'filter')
# pred_dir = os.path.join(home, 'result')

# for dirx in [save_base, record_dir, log_dir, data_dir, pred_dir]:
#     if not os.path.exists(dirx):
#         os.mkdir(dirx)

# train_record = os.path.join(record_dir, 'train.json')
# test_record = os.path.join(record_dir, 'test.json')
# train_meta = os.path.join(record_dir, 'train.meta')
# test_meta = os.path.join(record_dir, 'test.meta')
# status_label = os.path.join(data_dir, 'status.label')

flags_d = flags.FLAGS

flags_d.DEFINE_string('train_json', "train_record", 'the processed train json file')
flags_d.DEFINE_string('test_json', "test_record", 'the processed test json file')
flags_d.DEFINE_string('train_meta', "train_meta", 'the processed train number')
flags_d.DEFINE_string('test_meta', "test_meta", 'the processed test number')
if 'log_dir' in flags.FLAGS:
    delattr(flags.FLAGS, 'log_dir')
flags_d.DEFINE_string('log_dir', "log_dir", 'where to save the log')
flags_d.DEFINE_string('model_dir', "log_dir", 'where to save the model')
flags_d.DEFINE_string('data_dir', "data_dir", 'where to read data')
flags_d.DEFINE_integer('class_num', 18, 'the class number')
flags_d.DEFINE_integer('length_block', 1, 'the length of a block')
flags_d.DEFINE_integer('min_length', 2, 'the flow under this parameter will be filtered')
flags_d.DEFINE_integer('max_packet_length', 5000, 'the largest packet length')
flags_d.DEFINE_float('split_ratio', 0.8, 'ratio of train set of target app')
flags_d.DEFINE_float('keep_ratio', 1, 'ratio of keeping the example (for small dataset test)')
flags_d.DEFINE_integer('max_flow_length_train', 200, 'the max flow length, if larger, drop')
flags_d.DEFINE_integer('max_flow_length_test', 2000, 'the max flow length, if larger, drop')
flags_d.DEFINE_string('test_model_dir', "log_dir", 'the model dir for test result')
flags_d.DEFINE_string('pred_dir', "pred_dir", 'the dir to save predict result')

flags_d.DEFINE_integer('batch_size', 128, 'train batch size')
flags_d.DEFINE_integer('hidden', 128, 'GRU dimension of hidden state')
flags_d.DEFINE_integer('layer', 2, 'layer number of length RNN')
flags_d.DEFINE_integer('length_dim', 16, 'dimension of length embedding')
flags_d.DEFINE_string('length_num', 'auto', 'length_num')

flags_d.DEFINE_float('keep_prob', 0.8, 'the keep probability for dropout')
flags_d.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags_d.DEFINE_integer('iter_num', int(1.2e5), 'iteration number')
flags_d.DEFINE_integer('eval_batch', 77, 'evaluated train batches')
flags_d.DEFINE_integer('train_eval_batch', 77, 'evaluated train batches')
flags_d.DEFINE_string('decay_step', 'auto', 'the decay step')
flags_d.DEFINE_float('decay_rate', 0.5, 'the decay rate')

flags_d.DEFINE_string('mode', 'train', 'model mode: train/prepro/test')
flags_d.DEFINE_integer("capacity", int(1e3), "size of dataset shuffle")
flags_d.DEFINE_integer("loss_save", 100, "step of saving loss")
flags_d.DEFINE_integer("checkpoint", 5000, "checkpoint to save and evaluate the model")
flags_d.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")

flags_d.DEFINE_boolean('is_cudnn', True, 'whether take the cudnn gru')
flags_d.DEFINE_float('rec_loss', 0.5, 'the parameter to control the reconstruction of length sequence')


def main(_):
    config = flags.FLAGS
    if config.length_num == 'auto':
        config.length_num = config.max_packet_length // config.length_block + 4
    else:
        config.length_num = int(config.length_num)
    if config.decay_step != 'auto':
        config.decay_step = int(config.decay_step)
    if config.mode == 'train':
        train.train(config)
    elif config.mode == 'prepro':
        preprocess.preprocess(config)
    elif config.mode == 'test':
        print(config.test_model_dir)
        train.predict(config)
    else:
        print('unknown mode, only support train now')
        raise Exception


if __name__ == '__main__':
    tf.app.run()
