import logging, os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

from collections import namedtuple
from batcher import Batcher
from data import Vocab
from generator import Generator
from discriminator import Discriminator
from decode import BeamSearchDecoder
import trainer as trainer
import util
import tensorflow as tf


# Data dirs
tf.app.flags.DEFINE_string('data_path', '', 'Đường dẫn đến các tập dữ liệu mẫu tf.Example .\
                           Có thể bao gồm các ký tự đại diện để truy cập nhiều tập dữ liệu.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Đường dẫn đên file text vocabulary.')
tf.app.flags.DEFINE_string('log_root', 'log', 'Đường dẫn đến thư mục root cho tất cả các logging.')
tf.app.flags.DEFINE_string('pretrain_dis_data_path', '', ' Đường dẫn dữ liệu cho bộ pre-train discriminator')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of pretrain/train/decode')
tf.app.flags.DEFINE_boolean('single_pass', False,
                            'Dùng cho chế độ Decode.\
                            nếu True, Thực hiện đánh giá trên toàn dataset sử dụng checkpoint cố định, \
                            hay checkpoint hiện tại và dùng mô hình tại checkpoint đó để tạo một bản tóm tắt cho mỗi ví dụ\
                            trong tập dữ liệu, Sau đó ghi các bản tóm tắt vào file và thực hiện đánh giá ROUGE cho toàn bộ tập dữ liệu.\
                            Nếu Sai (mặc định), thực hiện giải mã đồng thời, tức là tải liên tục checkpoint mới nhất, sử dụng nó\
                            để tạo tóm tắt cho các ví dụ được chọn ngẫu nhiên và show kết quả ra màn hình ')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'Số chiều của tầng ẩn')

tf.app.flags.DEFINE_integer('emb_dim', 128, 'Số chiều của word embeddings')

tf.app.flags.DEFINE_integer('batch_size', 32, 'minibatch size')

tf.app.flags.DEFINE_integer('dis_batch_size', 128, 'batch size cho pretrain discriminator') #256

tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'thời gian tối đa của encoder (max source text tokens)')

tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'thời gian tối đa của decoder (max summary tokens)')

tf.app.flags.DEFINE_integer('beam_size', 4, 'Số lượng từ tìm kiếm bằng beam search')

tf.app.flags.DEFINE_integer('min_dec_steps', 35,
                            'Độ dài ngắn nhất của một sequence được tạo. chỉ áp dụng cho chế độ giả mã từ beam search ')

tf.app.flags.DEFINE_integer('vocab_size', 50000,
                            'Kích thước của bộ từ vựng. Sẽ được đọc theo thứ tự các từ bên trong file \
                             nếu set vocab_size= 0, Thì nó sẽ đọc hết các từ của toàn tập dữ liệu.')

tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')

tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'độ lớn cho các ô LSTM được khởi tạo đồng nhất ngẫu nhiên')

tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std của trunc norm init, được sử dụng để khởi tạo mọi thứ khác')

tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'dùng cho gradient clipping')

tf.app.flags.DEFINE_integer('rollout', 24, 'Kích thước của số rollout')

# Pointer-generator  model
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'Sủ dụng pointer-generator model, ngược lại sử dụng baseline model.')
# Baseline model model
tf.app.flags.DEFINE_boolean('seqgan', True, 'nếu False Vô hiệu hóa seqgan')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False,
                            'Sử dụng  như một cơ chế bao phủ. Tắt chức năng này trong quá trình train ban đầu \
                            bật trong một đoạn ngắn ở cuối.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight của coverage loss. không khuyến khích set cov_loss_wt = 0 khi  \
                           minimal coverage loss.')

# Utility flags, Dùng để khôi phục và thay đổi các điểm kiểm tra
tf.app.flags.DEFINE_boolean('restore_best_model', False,
                            'Phục hồi mô hình tốt nhất được lưu trong\
                             thư mục train thường dùng trong early stoping')

tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")
FLAGS = tf.app.flags.FLAGS


def prepare_hps():
    hparam_list = ['mode', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps',
                   'coverage', 'cov_loss_wt', 'pointer_gen', 'seqgan', 'rollout', 'lr',
                   'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm']
    hps_dict = {}
    for key,val in FLAGS.flag_values_dict().items():
        if key in hparam_list:
            hps_dict[key] = val
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    return hps


def restore_best_model():
    # load file best-model từ thư mục eval, thêm các biến cho adagrad và lưu vào thư mục huấn luyện
    print("Phục hồi mô hình huấn luyện tốt nhất...")

    # Khởi tạo tất cả các vars trong mô hình.
    sess = tf.Session(config=util.get_config())
    print("Initializing all variables...")
    sess.run(tf.global_variables_initializer())

    # Phục hồi mô hình tốt nhất từ thư moục eval
    saver = tf.compat.v1.train.Saver([v for v in tf.global_variables()
                            if "Adagrad" not in v.name and 'discriminator' not in v.name and 'beta' not in v.name])
    print("Phục hồi tất cả các biến non-adagrad từ mô hình tốt nhất trong trong thư mục eval...")
    curr_ckpt = util.load_ckpt(saver, sess, "train")
    print("Restored %s." % curr_ckpt)

    # Lưu model vào thu mục train dir và thoát
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
    print("Đang lưu mô hình vào {}...".format(new_fname))

    # Lưu tât cả các biến đang tồn tại bao gồm cả Adagrad variables
    new_saver = tf.compat.v1.train.Saver()
    new_saver.save(sess, new_fname)
    print("Saved.")


def build_seqgan_graph(hps, vocab):
    print('Xây dựng biểu đồ cho generator...')
    with tf.device('/gpu:0'):
        generator = Generator(hps, vocab)
        generator.build_graph()

    print('Xây dựng biểu đồ cho  discriminator...')
    with tf.device('/gpu:0'):
        # TODO: Settings in args
        dis_filter_sizes = [2, 3, 4, 5]
        dis_num_filters = [100, 100, 100, 100]
        dis_l2_reg_lambda = 0.2
        discriminator = Discriminator(sequence_length=hps.max_dec_steps,
                                      num_classes=2,
                                      vocab_size=FLAGS.vocab_size,
                                      embedding_size=hps.emb_dim,
                                      filter_sizes=dis_filter_sizes,
                                      num_filters=dis_num_filters,
                                      pretrained_path=False,
                                      l2_reg_lambda=dis_l2_reg_lambda)
    return generator, discriminator


def setup_training(mode, generator, discriminator, generator_batcher, discriminator_batcher):
    train_dir = os.path.join(FLAGS.log_root, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if FLAGS.restore_best_model:
        restore_best_model()
        return

    # Giữ tối đa 3 checkpoints tại cùng một thời điểm
    saver = tf.compat.v1.train.Saver(max_to_keep=3)  
    supervisor = tf.train.Supervisor(logdir=train_dir,
                                     is_chief=True,
                                     saver=saver,
                                     summary_op=None,
                                     save_summaries_secs=60,  # Lưu các tóm tắt cho  tensorboard mỗi 60s
                                     save_model_secs=60,  # Tạo checkpoint mỗi 60 secs
                                     global_step=generator.global_step)

    summary_writer = supervisor.summary_writer
    sess_context_manager = supervisor.prepare_or_wait_for_session(config=util.get_config())

    try:
        if mode == "pretrain":
            trainer.pretrain_generator(generator, generator_batcher, summary_writer, sess_context_manager)
        elif mode == "train":
            trainer.pretrain_discriminator(discriminator, sess_context_manager)
            trainer.adversarial_train(generator, discriminator, generator_batcher, discriminator_batcher,
                                      summary_writer, sess_context_manager)
        else:
            raise ValueError("Đã bắt được các giá trị không hợp lệ của mô hình!")
    except KeyboardInterrupt:
        Print("Phát hiện lỗi bàn phím gây gián đoạn trên worker. Đang dừng giám sát ....")
        supervisor.stop()


def main(args):
    # In một thông báo nếu bạn nhập flags không chính xác
    if len(args) != 1:
        raise Exception("Problem with flags: %s" % args)

    # Nếu ở chế độ giải mã (decode), hãy set batch_size = beam_size 
    # Vì decode chỉ có thể giải quyệt một ví dụ (bài báo) tại một thời điểm
    if FLAGS.mode == 'decode':
        FLAGS.batch_size = FLAGS.beam_size

    # nếu single_pass=True, thì mô hình đang ở chế độ decode
    if FLAGS.single_pass and FLAGS.mode != 'decode':
        raise Exception("Bắt buộc khai báo single_pass = True nếu sử dụng chế độ decode")
    hps = prepare_hps()
    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)
    generator_batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)
    discriminator_batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

    if hps.mode == "pretrain" or hps.mode == "train":
        generator, discriminator = build_seqgan_graph(hps, vocab)
        setup_training(hps.mode, generator, discriminator, generator_batcher, discriminator_batcher)
    elif hps.mode == 'decode':

        # Mô hình được định cấu hình với max_dec_steps = 1 vì decoder chỉ chạy một bước mỗi lần (phục vụ beam search).
        decode_model_hps = hps._replace(max_dec_steps=1)
        generator = Generator(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(generator, generator_batcher, vocab)
        decoder.decode()
    else:
        raise ValueError("The 'mode' flag phải thuộc một trong các pretrain/train/decode")


if __name__ == '__main__':
     tf.compat.v1.app.run()
