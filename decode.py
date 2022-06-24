import logging, os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import json
import time
import pyrouge
import tensorflow as tf
import beam_search
import data
import util

FLAGS = tf.app.flags.FLAGS

#Số giây tối đa đến khi load check point mới
SECS_UNTIL_NEW_CKPT = 60  


class BeamSearchDecoder(object):
    """Beam search decoder."""

    def __init__(self, model, batcher, vocab):
        #Khởi tạo các tham số ban đầu:
        self._model = model
        self._model.build_graph()
        self._batcher = batcher
        self._vocab = vocab

        # load lại các checkpoints cho quá trình decoding
        self._saver = tf.train.Saver()  
        self._sess = tf.Session(config=util.get_config())

        # Khởi tạo checkpoint nếu chưa tồn tại (train lần đầu)
        ckpt_path = util.load_ckpt(self._saver, self._sess)

        if FLAGS.single_pass:
            # Make a descriptive decode directory name

            ckpt_name = "ckpt-" + ckpt_path.split('-')[-1]  # Ex: "ckpt-696969"
            cur_time = str(time.time())
            self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name) + '_' + cur_time)
            if os.path.exists(self._decode_dir):
                raise Exception(" Vui lòng không thư mục single_pass decode %s trước đó" % self._decode_dir)

        # join vào thư mục decode
        else:  
            self._decode_dir = os.path.join(FLAGS.log_root, "decode")

        # Tạo thư mục decode (nếu cần thiết)
        if not os.path.exists(self._decode_dir):
            os.mkdir(self._decode_dir)

        # Tạo các thư mục để chứa đầu ra (được viết theo định dạng đầu vào của thư viện pyrouge)
        if FLAGS.single_pass:

            #Thư mục chứa các tập tin vắng (input tập test)
            self._rouge_article_dir = os.path.join(self._decode_dir, "Story_news")
            if not os.path.exists(self._rouge_article_dir): os.mkdir(self._rouge_article_dir)

            #Thư mục chứa các bản tóm tắt do con người tạo ra
            self._rouge_ref_dir = os.path.join(self._decode_dir, "References")
            if not os.path.exists(self._rouge_ref_dir): os.mkdir(self._rouge_ref_dir)

            #Thư mục chứa các bản tóm tắt do mô hình GAN tạo ra
            self._rouge_dec_dir = os.path.join(self._decode_dir, "GAN_Generated")
            if not os.path.exists(self._rouge_dec_dir): os.mkdir(self._rouge_dec_dir)


    def decode(self):
        """Decode tất cả các tin vắng trong tập test  
            (nếu FLAGS.single pass) và trả về, hoặc gặp lỗi giải mã vô thời hạn, 
                load lại checkpoint cuối cùng gần nhất."""
         
        t0 = time.time()
        counter = 0
        while True:
            # Lặp lại   ví dụ qua các batch
            batch = self._batcher.next_batch()

            # Hoàn thành quá trình decoding tập dữ trên chế độ single_pass
            if batch is None:  
                assert FLAGS.single_pass, "Đã duyệt hết dataset và thoát khỏi chế độ single_pass."
                print("Decoder đã đọc xong Dataset cho single_pass.")
                print("Output đã được lưu trong", self._rouge_ref_dir, "và",self._rouge_dec_dir," Bắt đầu quá trình đánh giá ROUGE...")
                results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
                rouge_log(results_dict, self._decode_dir)
                return

            original_article = batch.original_articles[0] # string
            original_abstract = batch.original_abstracts[0]  # string
            original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings
            article_withunks = data.show_art_oovs(original_article, self._vocab)  # string
            abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab,
                                                   (batch.art_oovs[0] if FLAGS.pointer_gen else None))  # string

            #Thực hiện beam search để có được giả thuyết(Hypothesis) tối ưu nhất
            best_hyp = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)

            #Trích xuất các đầu ra ids từ các hypothesis và chuyển đổi lại thành  các từ (words)
            output_ids = [int(t) for t in best_hyp.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self._vocab,
                                                 (batch.art_oovs[0] if FLAGS.pointer_gen else None))

            #Xóa các stop word từ decoded_words (nếu cần thiết)
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            for i in range(len(decoded_words)):
                if not type(decoded_words[i]) is str:
                    decoded_words[i] = str(decoded_words[i], encoding='utf-8')

            decoded_output = " ".join(decoded_words)

            # ghi bản tóm tắt tham chiếu tham khảo và bản tóm tắt được tạo vào file 
            # ( dùng cho việc đánh giá bằng pyrouge sau này)
            if FLAGS.single_pass:
                self.write_for_rouge(original_abstract_sents, decoded_words, original_article,counter)

                #Đếm số bản tóm tắt đã được tạo
                counter += 1  
            else:
                # log các kết quả ra màn hình
                print_results(article_withunks, abstract_withunks, decoded_output)

                # Ghi thông tin vào file .json cho visualization tool
                self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, 
                                        best_hyp.attn_dists, best_hyp.p_gens)

                #Kiểm tra SECS_UNTIL_NEW_CKPT đã duyệt chưa; nếu đã qua, ta sẽ load một checkpoint mới
                t1 = time.time()
                if t1 - t0 > SECS_UNTIL_NEW_CKPT:
                    print("Mô hình đã giải mã tại cùng một checkpoint trong",t1 - t0,"giây, đến lúc load một checkpoint mới")
                    _ = util.load_ckpt(self._saver, self._sess)
                    t0 = time.time()

    def write_for_rouge(self, reference_sents, decoded_words, article, ex_index):
        """ Thự hiện ghi các kết quả theo format pyrouge (chế độ single_pass )

        Mô tả dữ liệu:
          reference_sents: list (strings)
          decoded_words: list (strings)
          ex_index: int, Sô chỉ mục cho tên các tệp
        """
        # Đầu tiên, chia các tóm tắt thành các câu
        decoded_sents = []
        while len(decoded_words) > 0:
            try:
                fst_period_idx = decoded_words.index(".")

             # there is text remaining that doesn't end in "."   
            except ValueError:  
                fst_period_idx = len(decoded_words)
            sent = decoded_words[:fst_period_idx + 1]  # sentence up to and including the period
            decoded_words = decoded_words[fst_period_idx + 1:]  # everything else
            decoded_sents.append(' '.join(sent))

        # Thư viện pyrouge gọi một perl script để chuyển dữ liệu vào trong các files HTML.
        decoded_sents = [make_html_safe(w) for w in decoded_sents]
        reference_sents = [make_html_safe(w) for w in reference_sents]

        # Ghi vào file
        ref_file = os.path.join(self._rouge_ref_dir, "%06d_Reference.txt" % ex_index)
        decoded_file = os.path.join(self._rouge_dec_dir, "%06d_GAN_Generated.txt" % ex_index)
        article_file = os.path.join(self._rouge_article_dir, "%06d_article.txt" % ex_index)
        t0 = time.time()
        with open(ref_file, "w", encoding='utf-8') as f:
            for idx, sent in enumerate(reference_sents):
                f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
        with open(decoded_file, "w", encoding='utf-8') as f:
            for idx, sent in enumerate(decoded_sents):
                f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")
        with open(article_file, "w", encoding='utf-8') as f:
            f.write(article)
        print("Đã ghi ví dụ %i vào 3 file trong log" % ex_index,"~", time.time() - t0," giây")

    def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
        """
        Mô tả các tham số:
          article: Bài viết gốc (string)
          abstract: bản tóm tắt đúng (người) (string)
          attn_dists: List của các arrays; Phân bố các attention.
          decoded_words: List của các strings; các từ của các bản tóm tắt được tạo bởi GAN
        """
        article_lst = article.split()  # list of words
        decoded_lst = decoded_words  # list of decoded words
        to_write = {
            'article_lst': [make_html_safe(t) for t in article_lst],
            'GAN_Generated_lst': [make_html_safe(t) for t in decoded_lst],
            'abstract_str': make_html_safe(abstract),
            'attn_dists': attn_dists
        }
        if FLAGS.pointer_gen:
            to_write['p_gens'] = p_gens
        output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
        with open(output_fname, 'w', encoding='utf-8') as output_file:
            json.dump(to_write, output_file)
        print('Đã ghi dữ liệu minh họa vào',output_fname)

def print_results(article, abstract, decoded_output):
    """Prints the article, the reference summmary and the decoded summary to screen"""
    print("")
    print('CÁC BÀI VIẾT TIN TỨC:', article,  '\n')
    print('BẢN TÓM TẮT THAM KHẢO (NGƯỜI):' ,abstract,  '\n')
    print('BẢN TÓM TẮT TẠO BỞI MÔ HÌNH GAN:', decoded_output,  '\n')
    print("")

#Thay thế bất kỳ dấu ngoặc nhọn nào trong chuỗi s để tránh ảnh hưởng đến HTML attention visualizer
def make_html_safe(s):
    
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s

#Đánh giá cả file trong ref_dir và dec_dir với pyrouge, trả về results_dict
def rouge_eval(ref_dir, dec_dir):
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_Reference.txt'
    r.system_filename_pattern = '(\d+)_GAN_Generated.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir

    # silence pyrouge logging
    logging.getLogger('global').setLevel(logging.WARNING)  
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    """Show kết quả ROUGE ra màn hình và lưu vào file

    Mô tả các biến:
      results_dict: Dictionary được trả về từ pyrouge
      dir_to_write: Dictionary, Dùng để ghi kết quả đã trả về"""
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f với khoảng tin cậy (confidence interval) (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)

    print(log_str)  
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print("Đang ghi kết quả ROUGE vào file", results_file)
    with open(results_file, "w", encoding='utf-8') as f:
        f.write(log_str)


def get_decode_dir_name(ckpt_name):
    #Đặt tên mô tả cho thư mực decode, Bao gồm tên của checkpoint dùng để decode

    if "train" in FLAGS.data_path:
        dataset = "train"
    elif "val" in FLAGS.data_path:
        dataset = "val"
    elif "test" in FLAGS.data_path:
        dataset = "test"
    else:
        raise ValueError("FLAGS.data_path %s nên chứa một trong các thư mục train, val hoặc test" % (FLAGS.data_path))
    dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (
    dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
    if ckpt_name is not None:
        dirname += "_%s" % ckpt_name
    return dirname
