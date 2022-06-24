import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2


SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
#vocab id
PAD_TOKEN = '[PAD]'
#ký tự ko rõ
UNKNOWN_TOKEN = '[UNK]' 
# Bắt đầu môi câu đầu vào decoder
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'

class Vocab(object):
    def __init__(self, vocab_file, max_size):

        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # Biến đếm tự điển

        # [UNK], [PAD], [START] và [STOP] tương ứng với ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Đọc vocab file và thêm các từ lên đến max_size
        with open(vocab_file, 'r', encoding='utf-8') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print('Cảnh báo: Sai format tại dòng của file vocabulary: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Phát hiện từ trừng trong file vocabulary: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print("Kích thước tối đã của bộ từ diển đã được chỉ định là %i; Ta có %i từ. Dừng đọc."
                          % (max_size, self._count))
                    break

        print("Xây dựng bộ từ điển hoàn tất of %i tổng số từ. Từ cuối đã được thêm vào: %s"
              % (self._count, self._id_to_word[self._count - 1]))

    def word2id(self, word):
        # Trả về id (int) của một từ (string). 
        # Trả về  [UNK] id nếu một từ là OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        # Trả về từ (string) từ một id (int)
        if word_id not in self._id_to_word:
            raise ValueError('Không tìm thấy Id (%d) trong bộ từ điển ' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        #Trả về Tổng số từ của bộ từ điển
        return self._count

    def write_metadata(self, fpath):
        print("Đang ghi word embedding metadata file vào %s..." % (fpath))
        with open(fpath, "w") as f:
             fieldnames = ['word']
             writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
             for i in range(self.size()):
                 writer.writerow({"word": self._id_to_word[i]})


def example_generator(data_path, single_pass):
    while True:
         # Lấy danh sách các tệp tin
        filelist = glob.glob(data_path) 
        assert filelist, ('Error: Empty filelist tại %s' % data_path)  # check filelist isn't empty
        if single_pass:
            filelist = sorted(filelist)
        else:
            random.shuffle(filelist)
        for f in filelist:
            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes: break  # finished reading this file
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                yield example_pb2.Example.FromString(example_str)
        if single_pass:
            print("example_generator đã hoàn thành, đang đọc tất cả các tệp.")
            break


def article2ids(article_words, vocab):
    """Map các từ trong bài viết với id của chúng. Đồng thời trả về danh sách các OOV trong bài viết.

    Mô tả đầu vào:
      article_words: Danh sách các từ (strings)
      vocab: Vocabulary object"""

    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError(
                    'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                    i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def abstract2sents(abstract):
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p + len(SENTENCE_START):end_p])
        except ValueError as e:  # no more sentences
            return sents


def show_art_oovs(article, vocab):
    """Returns the article string, highlighting the OOVs by placing __underscores__ around them"""
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w) == unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        # w is oov
        if vocab.word2id(w) == unk_token:
            if article_oovs is None:  # baseline mode
                new_words.append("__%s__" % w)
            else:  # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else:  # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str
