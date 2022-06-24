import logging, os 
# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import queue
import time
from random import shuffle
from threading import Thread

import numpy as np
import tensorflow as tf

import data


class Example(object):

    def __init__(self, article, abstract_sentences, vocab, hps):

        self.hps = hps

        # Lấy ids của ký tự đặc biệt
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        # Xử lý bài viết
        article_words = article.split()
        if len(article_words) > hps.max_enc_steps:
            article_words = article_words[:hps.max_enc_steps]
        self.enc_len = len(article_words)  # store the length after truncation but before padding
        self.enc_input = [vocab.word2id(w) for w in
                          article_words]  # list of word ids; OOVs are represented by the id for UNK token

        # Xử lý abstract
        abstract = ' '.join(abstract_sentences)  # string
        abstract_words = abstract.split()  # list of strings
        abs_ids = [vocab.word2id(w) for w in
                   abstract_words]  # list của word ids; OOVs được tham chiếu bởi id cho UNK token

        # Lấy the decoder input sequence và target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, hps.max_dec_steps, start_decoding,
                                                                 stop_decoding)
        self.dec_len = len(self.dec_input)

        # Nếu sử dụng chế độ pointer-generator mode, chúng ta cần lưu một số thông tin
        if hps.pointer_gen:
            # Lưu một phiên bản của enc_input trong đó các OOV trong bài viết được thể hiện bằng id OOV tạm thời của chúng; cũng tự lưu trữ các từ OOVs trong bài viết
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)

            # Xem phần tóm tắt tham chiếu trong đó các OOV trong bài viết được thể hiện bằng id OOV bài viết tạm thời của chúng
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

            # Ghi đè chuỗi mục tiêu của decoder để nó sử dụng id OOV bài viết tạm thời
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, hps.max_dec_steps, start_decoding,
                                                        stop_decoding)

        # Lưuu các thông tin gốc
        self.original_article = article
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if self.hps.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


class Batch(object):

    def __init__(self, example_list, hps, vocab):

        self.pad_id = vocab.word2id(data.PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list, hps)  # initialize the input to the encoder
        self.init_decoder_seq(example_list, hps)  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list, hps):

        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Khởi tạo the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)
        self.enc_weight_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_weight_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if hps.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list, hps):
        """Initializes the following:
            self.dec_batch:
              numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.
            self.target_batch:
              numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.
            self.dec_padding_mask:
              numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
            """
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
        self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)
        self.batch_reward = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)
        self.batch_rouge_reward = np.zeros((hps.batch_size, 1), dtype=np.float32)


        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1


    def store_orig_strings(self, example_list):
        """Store the original article and abstract strings in the Batch object"""
        self.original_articles = [ex.original_article for ex in example_list]  # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list]  # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list]  # list of list of lists


class Batcher(object):
    """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

    def __init__(self, data_path, vocab, hps, single_pass, BATCH_QUEUE_MAX=100):
        """Initialize the batcher. Start threads that process the data into batches.

        Args:
          data_path: tf.Example filepattern.
          vocab: Vocabulary object
          hps: hyperparameters
          single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set). Otherwise generate random batches indefinitely (useful for training).
        """
        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._single_pass = single_pass

        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = queue.Queue(BATCH_QUEUE_MAX)
        self._example_queue = queue.Queue(BATCH_QUEUE_MAX * self._hps.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = 1  # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 16  # num threads to fill example queue
            self._num_batch_q_threads = 4  # num threads to fill batch queue
            self._bucketing_cache_size = 100  # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:  # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        """Return a Batch from the batch queue.

        If mode='decode' then each batch contains a single example repeated beam_size-many times; this is necessary for beam search.

        Returns:
          batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
        """
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.compat.v1.logging.warning(
                'Bucket input queue rỗng khi gọi next_batch. Bucket queue kích thước: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                tf.logging.info("Hoàn tất việc đọc dataset trong chế độ  single_pass .")
                return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_example_queue(self):
        """Reads data from file and processes into Examples which are then placed into the example queue."""

        input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

        while True:
            try:
                # read the next example from file. article and abstract are both strings.
                (article, abstract) = next(input_gen)
            except StopIteration:  # if there are no more examples:
                tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
                if self._single_pass:
                    tf.logging.info(
                        "single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")
            # Use the <s> and </s> tags in abstract to get a list of sentences.
            article = str(article, encoding='utf-8')
            abstract = str(abstract, encoding='utf-8')
            abstract_sentences = [sent.strip() for sent in data.abstract2sents(str(abstract))]
            example = Example(article, abstract_sentences, self._vocab, self._hps)  # Process into an Example.
            self._example_queue.put(example)  # place the Example in the example queue.

    def fill_batch_queue(self):
        """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.

        In decode mode, makes batches that each contain a single example repeated.
        """
        while True:
            if self._hps.mode != 'decode':
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self._hps.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_len)  # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self._hps.batch_size):
                    batches.append(inputs[i:i + self._hps.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(Batch(b, self._hps, self._vocab))

            else:  # beam search decode mode
                ex = self._example_queue.get()
                b = [ex for _ in range(self._hps.batch_size)]
                self._batch_queue.put(Batch(b, self._hps, self._vocab))

    def watch_threads(self):
        """Watch example queue and batch queue threads and restart if dead."""
        while True:
            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def text_generator(self, example_generator):
        # Tạo article and abstract text from tf.Example.
        while True:
            e = next(example_generator) # e is a tf.Example
            try:
                # Bài viarticle' in the data files
                article_text = e.features.feature['article'].bytes_list.value[0]  
                # The abstract text was saved under the key 'abstract' in the data files
                abstract_text = e.features.feature['abstract'].bytes_list.value[0]  #
            except ValueError:
                tf.logging.error('Failed to get article or abstract from example')
                continue
            if len(article_text) == 0:
                tf.compat.v1.logging.warning('Đã tìm thấy một ví dụ rỗng, Bỏ qua.')
            else:
                yield (article_text, abstract_text)
