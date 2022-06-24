import numpy as np
import tensorflow as tf

import data

FLAGS = tf.app.flags.FLAGS


class Hypothesis(object):
    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.p_gens = p_gens
        self.coverage = coverage

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
        return Hypothesis(tokens=self.tokens + [token],
                          log_probs=self.log_probs + [log_prob],
                          state=state,
                          attn_dists=self.attn_dists + [attn_dist],
                          p_gens=self.p_gens + [p_gen],
                          coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.log_prob / len(self.tokens)

    @property
    def current_sentences(self):
        return self.tokens

    def change_lastest_token(self, stop_token):
        self.tokens = self.tokens[:-1] + [stop_token]


def run_beam_search(sess, model, vocab, batch):

    enc_states, dec_in_state = model.run_encoder(sess, batch)

    hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                       log_probs=[0.0],
                       state=dec_in_state,
                       attn_dists=[],
                       p_gens=[],
                       coverage=np.zeros([batch.enc_batch.shape[1]])
                       ) for _ in range(FLAGS.beam_size)]
    results = []
    steps = 0
    while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
        latest_tokens = [h.latest_token for h in hyps]
        latest_tokens = [t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in
                         latest_tokens] 
        states = [h.state for h in hyps]
        prev_coverage = [h.coverage for h in hyps]

        (topk_ids, topk_log_probs, new_states,
         attn_dists, p_gens, new_coverage) = model.decode_onestep(sess=sess, batch=batch,
                                                                  latest_tokens=latest_tokens,
                                                                  enc_states=enc_states,
                                                                  dec_init_states=states,
                                                                  prev_coverage=prev_coverage)
        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(hyps) 
        for i in range(num_orig_hyps):
            h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], \
                                                             new_coverage[i]
            for j in range(FLAGS.beam_size * 2):
                new_hyp = h.extend(token=topk_ids[i, j],
                                   log_prob=topk_log_probs[i, j],
                                   state=new_state,
                                   attn_dist=attn_dist,
                                   p_gen=p_gen,
                                   coverage=new_coverage_i)
                all_hyps.append(new_hyp)

        hyps = []  
        cnt_trigram = 0
        for h in sort_hyps(all_hyps):  
            if h.latest_token == vocab.word2id(data.STOP_DECODING):

                if steps >= FLAGS.min_dec_steps:
                    results.append(h)
            else:  
                pre_trigram = [(h.tokens[i:i + 3]) for i in range(len(h.tokens) - 3)]
                lastest_trigram = h.tokens[(-1 * 3):]
                if lastest_trigram in pre_trigram:
                    cnt_trigram += 1
                    if cnt_trigram < len(all_hyps) - FLAGS.beam_size:

                        continue
                    else:
                        print("trigram reach max repeated time!")
                        h.change_lastest_token(vocab.word2id(data.STOP_DECODING))
                hyps.append(h)
            if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
                
                break
        steps += 1
    if len(results) == 0:
        results = hyps
    hyps_sorted = sort_hyps(results)

    return hyps_sorted[0]


def sort_hyps(hyps):
    # Trả về một list của các  Hypothesis objects, đã được sắp xếp giảm dần theo xác suất log trung bình
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
