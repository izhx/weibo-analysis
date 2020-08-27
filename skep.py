"""
skep附加代码，为了batch
"""

import os

from multiprocessing import Process, Queue

from util import output, generate_batch, SKEP_DIR


def _skep_feature(q, texts, batch_size):
    from paddlehub.reader.batching import pad_batch_data
    from paddlehub.reader.tokenization import convert_to_unicode, FullTokenizer

    def skep_batch(texts, tokenizer, max_seq_len=512):
        texts = [convert_to_unicode(t) for t in texts]
        tokens = [tokenizer.tokenize(t) for t in texts]
        # Account for [CLS] and [SEP] with "- 2"
        max_token = max_seq_len - 2
        tokens = [t[:max_token] if len(t) > max_token else t for t in tokens]
        # add [CLS] and [SEP], and convert to index
        tokens = [["[CLS]"] + t + ["[SEP]"] for t in tokens]
        token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        # get additional ids
        position_ids = [list(range(len(t))) for t in token_ids]
        text_type_ids = [[0] * len(t) for t in token_ids]
        task_ids = [[0] * len(t) for t in token_ids]

        pad_id = tokenizer.vocab["[PAD]"]
        padded_token_ids, input_mask = pad_batch_data(token_ids,
                                                      max_seq_len=max_seq_len,
                                                      pad_idx=pad_id,
                                                      return_input_mask=True)
        padded_text_type_ids = pad_batch_data(text_type_ids,
                                              max_seq_len=max_seq_len,
                                              pad_idx=pad_id)
        padded_position_ids = pad_batch_data(position_ids,
                                             max_seq_len=max_seq_len,
                                             pad_idx=pad_id)
        padded_task_ids = pad_batch_data(task_ids,
                                         max_seq_len=max_seq_len,
                                         pad_idx=pad_id)

        feature = [
            padded_token_ids, padded_position_ids, padded_text_type_ids,
            input_mask, padded_task_ids
        ]
        return feature

    tokenizer = FullTokenizer(
        os.path.join(SKEP_DIR, "assets/ernie_1.0_large_ch.vocab.txt"))

    output("SKEP feature transform started.")
    for batch in generate_batch(texts, batch_size):
        feature = skep_batch(batch, tokenizer)
        q.put(feature)
    output("SKEP feature compete.")


def skep_analysis(texts, batch_size=16):
    import numpy as np
    import paddlehub
    from paddle.fluid.core import PaddleTensor

    # 子进程处理特征
    q = Queue()
    p = Process(target=_skep_feature, args=(q, texts, batch_size))
    p.start()

    # 默认使用os.environ["CUDA_VISIBLE_DEVICES"]的第一个GPU
    skep = paddlehub.Module(directory=SKEP_DIR)

    scores, i = np.zeros(len(texts)), 0
    output(f"batch size {batch_size}, sentiment analysis started...")
    while not q.empty():
        feature = q.get()
        inputs = [PaddleTensor(ndarray) for ndarray in feature]
        tensor = skep.predictor.run(inputs)[0]
        probs = np.array(tensor.data.float_data()).reshape(tensor.shape)
        scores[i: i + tensor.shape[0]] = probs[:, 1] - probs[:, 0]
        i += tensor.shape[0]
    p.join()
    output("sentiment analysis compete.")
    return scores
