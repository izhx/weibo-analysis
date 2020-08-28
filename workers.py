"""
skep生产者消费者
"""

import os
# import time
from queue import Queue
from typing import List

from util import output, generate_batch, SKEP_DIR, LTP_DIR


def skep_producer(queue: Queue, texts: List[str], batch_size: int,
                  consumer_num: int):
    from paddlehub.reader.batching import pad_batch_data
    from paddlehub.reader.tokenization import convert_to_unicode, FullTokenizer

    def skep_batch(texts: List[str], max_seq_len=512):
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
        padded_token_ids, input_mask = pad_batch_data(
            token_ids, pad_id, max_seq_len, return_input_mask=True)
        padded_text_type_ids = pad_batch_data(text_type_ids, pad_id,
                                              max_seq_len)
        padded_position_ids = pad_batch_data(position_ids, pad_id, max_seq_len)
        padded_task_ids = pad_batch_data(task_ids, pad_id, max_seq_len)

        feature = [
            padded_token_ids, padded_position_ids, padded_text_type_ids,
            input_mask, padded_task_ids
        ]
        return feature

    tokenizer = FullTokenizer(
        os.path.join(SKEP_DIR, "assets/ernie_1.0_large_ch.vocab.txt"))

    output("SKEP feature transform started.")
    for _slice, batch in generate_batch(texts, batch_size):
        feature = skep_batch(batch)
        queue.put((_slice, feature))
    else:
        for _ in range(consumer_num):
            queue.put((None, None))  # queue.empty()不可靠，手动发送结束信号
        queue.close()
    output("SKEP feature compete.")


def skep_consumer(feature_queue: Queue, result_queue: Queue, gpu_id='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    from numpy import array
    from paddle.fluid.core import PaddleTensor
    from paddlehub import Module

    skep = Module(directory=SKEP_DIR)

    output(f"SKEP at GPU-{gpu_id} start...")
    while True:
        _slice, feature = feature_queue.get()
        if feature is None:  # 接收到结束信号
            result_queue.put((None, None))
            output(f"consumer at GPU-{gpu_id} stop.")
            break
        inputs = [PaddleTensor(ndarray) for ndarray in feature]
        tensor = skep.predictor.run(inputs)[0]
        probs = array(tensor.data.float_data()).reshape(tensor.shape)
        scores = probs[:, 1] - probs[:, 0]
        result_queue.put((_slice, scores))

    return True


def ltp_tokenzier(texts, batch_size=128, gpu_id='1') -> List[List[str]]:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    import torch
    from ltp.ltp import LTP, WORD_MIDDLE, get_entities, convert_idx_to_name

    ltp = LTP(LTP_DIR, torch.device('cuda'))
    result = list()
    try:
        output(f"LTP at GPU-{gpu_id} started...")
        with torch.no_grad():
            for _, inputs in generate_batch(texts, batch_size):
                # t = time.time()
                tokenizerd = ltp.tokenizer.batch_encode_plus(
                    inputs,
                    padding=True,
                    truncation=True,
                    return_tensors=ltp.tensor,
                    max_length=ltp.max_length)
                input_ids = tokenizerd['input_ids'].to(ltp.device)
                attention_mask = tokenizerd['attention_mask'].to(ltp.device)
                token_type_ids = tokenizerd['token_type_ids'].to(ltp.device)
                lengths = torch.sum(attention_mask, dim=-1) - 2

                # Electra 最后一层输出
                model_output, *_ = ltp.model.pretrained(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
                # remove [CLS] [SEP]
                char_input = torch.narrow(model_output, 1, 1,
                                          model_output.size(1) - 2)
                # 得到分词预测结果
                segment_output = torch.argmax(
                    ltp.model.seg_decoder(char_input), dim=-1).cpu().numpy()
                # id转换为标签
                segment_output = convert_idx_to_name(segment_output, lengths,
                                                     ltp.seg_vocab)
                sentences = []

                for source_text, length, encoding, seg_tag in zip(
                        inputs, lengths, tokenizerd.encodings, segment_output):
                    words = encoding.words[1:length + 1]
                    offsets = encoding.offsets[1:length + 1]
                    text = [source_text[start:end] for start, end in offsets]

                    for idx in range(1, length):
                        current_beg = offsets[idx][0]
                        forward_end = offsets[idx - 1][-1]
                        if forward_end < current_beg:
                            text[idx] = source_text[forward_end:
                                                    current_beg] + text[idx]
                        if words[idx - 1] == words[idx]:
                            seg_tag[idx] = WORD_MIDDLE

                    entities = get_entities(seg_tag)
                    sentences.append([
                        ''.join(text[entity[1]:entity[2] + 1]).strip()
                        for entity in entities
                    ])

                result.extend(sentences)
                # tokens, _ = ltp.seg(batch)  # batch 128, 1.8s
                # output(time.time() - t)
        output("tokenize compete.")
    except RuntimeError as e:
        output('分词进程错误', e)
    return result
