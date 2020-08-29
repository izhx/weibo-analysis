"""
微博情感分析，LDA主题聚类
"""

import os
import re
# import time
import logging
from codecs import open
from argparse import ArgumentParser
from multiprocessing import Pool, Manager

from numpy import zeros
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore
from pandas import DataFrame, Series, read_excel, concat

from util import LTP_DIR, output, generate_batch, to_excel, cache_path, dump_cache, load_cache
from workers import skep_consumer, skep_producer, ltp_tokenzier


with open(f"{LTP_DIR}/vocab.txt") as file:
    CHAR = set(w.strip() for w in file.readlines())
    CHAR = set(w for w in CHAR if len(w) == 1)

with open("./stopwords.txt") as file:
    STOP_WORDS = set(w.strip() for w in file.readlines()) | CHAR

FORWARD_SPLIT = re.compile(r"//@[^/:：]+[:：]")
FORWARD_CONTENT = re.compile(r"//@[^/:：]+[:：][^/]+")
URL_REGEX = re.compile(r"http[s]?://[a-zA-Z0-9.?/&=:]*")

_ARG_PARSER = ArgumentParser(description="我的实验，需要指定配置文件")
_ARG_PARSER.add_argument('--name', '-n',
                         type=str,
                         default='clean',
                         help='configuration file path.')
_ARG_PARSER.add_argument('--ltpIDS', '-l',
                         type=str,
                         default='6,7',
                         help='gpu ids, like: 1,2,3')
_ARG_PARSER.add_argument('--skepIDS', '-s',
                         type=str,
                         default='1,2,3,4,5',
                         help='gpu ids, like: 1,2,3')
_ARG_PARSER.add_argument('--range', '-r',
                         type=str,
                         default='5,21',
                         help='话题数搜索范围，左闭右开')
_ARG_PARSER.add_argument('--passes', '-p',
                         type=int,
                         default=50,
                         help='数据集迭代次数, epoch')
_ARG_PARSER.add_argument('--iterations', '-it',
                         type=int,
                         default=50,
                         help='推断时最大迭代次数')
_ARG_PARSER.add_argument('--keywords_num', '-k',
                         type=int,
                         default=50,
                         help='存储关键词数量')
_ARG_PARSER.add_argument('--pool_size', '-ps',
                         type=int,
                         default=16,
                         help='进程池大小')
_ARG_PARSER.add_argument('--debug', '-d', type=bool, default=False)
_ARGS = _ARG_PARSER.parse_args()

os.environ['OMP_NUM_THREADS'] = '1'

# logging.basicConfig(format='[%(asctime)s - %(levelname)s] : %(message)s', level=logging.DEBUG)


def read(path) -> DataFrame:
    def _clean(row):
        text = URL_REGEX.sub('', row.contents)
        if row.is_forward and '//@' in text:
            # 如果是转发的且格式正确
            if text.startswith('//@'):
                # 如果单纯转发，则内容设置为最原始微博的内容
                try:
                    text = FORWARD_CONTENT.findall(text)[-1]
                    i = FORWARD_SPLIT.match(text).regs[0][1]
                    text = text[i:]
                except IndexError:
                    text = text.replace('//@', '')  # TODO 可以用weibo的API处理
            else:
                # 否则截取新内容
                text = text[:text.find('//@')]
        return text

    temp_name = os.path.basename(path).replace('.xlsx', '')
    if os.path.isfile(cache_path(temp_name)):
        data, texts = load_cache(temp_name)
    else:
        output(f"===> Reading from <{path}>.")
        data: DataFrame = read_excel(path)  # .iloc[:280]

        # 只保留想要的4列，并去除空值，截取日期
        data = data[['contents', 'time', 'id', 'is_forward']].dropna().reset_index()
        data['date'] = data['time'].apply(lambda s: s[:10])
        data['contents'] = data['contents'].astype(str)

        # 预处理文本
        texts = data.apply(_clean, axis=1).to_list()
        dump_cache((data, texts), temp_name)
    output(f"===> got {len(data)} rows from <{path}>.")

    # 解析GPU ID
    ltp_ids = [i.strip() for i in _ARGS.ltpIDS.split(',')]
    skep_ids = [i.strip() for i in _ARGS.skepIDS.split(',')]

    # 初始化进程池，管理器，数据队列
    pool = Pool(1 + len(ltp_ids) + len(skep_ids))  # 分别分词、获取skep输入、skep运算
    manager = Manager()
    feqture_queue = manager.Queue(16 * len(skep_ids))
    result_queue = manager.Queue(16 * len(skep_ids))

    # 异步任务启动
    pool.apply_async(skep_producer, (feqture_queue, texts, 16, len(skep_ids)))
    tokens = dict()
    for i, (s, p) in zip(ltp_ids, generate_batch(texts, len(texts) // len(ltp_ids) + 1)):
        tokens[(s.start, s.stop)] = pool.apply_async(ltp_tokenzier, (p, 192, i))
    for i in skep_ids:
        pool.apply_async(skep_consumer, (feqture_queue, result_queue, i))

    # 接收结果
    scores, counter = zeros(len(texts)), 1
    while True:
        _slice, array = result_queue.get()
        # print(_slice)
        if array is None:
            if counter < len(skep_ids):
                counter += 1
            else:
                break
        else:
            scores[_slice] = array

    data['tokens'] = None
    for s, t in tokens.items():
        data['tokens'].update(Series(t.get(), range(*s)))
    data['sentiment_score'] = scores
    pool.close()
    pool.join()
    return data[['date', 'tokens', 'id', 'sentiment_score']]


def save_and_inference(model: LdaModel, corpus, num_topics):
    path = f"./dev/model/{_ARGS.name}_{num_topics}.pkl"
    try:
        model.save(path)
        output(f"model saved at <{path}>")
        gamma, _ = model.inference(corpus)
    except RuntimeError as e:
        logging.error(f"PID: {os.getpid()}, num_topics: {num_topics} error")
        print(e)
    output(f"num_topics {num_topics} inference compete.")
    return gamma.argmax(axis=1)


def get_model(corpus, num_topics, kwargs):
    output(f"running num_topics: {num_topics}.")
    try:
        model = LdaModel(corpus, num_topics, **kwargs)
        topic_ids = save_and_inference(model, corpus, num_topics)
    except RuntimeError as e:
        logging.error(f"PID: {os.getpid()}, num_topics: {num_topics} error")
        print(e)
    return model, topic_ids


def pipline(data: DataFrame):
    documents = data['tokens'].to_list()
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(documents)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    # 去停用词
    bad_ids = [dictionary.token2id[t] for t in STOP_WORDS if t in dictionary.token2id]
    dictionary.filter_tokens(bad_ids=bad_ids)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    _ = dictionary[0]  # This is only to "load" the dictionary.
    output('Number of unique tokens: ', len(dictionary))
    output('Number of documents: ', len(corpus))
    # test = get_model(6, corpus, dictionary.id2token)

    topic_range = tuple(int(s.strip()) for s in _ARGS.range.split(','))
    kwargs = dict(
        id2word=dictionary.id2token, chunksize=len(corpus),
        passes=_ARGS.passes, alpha='auto', eta='auto', eval_every=None,
        iterations=_ARGS.iterations, random_state=123)
    result_dict = dict()
    if len(corpus) < 5e5:  # 并行训练模型
        pool = Pool(_ARGS.pool_size)
        for k in range(*topic_range):
            result_dict[k] = pool.apply_async(get_model, (corpus, k, kwargs))
        result_dict = {k: v.get() for k, v in result_dict.items()}
    else:
        pool = Pool(1)
        kwargs['alpha'] = 'symmetric'
        for k in range(*topic_range):
            model = LdaMulticore(corpus, k, workers=_ARGS.pool_size, **kwargs)
            ids = pool.apply_async(save_and_inference, (model, corpus, k))
            result_dict[k] = [model, ids]
        result_dict = {k: (v[0], v[1].get()) for k, v in result_dict.items()}
    pool.close()  # 等子进程执行完毕后关闭进程池
    pool.join()

    output(f"Searched range{topic_range}")

    # 计算一致性的代码自己有多进程，所以只能串行
    for k, (model, ids) in result_dict.items():
        print('\nnum_topics: ', k)
        # print('Model perplexity: ', model.log_perplexity(corpus))  # 这个没做归一
        top_topics = model.top_topics(corpus, documents, dictionary,
                                      coherence='c_v', topn=_ARGS.keywords_num,
                                      processes=_ARGS.pool_size)
        scores = Series([t[1] for t in top_topics])
        print('Coherence Score: ', scores.mean())
        # 得到关键词词频
        topics_info = list()
        for _topic in top_topics:
            tokens = [(t[1], t[0], dictionary.cfs[dictionary.token2id[t[1]]]) for t in _topic[0]]
            topics_info.append((tokens, _topic[1]))
        to_excel(topics_info, data, ids, _ARGS.name)

    output(f"===> {_ARGS.name} compete. \n")


def main():
    if os.path.isfile(cache_path(_ARGS.name)):
        df = load_cache(_ARGS.name)
    else:
        if _ARGS.name == 'clean':
            dfs = list()
            for i in range(6):
                path = f"./dev/data/clean{i}_covid19.xlsx"
                if os.path.isfile(cache_path(f'clean{i}')):
                    part = load_cache(f'clean{i}')
                else:
                    part = read(path)
                    dump_cache(part, f'clean{i}')
                dfs.append(part)
            df = concat(dfs, ignore_index=True)
        else:
            path = f"./dev/data/{_ARGS.name}_covid19.xlsx"
            df = read(path)

        dump_cache(df, _ARGS.name)
        logging.disable(level=logging.INFO)

    pipline(df)
    return


if __name__ == "__main__":
    main()

"""
统计每天每主题数量，每天（每主题）感情变化
之前331k要处理7个小时，2张卡；现在20min，7张卡。

"""
