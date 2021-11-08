import codecs
import os
import subprocess
from createindex import get_index
from inverted_index import InvertedIndex
from query import query
from topic import parse_topics_file, Topic

_encoding = 'utf_16'


def run_evaluation_mode():
    _evaluate_topics('./dataset/topics.xml', './retrieval_results')


def _evaluate_topics(topics_file_path: str, results_dir: str):
    bm25_path = f'{results_dir}/bm25-evaluation.txt'
    _prepare_results_files(bm25_path)
    tf_idf_path = f'{results_dir}/tf-idf-evaluation.txt'
    _prepare_results_files(tf_idf_path)
    with codecs.open(bm25_path, 'a+', _encoding) as bm25_results_file, \
            codecs.open(tf_idf_path, 'a+', _encoding) as tf_idf_results_file:
        topics = parse_topics_file(topics_file_path)
        print(f'Loaded {len(topics)} topics')
        index = get_index(load_from_disk=True)
        for topic in topics:
            _evaluate_topic(index, topic, 'bm25', bm25_results_file)
            _evaluate_topic(index, topic, 'tf-idf', tf_idf_results_file)
    print(f'Evaluated {len(topics)} topics')
    _run_trec_eval(bm25_path)
    _run_trec_eval(tf_idf_path)


def _prepare_results_files(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with codecs.open(file_path, 'w+', _encoding) as file:
        file.truncate(0)


def _evaluate_topic(index: InvertedIndex, topic: Topic, eval_type: str, result_file):
    results = query(index, topic.query, eval_type, silent=True)
    for rank, result in enumerate(results):
        result_file.write(f'{topic.topic_id} Q0 {result[0]} {rank + 1} {result[1]} {eval_type}\n')


def _run_trec_eval(result_file_path: str, qrel_file_path: str = './dataset/eval.qrels'):
    command = f'trec_eval -m map -m ndcg_cut.10 -m P.10 -m recall.10 {qrel_file_path} {result_file_path}'
    result = subprocess.check_output(command.split(' '))
    print(result)
