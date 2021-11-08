import codecs
import os
import subprocess

from createindex import get_index
from inverted_index import InvertedIndex
from query import query
from topic import parse_topics_file, Topic
from utils import error, info

_encoding = 'utf_16'


def run_evaluation_mode(index: InvertedIndex):
    _evaluate_topics(index, './dataset/topics.xml', './retrieval_results')


def _evaluate_topics(index: InvertedIndex, topics_file_path: str, results_dir: str):
    bm25_path = f'{results_dir}/bm25-evaluation.txt'
    _prepare_results_files(bm25_path)
    tf_idf_path = f'{results_dir}/tf-idf-evaluation.txt'
    _prepare_results_files(tf_idf_path)
    with codecs.open(bm25_path, 'a+', _encoding) as bm25_results_file, \
            codecs.open(tf_idf_path, 'a+', _encoding) as tf_idf_results_file:
        topics = parse_topics_file(topics_file_path)
        print(f'Loaded {info(str(len(topics)))} topics')
        for topic in topics:
            _evaluate_topic(index, topic, 'bm25', bm25_results_file)
            _evaluate_topic(index, topic, 'tf-idf', tf_idf_results_file)
    print(f'Evaluated {info(str(len(topics)))} topics\n')
    if _trec_eval_is_available():
        _run_trec_eval(bm25_path)
        _run_trec_eval(tf_idf_path)


def _prepare_results_files(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with codecs.open(file_path, 'w+', _encoding) as file:
        file.truncate(0)


def _evaluate_topic(index: InvertedIndex, topic: Topic, eval_type: str, result_file):
    results = query(index, topic.query, eval_type)
    for rank, result in enumerate(results):
        result_file.write(f'{topic.topic_id} Q0 {result[0]} {rank + 1} {result[1]} {eval_type}\n')


def _trec_eval_is_available() -> bool:
    try:
        subprocess.run(['trec_eval'])
        return True
    except FileNotFoundError:
        print(error('trec_eval not found'))
        return False


def _run_trec_eval(result_file_path: str, qrel_file_path: str = './dataset/eval.qrels'):
    command = f'trec_eval -m map -m ndcg_cut.10 -m P.10 -m recall.10 {qrel_file_path} {result_file_path}'
    result = subprocess.check_output(command.split(' '))
    print(result)
