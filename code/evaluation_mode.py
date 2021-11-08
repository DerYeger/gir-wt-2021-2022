"""
This file contains your code to generate the evaluation files that are input to the trec_eval algorithm.
"""

import codecs
import os
import subprocess
from createindex import get_index
from inverted_index import InvertedIndex
from query import query
from topic import parse_topics_file, Topic

encoding = 'utf_16'


def prepare_results_files(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with codecs.open(file_path, 'w+', encoding) as file:
        file.truncate(0)


def evaluate_topic(index: InvertedIndex, topic: Topic, eval_type: str, result_file):
    results = query(index, topic.query, eval_type, silent=True)
    for rank, result in enumerate(results):
        result_file.write(f'{topic.topic_id} Q0 {result[0]} {rank + 1} {result[1]} {eval_type}\n')


def evaluate_topics(topics_file_path: str, results_dir: str):
    bm25_path = f'{results_dir}/bm25-evaluation.txt'
    prepare_results_files(bm25_path)
    tf_idf_path = f'{results_dir}/tf-idf-evaluation.txt'
    prepare_results_files(tf_idf_path)
    with codecs.open(bm25_path, 'a+', encoding) as bm25_results_file, codecs.open(tf_idf_path, 'a+', encoding) as tf_idf_results_file:
        topics = parse_topics_file(topics_file_path)
        print(f'Loaded {len(topics)} topics')
        index = get_index(load_from_disk=True)
        for topic in topics:
            evaluate_topic(index, topic, 'bm25', bm25_results_file)
            evaluate_topic(index, topic, 'tf-idf', tf_idf_results_file)
    print(f'Evaluated {len(topics)} topics')
    run_trec_eval(bm25_path)
    run_trec_eval(tf_idf_path)


def run_trec_eval(result_file_path: str, qrel_file_path: str = './wiki_files/dataset/eval.qrels'):
    command = f'trec_eval -m map -m ndcg_cut.10 -m P.10 -m recall.10 {qrel_file_path} {result_file_path}'
    result = subprocess.check_output(command.split(' '))
    print(result)


if __name__ == '__main__':
    evaluate_topics('./wiki_files/dataset/topics.xml', './retrieval_results')

