"""
This file contains your code to generate the evaluation files that are input to the trec_eval algorithm.
"""

import codecs
import os
from createindex import get_index
from inverted_index import InvertedIndex
from query import query
from topic import parse_topics_file, Topic

encoding = 'utf_16'


def evaluate_topic(index: InvertedIndex, topic: Topic, eval_type: str, result_file):
    results = query(index, topic.query, eval_type, silent=True)
    for rank, result in enumerate(results):
        result_file.write(f'{topic.topic_id} Q0 {result[0]} {rank + 1} {result[1]} {eval_type}\n')


def evaluate_topics(topics_file_path: str, result_file_path: str):
    os.makedirs(os.path.dirname(result_file_path), exist_ok=True)
    with codecs.open(result_file_path, 'w+', encoding) as result_file:
        result_file.truncate(0)
    with codecs.open(result_file_path, 'a+', encoding) as result_file:
        topics = parse_topics_file(topics_file_path)
        print(f'Loaded {len(topics)} topics')
        index = get_index(load_from_disk=True)
        for topic in topics:
            print(f'Evaluating {topic.title}')
            evaluate_topic(index, topic, 'bm25', result_file)
            evaluate_topic(index, topic, 'tf-idf', result_file)


if __name__ == '__main__':
    evaluate_topics('./wiki_files/dataset/topics.xml', './evaluation/results.txt')
