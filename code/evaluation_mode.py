"""
This file contains your code to generate the evaluation files that are input to the trec_eval algorithm.
"""

from createindex import get_index
from inverted_index import InvertedIndex
from query import query
from topic import parse_topics_file, Topic


def evaluate_topic(index: InvertedIndex, topic: Topic, eval_type: str):
    query(index, topic.query, eval_type, silent=True)


def evaluate_topics(topics_file_path: str):
    topics = parse_topics_file(topics_file_path)
    print(f'Loaded {len(topics)} topics')
    index = get_index(load_from_disk=True)
    for topic in topics:
        print(f'Evaluating {topic.title}')
        evaluate_topic(index, topic, 'bm25')
        evaluate_topic(index, topic, 'tf-idf')


if __name__ == '__main__':
    evaluate_topics('./wiki_files/dataset/topics.xml')
