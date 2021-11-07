"""
This file contains your code to create the inverted index.
Besides implementing and using the predefined tokenization function (text2tokens),
there are no restrictions in how you organize this file.
"""

import time
from inverted_index import InvertedIndex
from scoring import bm25, tf_idf
from tokenizer import tokenize

actual_dir: str = './wiki_files/dataset/articles/'
test_dir: str = './wiki_files/test/'
current_dir: str = actual_dir


def map_dict(f, dic: dict) -> dict:
    return dict(zip(dic, map(f, dic.values())))


def query(index: InvertedIndex, query_string: str, eval_type: str):
    query_tokens = tokenize(query_string)
    scores = tf_idf(index, query_tokens) if eval_type == 'tf-idf' else bm25(index, query_tokens)
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rank = 0
    for article_id, article_score in sorted_results:
        rank += 1
        article_title = index.get_article_by_id(str(article_id))[0]
        print(f'#{rank} is article {article_id} with score {article_score} and title {article_title}')
        if rank >= 100:
            return


def get_index(load_from_disk: bool) -> InvertedIndex:
    return InvertedIndex('./tables', current_dir, max_files=1, load_from_disk=load_from_disk)


if __name__ == '__main__':
    inverted_index = get_index(load_from_disk=True)
    inverted_index.save_to_disk()

    print('\n--- tf-idf ---')
    query_start_time = time.time_ns()
    query(inverted_index, 'Freestyle', 'tf-idf')
    query_end_time = time.time_ns()
    query_duration = (query_end_time - query_start_time) / 1000000.0
    print(f'--- Query took {query_duration} milliseconds ---')

    print('\n--- bm25 ---')
    query_start_time = time.time_ns()
    query(inverted_index, 'Freestyle', 'bm-25')
    query_end_time = time.time_ns()
    query_duration = (query_end_time - query_start_time) / 1000000.0
    print(f'--- Query took {query_duration} milliseconds ---')

# todo
#   implement evaluation_mode and exploration_mode
#   index all files
#   make text content of articles accessible
