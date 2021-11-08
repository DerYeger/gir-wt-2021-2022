import time

from inverted_index import InvertedIndex
from query import query

_index_dir: str = './tables'
_dataset_dir: str = './dataset/articles'


def map_dict(f, dic: dict) -> dict:
    return dict(zip(dic, map(f, dic.values())))


def get_index(load_from_disk: bool) -> InvertedIndex:
    return InvertedIndex(_index_dir, _dataset_dir, max_files=1, load_from_disk=load_from_disk)


if __name__ == '__main__':
    inverted_index = get_index(load_from_disk=True)

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
