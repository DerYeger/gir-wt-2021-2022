import time

from inverted_index import InvertedIndex
from query import query

_index_dir: str = './tables'
_dataset_dir: str = './dataset/articles'


def map_dict(f, dic: dict) -> dict:
    return dict(zip(dic, map(f, dic.values())))


def get_index(load_from_disk: bool, max_files) -> InvertedIndex:
    return InvertedIndex(_index_dir, _dataset_dir, max_files, load_from_disk=load_from_disk)
