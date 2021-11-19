import ast
import codecs
import numpy as np
import os
import time

from bs4 import BeautifulSoup
from tokenizer import tokenize
from typing import Callable, Dict, List, Tuple, Union
from utils import encoding, info, path_color


class InvertedIndex:
    __index: Dict[str, np.ndarray]

    def __init__(self, disk_path: str, files_path: str, load_from_disk: bool, get_max_file_count: Callable[[], int]):
        self.__index = {}
        self.__article_table: Dict[int, Tuple[str, str, int]] = {}
        self.__average_word_count: float = 0
        self.__total_word_count: int = 0
        self.__index_path: str = disk_path + '/inverted_index.npy'
        self.__article_table_path: str = disk_path + '/article_table.txt'
        self.__average_word_count_path: str = disk_path + '/average_word_count.txt'
        self.__index_restored: bool = False

        index_exists: bool = os.path.exists(self.__index_path)
        article_table_exists: bool = os.path.exists(self.__article_table_path)
        average_word_count_exists: bool = os.path.exists(self.__average_word_count_path)
        if not load_from_disk or not index_exists or not article_table_exists or not average_word_count_exists:
            print(f'Indexing files from {path_color(files_path)}')
            self.__parse_files(files_path, get_max_file_count())
            self.__save_to_disk()
            return

        print(f'Restoring index from {path_color(disk_path)}')

        self.__index = np.load(self.__index_path, allow_pickle=True).item()
        with codecs.open(self.__article_table_path, 'r', encoding) as f:
            table = f.read()
            self.__article_table = set() if table == str(set()) else ast.literal_eval(table)
        with codecs.open(self.__average_word_count_path, 'r', encoding) as f:
            value = f.read()
            self.__average_word_count = ast.literal_eval(value)

        print(
            f'Loaded index of {info(self.get_article_count())} articles with {info(len(self.__index))} tokens'
        )
        self.__index_restored = True

    def get_average_word_count(self) -> int:
        return self.__average_word_count

    def get_entries_for_token(self, token: str) -> np.ndarray:
        if token in self.__index:
            return self.__index[token]
        return np.empty(shape=(0, 2), dtype=np.uint32)

    def get_article_by_id(self, article_id: int) -> Union[Tuple[str, str, int], None]:
        if article_id in self.__article_table:
            return self.__article_table[article_id]
        return None

    def get_article_count(self) -> int:
        return len(self.__article_table)

    def __parse_files(self, path: str, max_files: int):
        if self.__index_restored:
            return
        start_time: float = time.time()
        file_entries: List[str] = os.listdir(path)
        if max_files >= 0:
            file_entries = file_entries[:max_files]
        file_paths: List[str] = list(map(lambda entry: path + '/' + entry, file_entries))
        file_count: int = len(file_paths)
        for index, file_path in enumerate(file_paths):
            self.__parse_file(file_path, index, file_count)
        self.__average_word_count = self.__total_word_count / len(self.__article_table)
        end_time: float = time.time()
        print(
            f'Indexed {info(self.get_article_count())} article(s) of {info(file_count)} file(s) in {info(round(end_time - start_time, 2))} seconds '
        )

    def __parse_file(self, file_path: str, current: int, total: int):
        with open(file_path, encoding='utf-8') as file:
            soup: BeautifulSoup = BeautifulSoup(file.read(), 'html.parser')
            start_time: float = time.time()
            for article in soup.find_all('article'):
                self.__parse_article(article, file.name)
            end_time: float = time.time()
            print(f'{path_color(current + 1)}/{path_color(total)}: Indexing {path_color(file.name)} took {info(round(end_time - start_time, 2))} seconds')

    def __parse_article(self, article, file_name: str):
        article.find('revision').decompose()  # remove revision tag
        article_id: int = int(article.find('id').string)  # get article id
        article_title: str = str(article.find('title').string)  # get article id

        article_content: str = ' '.join([article_title, _get_categories(article), _get_body(article)])
        article_tokens: List[str] = tokenize(article_content)

        token_occurrences: Dict[str, int] = {}
        for token in article_tokens:
            if token not in token_occurrences:
                token_occurrences[token] = 0
            token_occurrences[token] += 1

        for token, frequency in token_occurrences.items():
            if token not in self.__index:
                self.__index[token] = np.empty(shape=(0, 2), dtype=np.uint32)
            self.__index[token] = np.vstack((self.__index[token], [article_id, frequency]))

        self.__article_table[article_id] = (article_title, file_name, len(article_tokens))
        self.__total_word_count += len(article_tokens)

    def __save_to_disk(self):
        os.makedirs(os.path.dirname(self.__article_table_path), exist_ok=True)
        np.save(self.__index_path, self.__index)
        with codecs.open(self.__article_table_path, 'w+', encoding) as f:
            f.write(str(self.__article_table))
        with codecs.open(self.__average_word_count_path, 'w+', encoding) as f:
            f.write(str(self.__average_word_count))


def _get_categories(article) -> str:
    return ' '.join(map(lambda category: str(category.string), article.find_all('category')))


def _get_body(article) -> str:
    article_body = article.find('bdy')
    if article_body is None:
        return ''
    return str(article_body.string)
