import ast
import codecs
import os
import time

from bs4 import BeautifulSoup
from tokenizer import tokenize
from utils import encoding, info, path_color


class InvertedIndex:
    def __init__(self, disk_path, files_path, max_files, load_from_disk):
        self.__index = {}
        self.__article_table = {}
        self.__average_word_count = 0
        self.__total_word_count = 0
        self.__disk_path = disk_path
        self.__index_path = disk_path + '/inverted_index.txt'
        self.__article_table_path = disk_path + '/article_table.txt'
        self.__average_word_count_path = disk_path + '/average_word_count.txt'
        self.__index_restored = False

        index_exists = os.path.exists(self.__index_path)
        article_table_exists = os.path.exists(self.__article_table_path)
        average_word_count_exists = os.path.exists(self.__average_word_count_path)
        if not load_from_disk or not index_exists or not article_table_exists or not average_word_count_exists:
            print(f'Indexing files from {path_color(files_path)}')
            self.__parse_files(files_path, max_files)
            self.__save_to_disk()
            return

        print(f'Restoring index from {path_color(disk_path)}')
        with codecs.open(self.__index_path, 'r', encoding) as f:
            table = f.read()
            self.__index = set() if table == str(set()) else ast.literal_eval(table)
        with codecs.open(self.__article_table_path, 'r', encoding) as f:
            table = f.read()
            self.__article_table = set() if table == str(set()) else ast.literal_eval(table)
        with codecs.open(self.__average_word_count_path, 'r', encoding) as f:
            value = f.read()
            self.__average_word_count = ast.literal_eval(value)

        print(f'Loaded index of {info(str(self.get_article_count()))} articles with {info(str(len(self.__index)))} tokens')
        self.__index_restored = True

    def get_average_word_count(self) -> int:
        return self.__average_word_count

    def get_entries_for_token(self, token: str) -> [(int, int)]:
        if token in self.__index:
            return self.__index[token]
        return []

    def get_article_by_id(self, article_id: str):
        if article_id in self.__article_table:
            return self.__article_table[article_id]
        return None

    def get_article_count(self):
        return len(self.__article_table)

    def __parse_files(self, path, max_files):
        if self.__index_restored:
            return
        file_entry: str
        files_read = 0
        last_time = time.time()
        for file_entry in os.listdir(path):
            if max_files == files_read:
                return
            with open(path + '/' + file_entry, encoding='utf-8') as file:
                self.__parse_file(file)

            curr_time = time.time()
            print(f'{path_color(file_entry)} took {info(str(round(curr_time - last_time, 2)))} seconds')
            last_time = time.time()
            files_read += 1
            self.__average_word_count = self.__total_word_count / len(self.__article_table)

    def __parse_file(self, file):
        soup = BeautifulSoup(file.read(), 'html.parser')
        for article in soup.find_all('article'):
            self.__parse_article(article, file.name)

    def __parse_article(self, article, file_name):
        article.find('revision').decompose()  # remove revision tag
        article_id: str = article.find('id').string  # get article id
        article_title: str = article.find('title').string  # get article id

        article_content = ' '.join([article_title, _get_categories(article), _get_body(article)])
        article_tokens = tokenize(article_content)

        article_tokens: [str] = [article_id, *article_tokens]

        token_occurrences = {}
        for token in article_tokens:
            if token not in token_occurrences:
                token_occurrences[token] = 0
            token_occurrences[token] += 1

        for token, frequency in token_occurrences.items():
            if token not in self.__index:
                self.__index[token] = []
            self.__index[token].append((int(article_id), frequency))

        self.__article_table[article_id] = [article_title, file_name, len(article_tokens)]
        self.__total_word_count += len(article_tokens)

    def __save_to_disk(self):
        os.makedirs(os.path.dirname(self.__article_table_path), exist_ok=True)
        with codecs.open(self.__index_path, 'w+', encoding) as f:
            f.write(str(self.__index))
        with codecs.open(self.__article_table_path, 'w+', encoding) as f:
            f.write(str(self.__article_table))
        with codecs.open(self.__average_word_count_path, 'w+', encoding) as f:
            f.write(str(self.__average_word_count))


def _get_categories(article) -> str:
    return ' '.join(map(lambda category: category.string, article.find_all('category')))


def _get_body(article) -> str:
    article_body = article.find('bdy')
    if article_body is None:
        return ''
    return article_body.string
