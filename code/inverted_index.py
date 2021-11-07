import ast
import codecs
import os
import time

from bs4 import BeautifulSoup
from tokenizer import tokenize

encoding = 'utf_16'


class InvertedIndex:
    def __init__(self, disk_path, files_path, max_files, load_from_disk):
        self.index = {}
        self.article_table = {}
        self.average_word_count = 0
        self.total_word_count = 0
        self.disk_path = disk_path
        self.index_path = disk_path + '/inverted_index.txt'
        self.article_table_path = disk_path + '/article_table.txt'
        self.average_word_count_path = disk_path + '/average_word_count.txt'
        self.index_restored = False

        index_exists = os.path.exists(self.index_path)
        article_table_exists = os.path.exists(self.article_table_path)
        average_word_count_exists = os.path.exists(self.average_word_count_path)
        if not load_from_disk or not index_exists or not article_table_exists or not average_word_count_exists:
            print(f'Indexing files from {files_path}')
            self.__parse_files(files_path, max_files)
            self.__save_to_disk()
            return

        print(f'Restoring index from {disk_path}')
        with codecs.open(self.index_path, 'r', encoding) as f:
            table = f.read()
            self.index = set() if table == str(set()) else ast.literal_eval(table)
        with codecs.open(self.article_table_path, 'r', encoding) as f:
            table = f.read()
            self.article_table = set() if table == str(set()) else ast.literal_eval(table)
        with codecs.open(self.average_word_count_path, 'r', encoding) as f:
            value = f.read()
            self.average_word_count = ast.literal_eval(value)

        print(f'Loaded index of {self.get_article_count()} articles with {len(self.index)} tokens')
        self.index_restored = True

    def __save_to_disk(self):
        os.makedirs(os.path.dirname(self.article_table_path), exist_ok=True)
        with codecs.open(self.index_path, 'w+', encoding) as f:
            f.write(str(self.index))
        with codecs.open(self.article_table_path, 'w+', encoding) as f:
            f.write(str(self.article_table))
        with codecs.open(self.average_word_count_path, 'w+', encoding) as f:
            f.write(str(self.average_word_count))

    def __parse_article(self, article, file_name, file_index):
        article.find('revision').decompose()  # remove revision tag
        article_id: str = article.find('id').string  # get article id
        article_title: str = article.find('title').string  # get article id

        article_content = ' '.join([article_title, get_categories(article), get_body(article)])
        article_tokens = tokenize(article_content)

        article_tokens: [str] = [article_id, *article_tokens]

        token_occurrences = {}
        for token in article_tokens:
            if token not in token_occurrences:
                token_occurrences[token] = 0
            token_occurrences[token] += 1

        for token, frequency in token_occurrences.items():
            if token not in self.index:
                self.index[token] = []
            self.index[token].append((int(article_id), frequency))

        self.article_table[article_id] = [article_title, file_name, file_index, len(article_tokens)]
        self.total_word_count += len(article_tokens)

    def __parse_file(self, file):
        soup = BeautifulSoup(file.read(), 'html.parser')
        for article in soup.find_all('article'):
            self.__parse_article(article, file.name, soup.index(article))

    def __parse_files(self, path, max_files):
        if self.index_restored:
            return
        file_entry: str
        files_read = 0
        last_time = time.time()
        for file_entry in os.listdir(path):
            if max_files == files_read:
                return
            print(f'--- File {file_entry} ---')
            with open(path + '/' + file_entry, encoding='utf8') as file:
                self.__parse_file(file)

            curr_time = time.time()
            print(f'--- {curr_time - last_time} seconds for file #{files_read + 1} ---')
            last_time = time.time()
            files_read += 1
            self.average_word_count = self.total_word_count / len(self.article_table)

    def get_entries_for_token(self, token: str) -> [(int, int)]:
        if token in self.index:
            return self.index[token]
        return []

    def get_article_by_id(self, article_id: str):
        if article_id in self.article_table:
            return self.article_table[article_id]
        return None

    def get_article_count(self):
        return len(self.article_table)


def get_categories(article) -> str:
    return ' '.join(map(lambda category: category.string, article.find_all('category')))


def get_body(article) -> str:
    article_body = article.find('bdy')
    if article_body is None:
        return ''
    return article_body.string
