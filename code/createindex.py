"""
This file contains your code to create the inverted index.
Besides implementing and using the predefined tokenization function (text2tokens),
there are no restrictions in how you organize this file.
"""

import os
from bs4 import BeautifulSoup
import time
import numpy as np
import ast
import codecs
from tokenizer import tokenize


# Settings
encoding = 'utf_16'

inverted_index = {}
article_table = {}  # {article_id: [title, path, offset, word_count]}
total_word_count = 0
avg_word_count = 0  # average document length in words

actual_dir: str = './wiki_files/dataset/articles/'
test_dir: str = './wiki_files/test/'
current_dir: str = actual_dir

max_files: int = 1


def map_dict(f, dic: dict) -> dict:
    return dict(zip(dic, map(f, dic.values())))


def save_tables():
    os.makedirs(os.path.dirname('./tables/'), exist_ok=True)
    with codecs.open('./tables/inverted_index_table.txt', 'w+', encoding) as f:
        f.write(str(inverted_index))
    with codecs.open('./tables/article_table.txt', 'w+', encoding) as f:
        f.write(str(article_table))
    with codecs.open('./tables/avg_dl.txt', 'w+', encoding) as f:
        f.write(str(avg_word_count))


def load_tables() -> bool:
    global inverted_index, article_table, avg_word_count

    if not os.path.exists('./tables/inverted_index_table.txt') or not os.path.exists(
            './tables/article_table.txt') or not os.path.exists('./tables/avg_dl.txt'):
        return False

    with codecs.open('./tables/inverted_index_table.txt', 'r', encoding) as f:
        table = f.read()
        inverted_index = set() if table == str(set()) else ast.literal_eval(table)
    with codecs.open('./tables/article_table.txt', 'r', encoding) as f:
        table = f.read()
        article_table = set() if table == str(set()) else ast.literal_eval(table)
    with codecs.open('./tables/avg_dl.txt', 'r', encoding) as f:
        value = f.read()
        avg_word_count = ast.literal_eval(value)
    return True


def load_wiki_files():
    file_entry: str
    files_read = 0
    last_time = time.time()
    for file_entry in os.listdir(current_dir):
        if max_files == files_read:
            return
        print(file_entry)
        with open(current_dir + file_entry, encoding='utf8') as file:
            eval_wiki_data(file)

        curr_time = time.time()
        print("--- %s seconds for file number %d---" % (curr_time - last_time, files_read + 1))
        last_time = time.time()
        files_read += 1


def insert_index(article_id: str, token: str, frequency: int):
    if token not in inverted_index:
        inverted_index[token] = []
    inverted_index[token].append((int(article_id), frequency))


def eval_wiki_data(file):
    global total_word_count
    global avg_word_count
    soup = BeautifulSoup(file.read(), 'html.parser')
    for article in soup.find_all('article'):
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
            insert_index(article_id, token, frequency)

        article_table[article_id] = [article_title, file.name, soup.index(article), len(article_tokens)]
        total_word_count += len(article_tokens)
    avg_word_count = total_word_count / len(article_table)


def get_categories(article) -> str:
    return ' '.join(map(lambda category: category.string, article.find_all('category')))


def get_body(article) -> str:
    article_body = article.find('bdy')
    if article_body is None:
        return ''
    return article_body.string


def query(query_string: str, eval_type: str):
    query_tokens = tokenize(query_string)
    scores = tf_idf(query_tokens) if eval_type == 'tf-idf' else bm25(query_tokens)
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rank = 0
    for article_id, article_score in sorted_results:
        rank += 1
        article_title = article_table[str(article_id)][0]
        print("#{} is article {} with score {} and title {}".format(rank, article_id, article_score, article_title))
        if rank >= 100:
            return


def bm25(query_tokens: [str]):
    article_scores = {}
    for token in query_tokens:
        token_idf = idf(token)
        for (article_id, frequency) in inverted_index[token]:
            id_string = str(article_id)
            if id_string not in article_scores:
                article_scores[id_string] = 0
            k = (1.2 + 2) / 2
            b = 0.75
            dl = article_table[id_string][3]
            nominator = frequency * (k + 1)
            denominator = frequency + k * (1 - b + (b * (dl / avg_word_count)))
            article_scores[id_string] += token_idf * (nominator / denominator)
            article_scores[id_string] += token_idf * frequency
    return article_scores


def idf(token):
    N = len(article_table)
    n_qi = len(inverted_index[token])
    nominator = N - n_qi + 0.5
    denominator = n_qi + 0.5
    return np.log((nominator / denominator) + 1)


def tf_idf(query_tokens: [str]):
    article_scores = {}
    for token in query_tokens:
        token_idf = idf(token)
        for (article_id, frequency) in inverted_index[token]:
            id_string = str(article_id)
            if id_string not in article_scores:
                article_scores[id_string] = 0
            article_scores[id_string] += token_idf * frequency
    return article_scores


if __name__ == '__main__':
    forceReIndex = True
    if forceReIndex or not load_tables():
        start_time = time.time()
        load_wiki_files()
        print("re-indexing took --- %s seconds ---" % (time.time() - start_time))
        save_tables()
    else:
        print("indexes loaded from memory")

    print("\n--- tf-idf ---")
    query_start_time = time.time_ns()
    query("Freestyle", 'tf-idf')
    query_end_time = time.time_ns()
    print("--- Query took %s milliseconds ---" % ((query_end_time - query_start_time) / 1000000.0))

    print("\n--- bm25 ---")
    query_start_time = time.time_ns()
    query("Freestyle", 'bm-25')
    query_end_time = time.time_ns()
    print("--- Query took %s milliseconds ---" % ((query_end_time - query_start_time) / 1000000.0))

# todo
#   implement evaluation_mode and exploration_mode
#   index all files
#   make text content of articles accessible
