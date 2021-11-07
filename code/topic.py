from bs4 import BeautifulSoup
from tokenizer import tokenize


class Topic:
    def __init__(self, title):
        self.title = title

    def tokenize(self) -> [str]:
        return list(tokenize(self.title))


def parse_topic(topic_tag) -> Topic:
    title = topic_tag.find('title').string
    print(title)
    return Topic(title)


def parse_topics_file(file_path) -> [Topic]:
    with open(file_path, encoding='utf8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
        return list(map(parse_topic, soup.find_all('topic')))


if __name__ == '__main__':
    print(parse_topics_file('./wiki_files/dataset/topics.xml'))
