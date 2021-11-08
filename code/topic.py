from bs4 import BeautifulSoup


class Topic:
    def __init__(self, topic_id, title, phrase_title, description, narrative):
        self.topic_id = topic_id
        self.title = title
        self.phrase_title = phrase_title
        self.description = description
        self.narrative = narrative
        parts = [self.title, self.phrase_title, self.description, self.narrative]
        self.query = ' '.join(filter(None, parts))


def _parse_topic(topic_tag) -> Topic:
    topic_id = topic_tag.attrs['id']
    title = topic_tag.find('title').string
    phrase_title = topic_tag.find('phrasetitle').string
    description = topic_tag.find('description').string
    narrative = topic_tag.find('narrative').string
    return Topic(topic_id, title, phrase_title, description, narrative)


def parse_topics_file(file_path) -> [Topic]:
    with open(file_path, encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
        return list(map(_parse_topic, soup.find_all('topic')))
