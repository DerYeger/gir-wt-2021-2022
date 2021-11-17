from bs4 import BeautifulSoup


class Topic:
    def __init__(self, topic_id: str, title: str, phrase_title: str, description: str, narrative: str):
        self.topic_id = topic_id
        self.title = title
        self.phrase_title = phrase_title
        self.description = description
        self.narrative = narrative
        parts: list[str] = [self.title]  # , self.phrase_title, self.description, self.narrative]
        self.query = ' '.join(filter(None, parts))


def _parse_topic(topic_tag) -> Topic:
    topic_id = str(topic_tag.attrs['id'])
    title = str(topic_tag.find('title').string)
    phrase_title = str(topic_tag.find('phrasetitle').string)
    description = str(topic_tag.find('description').string)
    narrative = str(topic_tag.find('narrative').string)
    return Topic(topic_id, title, phrase_title, description, narrative)


def parse_topics_file(file_path) -> list[Topic]:
    with open(file_path, encoding='utf-8') as file:
        soup: BeautifulSoup = BeautifulSoup(file.read(), 'html.parser')
        return list(map(_parse_topic, soup.find_all('topic')))
