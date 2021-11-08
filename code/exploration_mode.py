import time

from bs4 import BeautifulSoup
from inverted_index import InvertedIndex
from prompt_toolkit.validation import Validator, ValidationError
from PyInquirer import prompt
from query import query
from scoring import scoring_modes
from utils import clear_console, info


def run_exploration_mode(index: InvertedIndex):
    while True:
        answers = prompt(_questions)
        query_string = answers.get('query')
        scoring_mode = answers.get('scoring_mode')
        _run_query(index, query_string, scoring_mode)
        print()
        post_answers = prompt(_post_run_questions)
        if not post_answers.get('run_again'):
            return
        clear_console()
        print()


def _run_query(index: InvertedIndex, query_string: str, scoring_mode: str):
    query_start_time = time.time_ns()
    results = query(index, query_string, scoring_mode)
    query_end_time = time.time_ns()
    query_duration = (query_end_time - query_start_time) / 1000000.0

    return_choice = '   Return'
    choices = [return_choice]
    for rank, (article_id, article_score) in enumerate(results):
        article_title = index.get_article_by_id(str(article_id))[0]
        choices.append(f'{rank + 1}. {article_title} ({round(article_score, 4)}) [{article_id}]')
    print(
        f'\n{info(scoring_mode)} query for "{info(query_string)}" took {info(str(round(query_duration, 10)))} milliseconds\n')
    questions = [
        {
            'type': 'list',
            'name': 'selection',
            'message': 'Select result',
            'choices': choices
        },
    ]
    if len(choices) is 0:
        print('No results')
        return
    answers = prompt(questions)
    selection = answers.get('selection')
    if selection is return_choice:
        return
    article_id = selection[selection.rfind('[') + 1: selection.rfind(']')]
    print_article_content(index, article_id)


def print_article_content(index: InvertedIndex, article_id: str):
    article = index.get_article_by_id(article_id)
    article_title = article[0]
    article_path = article[1]
    with open(article_path, encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
        article_tag = soup.find('id', text=article_id).parent.parent
        print(info(f'\n--- {article_title} ---'))
        print(article_tag.find('bdy').string)
        print(info('---------'))


class QueryValidator(Validator):
    def validate(self, document):
        try:
            if not str(document.text):
                raise ValidationError(message='Query can not be empty',
                                      cursor_position=len(document.text))
        except ValueError:
            raise ValidationError(message='Please enter a query',
                                  cursor_position=len(document.text))


_questions = [
    {
        'type': 'list',
        'name': 'scoring_mode',
        'message': 'Select scoring function',
        'choices': scoring_modes
    },
    {
        'type': 'input',
        'name': 'query',
        'message': 'Enter the search query',
        'validate': QueryValidator,
        'filter': lambda val: str(val)
    },
]

_post_run_questions = [
    {
        'type': 'confirm',
        'name': 'run_again',
        'message': 'Run another query',
        'default': False,
    },
]
