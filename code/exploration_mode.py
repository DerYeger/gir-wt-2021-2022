import time

from inverted_index import InvertedIndex
from prompt_toolkit.validation import Validator, ValidationError
from PyInquirer import prompt
from query import query
from scoring import scoring_modes
from utils import highlight, info


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
        print()


def _run_query(index: InvertedIndex, query_string: str, scoring_mode: str):
    print(f'\n--- {info(scoring_mode)} results for "{info(query_string)}" ---')
    query_start_time = time.time_ns()
    results = query(index, query_string, scoring_mode)
    query_end_time = time.time_ns()
    query_duration = (query_end_time - query_start_time) / 1000000.0
    for rank, (article_id, article_score) in enumerate(results):
        article_title = index.get_article_by_id(str(article_id))[0]
        print(f'{highlight(f"#{rank + 1}")} is article {highlight(article_id)} with score {highlight(article_score)} and title {highlight(article_title)}')
    print(f'--- Query took {info(str(round(query_duration, 10)))} milliseconds ---')


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
