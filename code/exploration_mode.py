import time
from createindex import get_index
from inverted_index import InvertedIndex
from prompt_toolkit.validation import Validator, ValidationError
from PyInquirer import prompt
from query import query
from scoring import scoring_modes


def run_exploration_mode():
    index = get_index(load_from_disk=True)
    run_again = True
    while run_again:
        print()
        answers = prompt(_questions)
        query_string = answers.get('query')
        scoring_mode = answers.get('scoring_mode')
        _run_query(index, query_string, scoring_mode)
        print()
        post_answers = prompt(_post_run_questions)
        run_again = post_answers.get('run_again')


def _run_query(index: InvertedIndex, query_string: str, scoring_mode: str):
    print(f'\n--- {scoring_mode} ---')
    query_start_time = time.time_ns()
    query(index, query_string, scoring_mode)
    query_end_time = time.time_ns()
    query_duration = (query_end_time - query_start_time) / 1000000.0
    print(f'--- Query took {query_duration} milliseconds ---')


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
