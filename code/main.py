import os
import sys

from createindex import get_index
from evaluation_mode import run_evaluation_mode
from exploration_mode import run_exploration_mode
from prompt_toolkit.validation import Validator, ValidationError
from PyInquirer import prompt


def main():
    os.system('color')
    args = sys.argv[1:]
    load_from_disk = '-c' not in args and '--clean' not in args
    print()
    max_files = -1
    if not load_from_disk:
        indexing_answers = prompt(_indexing_questions)
        max_files = indexing_answers.get('max_files')
        print()
    index = get_index(load_from_disk, max_files)
    while True:
        print()
        answers = prompt(_search_questions)
        print()
        search_mode = answers.get('search_mode')
        if search_mode == _evaluation_mode_name:
            run_evaluation_mode(index)
        elif search_mode == _exploration_mode_name:
            run_exploration_mode(index)
        elif search_mode == _exit_mode:
            exit(0)


class NumberValidator(Validator):
    def validate(self, document):
        try:
            value = int(document.text)
            return value > 0 or value == -1
        except ValueError:
            raise ValidationError(message="Please enter a valid number larger",
                                  cursor_position=len(document.text))


_indexing_questions = [
    {
        'type': 'input',
        'name': 'max_files',
        'message': 'Amount of files to index (-1 for all files)',
        'validate': NumberValidator,
        'filter': lambda val: int(val)
    }
]

_evaluation_mode_name = 'evaluation'
_exploration_mode_name = 'exploration'
_exit_mode = 'exit'

_search_questions = [
    {
        'type': 'list',
        'name': 'search_mode',
        'message': 'Select search mode',
        'choices': [_evaluation_mode_name, _exploration_mode_name, _exit_mode]
    },
]

if __name__ == '__main__':
    main()
