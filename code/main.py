import os

from createindex import get_index
from evaluation_mode import run_evaluation_mode
from exploration_mode import run_exploration_mode
from PyInquirer import prompt

_evaluation_mode_name = 'evaluation'
_exploration_mode_name = 'exploration'

_questions = [
    {
        'type': 'list',
        'name': 'search_mode',
        'message': 'Select search mode',
        'choices': [_evaluation_mode_name, _exploration_mode_name]
    },
]


def main():
    os.system('color')
    print()
    index = get_index(load_from_disk=True)
    print()
    answers = prompt(_questions)
    print()
    search_mode = answers.get('search_mode')

    if search_mode == _evaluation_mode_name:
        run_evaluation_mode(index)
    elif search_mode == _exploration_mode_name:
        run_exploration_mode(index)


if __name__ == '__main__':
    main()
