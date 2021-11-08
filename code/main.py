import os
import sys

from createindex import get_index
from evaluation_mode import run_evaluation_mode
from exploration_mode import run_exploration_mode
from PyInquirer import prompt
from utils import clear_console


def main():
    os.system('color')
    clear_console()
    args = sys.argv[1:]
    load_from_disk = '-c' not in args and '--clean' not in args
    print()
    index = get_index(load_from_disk)
    while True:
        print()
        answers = prompt(_questions)
        print()
        action = answers.get('action')
        if action is _evaluation_action:
            run_evaluation_mode(index)
        elif action is _exploration_action:
            run_exploration_mode(index)
        elif action is _reset_index_action:
            index = get_index(load_from_disk=False)
        elif action is _exit_action:
            exit(0)
        clear_console()


_evaluation_action = 'Evaluation'
_exploration_action = 'Exploration'
_reset_index_action = 'Reset index'
_exit_action = 'Exit'

_questions = [
    {
        'type': 'list',
        'name': 'action',
        'message': 'Select action',
        'choices': [_evaluation_action, _exploration_action, _reset_index_action, _exit_action]
    },
]

if __name__ == '__main__':
    main()
