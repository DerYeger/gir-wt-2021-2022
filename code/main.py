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
    print()
    answers = prompt(_questions)
    print()
    search_mode = answers.get('search_mode')

    if search_mode == _evaluation_mode_name:
        run_evaluation_mode()
    elif search_mode == _exploration_mode_name:
        run_exploration_mode()


if __name__ == '__main__':
    main()
