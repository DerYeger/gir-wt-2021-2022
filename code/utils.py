import os

from termcolor import colored

encoding = 'utf_8'


def clear_console():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)


def error(text) -> str:
    return colored(text, 'red')


def highlight(text) -> str:
    return colored(str(text), 'cyan')


def info(text: str) -> str:
    return colored(text, 'green')


def path_color(text: str) -> str:
    return colored(text, 'blue')
