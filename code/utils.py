import os

from termcolor import colored

encoding = 'utf_8'


def clear_console():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)


def error(text: any) -> str:
    return colored(str(text), 'red')


def highlight(text: any) -> str:
    return colored(str(text), 'cyan')


def info(text: any) -> str:
    return colored(str(text), 'green')


def path_color(text: any) -> str:
    return colored(str(text), 'blue')
