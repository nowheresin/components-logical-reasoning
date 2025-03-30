# LEVEL = 'CRITICAL'


def logging(func):
    # DEBUG < INFO < WARNING < ERROR < CRITICAL
    colour = {'DEBUG': '\033[32m', 'INFO': '\033[34m', 'WARNING': '\033[33m', 'ERROR': '\033[35m',
              'CRITICAL': '\033[31m', 'None': ''}
    log_level_list = list(colour.keys())

    def wrapper(*args, **kwargs):
        # LEVEL = print.LEVEL if hasattr(print, 'LEVEL') and exec(print.LEVEL) \
        #     and print.LEVEL in log_level_list else 'DEBUG'

        LEVEL = print.LEVEL if hasattr(print, 'LEVEL') and print.LEVEL in log_level_list else 'DEBUG'

        level = kwargs.get('LOG', 'None')
        # _print(level)
        if log_level_list.index(level) >= log_level_list.index(LEVEL):
            if level == 'None':
                return func(*args, **kwargs)
            _print(f'{colour[level]}[{level}]\033[30m', end=' ')
            kwargs.pop('LOG') if 'LOG' in kwargs.keys() else ...
            return func(*args, **kwargs)
    return wrapper

_print = print
@logging
def print(*args, **kwargs):
    _print(*args, **kwargs)

if __name__ == '__main__':
    # print.LEVEL = "try:\n\t__import__('os').remove('001.txt')\nexcept: pass"
    print.LEVEL = 'WARNING'

    print('1', LOG='DEBUG')
    print('2', LOG='INFO')
    print('3', LOG='WARNING')
    print('4', LOG='ERROR')
    print('5', LOG='CRITICAL')
    print(123)

    # import sys
    # print(123, file = sys.stderr)
