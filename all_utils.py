import sys


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


# class Logger(object):
#     def __init__(self, filename='default.log', stream=sys.stdout):
#         self.terminal = stream
#         self.log = open(filename, 'a')

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)

#     def flush(self):
#         pass