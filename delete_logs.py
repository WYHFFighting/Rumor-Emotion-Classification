# -*- coding: utf-8 -*-
import re
import os


def run(pattern):
    files = os.listdir('./logs')
    for f in files:
        if re.search(pattern, f):
            with open('./logs/{}'.format(f), 'r', encoding = 'utf-8') as fr:
                L = len(fr.read().split('\n'))
            if L <= 30:
                os.remove('./logs/{}'.format(f))


if __name__ == '__main__':
    pattern = re.compile(r'roberta_fnn_24.*?\.log')
    run(pattern)
