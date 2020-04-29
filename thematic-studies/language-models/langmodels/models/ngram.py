# -*- coding: utf-8 -*-
__project__ = 'inforet'
__author__ = 'Alfio Ferrara'
__email__ = 'alfio.ferrara@unimi.it'
__institution__ = 'Università degli Studi di Milano'
__date__ = '29 apr 2020'
__comment__ = '''
    Implements n-gram models for example purposes
    '''
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from corpora.corpus import Corpus
from collections import defaultdict


class NGram(object):

    def __init__(self, n_grams: int = 2, k_smooth: float = 1):
        self.n = n_grams
        self.k = k_smooth
        self.index = self._init_index(self.n, lambda : self.k)

    def _init_index(self, k, type):
        if k == 1:
            return defaultdict(type)
        else:
            return defaultdict(lambda: self._init_index(k - 1, type))

