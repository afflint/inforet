# -*- coding: utf-8 -*-
__project__ = 'inforet'
__author__ = 'Alfio Ferrara'
__email__ = 'alfio.ferrara@unimi.it'
__institution__ = 'Università degli Studi di Milano'
__date__ = '24 apr 2020'
__comment__ = '''
    Abstract definition of corpus
    '''
from abc import ABC, abstractmethod
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import spacy


SPACY_MODELS = {
    'english': "en_core_web_sm"
}


class UnsupportedLanguage(Exception):

    def __init__(self, *args):
        if args > 0:
            self.lang = args[0]
        else:
            self.lang = None

    def __str__(self):
        if self.lang:
            return "UnsupportedLanguage: `{}` id not currently supported".format(self.lang)
        else:
            return "UnsupportedLanguage"


class Corpus(ABC):

    def __init__(self, drop_stopwords: bool = False,
                 use_pos: bool = False, mix_pos: bool = False,
                 language='english',
                 pos_filter: list = None, lemma: bool = False):
        """
        We currently use nltk for tokenization whenever possible and spacy for pos detection.
        :param drop_stopwords: if True stopwords are removed
        :param use_pos: if True POS sequences are returned instead of token sequences
        :param mix_pos: if True POS are combined with tokens, like cat_NOUN
        :param language: language to use for spacy and nltk models
        :param pos_filter: if given, filters only specified pos
        :param lemma: if True keep only the lemma
        """
        self.drop_stopwords = drop_stopwords
        self.use_pos = use_pos
        self.mix_pos = mix_pos
        self.language = language
        self.stopwords = set(stopwords.words(self.language))
        self.pos_filter = pos_filter
        self.lemma = lemma
        try:
            self.nlp = spacy.load(SPACY_MODELS[self.language])
        except KeyError:
            raise UnsupportedLanguage(self.language)
        super().__init__()

    def _check(self, token):
        if token in string.punctuation:
            return False
        elif token.startswith("'"):
            return False
        elif token == '--':
            return False
        elif token == '``':
            return False
        elif self.drop_stopwords and token in self.stopwords:
            return False
        else:
            return True

    def _tokenize(self, text):
        try:
            t = re.sub(r'(?<=\S)\.(?=\w)', '. ', text)
            return [x for x in word_tokenize(t.lower()) if self._check(x)]
        except TypeError:
            return []

    def _pos_tokenize(self, text):
        return [x.pos_ for x in self.nlp(text.lower())]

    def tokenize(self, text):
        if self.use_pos:
            tokens = self._pos_tokenize(text)
        else:
            if self.pos_filter is None and not self.lemma:
                if self.mix_pos:
                    tokens = ["{}_{}".format(x.text, x.pos_) for x in self.nlp(text.lower())]
                else:
                    tokens = self._tokenize(text)
            elif self.lemma:
                if self.pos_filter is not None:
                    if self.mix_pos:
                        tokens = ["{}_{}".format(x.lemma_, x.pos_) for x in
                                  self.nlp(text.lower()) if x.pos_ in self.pos_filter]
                    else:
                        tokens = [x.lemma_ for x in self.nlp(
                            text.lower()) if x.pos_ in self.pos_filter]
                else:
                    if self.mix_pos:
                        tokens = ["{}_{}".format(x.lemma_, x.pos_) for x in
                                  self.nlp(text.lower())]
                    else:
                        tokens = [x.lemma_ for x in self.nlp(text.lower())]
            else:
                if self.pos_filter is not None:
                    if self.mix_pos:
                        tokens = ["{}_{}".format(x.text, x.pos_) for x in
                                  self.nlp(text.lower()) if x.pos_ in self.pos_filter]
                    else:
                        tokens = [x.text for x in self.nlp(
                            text.lower()) if x.pos_ in self.pos_filter]
                else:
                    if self.mix_pos:
                        tokens = ["{}_{}".format(x.text, x.pos_) for x in self.nlp(text.lower())]
                    else:
                        tokens = self._tokenize(text)
        return tokens

    @staticmethod
    def skip(sequence, n=2, s=2):
        k_grams = []
        for i in range(len(sequence)):
            for z in range(s):
                seq = [sequence[i]] + sequence[i + z + 1:i + z + n]
                if len(seq) == n and seq not in k_grams:
                    k_grams.append(tuple(seq))
        return k_grams

    @abstractmethod
    def get_tokens(self) -> list:
        """
        Get documents tokenized as per drop_stopwords and use_pos.
        :return: list of pairs (line_id, tokens)
        """
        pass

    @abstractmethod
    def get_text(self) -> list:
        """
        Get document original text.
        :return: list of pairs (line_id, text as string)
        """
        pass

    @property
    def size(self) -> int:
        """
        Number of documents in the corpus
        :return: The number of documents in the corpus
        """
        return 0

    @property
    def vocabulary(self) -> list:
        """
        Corpus vocabulary
        :return: list, unique tokens
        """
        return []
