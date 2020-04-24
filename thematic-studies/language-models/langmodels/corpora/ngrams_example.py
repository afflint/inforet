# -*- coding: utf-8 -*-
__project__ = 'inforet'
__author__ = 'Alfio Ferrara'
__email__ = 'alfio.ferrara@unimi.it'
__institution__ = 'Università degli Studi di Milano'
__date__ = '21 apr 2020'
__comment__ = '''
    Classes for accessing
    The Cornell Movie-Dialogs Corpus
    Available at https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html.
    Danescu-Niculescu-Mizil, C., & Lee, L. (2011, June). 
    Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs. 
    In Proceedings of the 2nd workshop on cognitive modeling and computational linguistics (pp. 76-87). 
    Association for Computational Linguistics.
    The dataset has been uploaded in MongoDb using the script available at 
    https://github.com/afflint/inforet/blob/master/thematic-studies/language-models/dataset/acquisition.ipynb.
    '''
import pymongo
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from collections import defaultdict
from typing import Iterable
import string
import re
import numpy as np
import spacy


STOPWORDS = set(stopwords.words('english'))


class EntityNotFound(Exception):

    def __init__(self, *args):
        if args > 1:
            self.entity, self.id = args[0], args[1]
        else:
            self.entity, self.id = None, None

    def __str__(self):
        if self.entity and self.id:
            return "EntityNotFound: Entity `{}` with id `{}` is not found in corpus".format(self.entity, self.id)
        else:
            return "EntityNotFound"


class UnsupportedLanguageModel(Exception):

    def __init__(self, *args):
        if args > 0:
            self.count = args[0]
        else:
            self.count = None

    def __str__(self):
        if self.count:
            return "UnsupportedLanguageModel: `{}-gram` Language Models are not currently supported".format(self.count)
        else:
            return "UnsupportedLanguageModel"


class Corpus(object):

    UNIGRAM = 1
    BIGRAM = 2
    TRIGRAM = 3

    def __init__(self, db_name: str, k_smoothing=0):
        """
        Access the database and set all the collections.
        Assumes to connect a local open instance of MongoDb storing the dataset in
        characters
        conversations
        lines
        movies
        :param db_name: name of the database
        """
        self.db = pymongo.MongoClient()[db_name]
        self.characters = self.db['characters']
        self.conversations = self.db['conversations']
        self.lines = self.db['lines']
        self.movies = self.db['movies']
        self.k_smooting = k_smoothing
        self.unigram = defaultdict(lambda: self.k_smooting)
        self.bigram = defaultdict(lambda: self.k_smooting)
        self.trigram = defaultdict(lambda: self.k_smooting)
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def _check(token, remove_stopwords=True):
        if token in string.punctuation:
            return False
        elif token.startswith("'"):
            return False
        elif token == '--':
            return False
        elif token == '``':
            return False
        elif remove_stopwords and token in STOPWORDS:
            return False
        else:
            return True

    @staticmethod
    def tokenize(text, remove_stopwords=True):
        t = re.sub(r'(?<=\S)\.(?=\w)', '. ', text)
        return [x for x in word_tokenize(t.lower()) if Corpus._check(x, remove_stopwords=remove_stopwords)]

    def pos_tokenize(self, text):
        return [x.pos_ for x in self.nlp(text.lower())]

    @property
    def N(self):
        return sum(self.unigram.values())

    def predict(self, sequence: Iterable[str], model: int = UNIGRAM, log_probabilities: bool = False):
        probabilities = None
        if model == Corpus.UNIGRAM:
            probabilities = [self.p(x, log_probabilities=log_probabilities) for x in sequence]
        elif model == Corpus.BIGRAM:
            s = nltk.ngrams(sequence, 2)
            probabilities = [self.p(x, y, log_probabilities=log_probabilities) for x, y in s]
        elif model == Corpus.TRIGRAM:
            s = nltk.ngrams(sequence, 3)
            probabilities = [self.p(x, y, z, log_probabilities=log_probabilities) for x, y, z in s]
        else:
            raise UnsupportedLanguageModel(model)
        if probabilities is not None:
            return np.prod(probabilities), probabilities

    def perplexity(self, sequence: Iterable[str],
                   model: int = UNIGRAM, log_probabilities=False):
        _, p = self.predict(sequence, model, log_probabilities=False)
        if log_probabilities:
            pp = np.power(2, -np.sum(p) / len(p))
        else:
            pp = np.power(np.prod([1 / x for x in p]), 1 / len(p))
        return pp

    def get_testset(self, texts: Iterable[str], remove_stopwords: bool = False, pos: bool = False):
        test = []
        for text in texts:
            if pos:
                tokens = ['#S'] + self.pos_tokenize(text) + ['#E']
            else:
                tokens = ['#S'] + Corpus.tokenize(text, remove_stopwords=remove_stopwords) + ['#E']
            test.append(tokens)
        return test

    def p(self, *args, log_probabilities=False):
        p = None
        if len(args) == 1:
            p = self.unigram[args[0]] / self.N
        elif len(args) == 2:
            p = self.bigram[(args[0], args[1])] / self.unigram[args[0]]
        elif len(args) == 3:
            p = self.trigram[(args[0], args[1], args[2])] / self.bigram[(args[0], args[1])]
        else:
            raise UnsupportedLanguageModel(len(args))
        if p is not None:
            if log_probabilities:
                return np.log(p)
            else:
                return p

    def unigram_frequencies(self, texts: Iterable[str], refresh: bool = True,
                            remove_stopwords: bool = False, pos: bool = False):
        """
        Performs tokenization and stores frequencies in self.unigram
        :param texts: iterable over texts
        :param refresh: if True, the unigram index will be drop before indexing
        :param remove_stopwords: if True perform stopwords removal
        :param pos: if True performs POS tokenization
        :return: None
        """
        if refresh:
            self.unigram = defaultdict(lambda : self.k_smooting)
        for text in texts:
            if pos:
                tokens = ['#S'] + self.pos_tokenize(text) + ['#E']
            else:
                tokens = ['#S'] + Corpus.tokenize(text, remove_stopwords=remove_stopwords) + ['#E']
            for token in tokens:
                self.unigram[token] += 1

    def bigram_frequencies(self, texts: Iterable[str], refresh: bool = True,
                           remove_stopwords: bool = False, pos: bool = False):
        """
        Performs tokenization and stores frequencies in self.bigram
        :param texts: iterable over texts
        :param refresh: if True, the bigram index will be drop before indexing
        :param remove_stopwords: if True perform stopwords removal
        :param pos: if True performs POS tokenization
        :return: None
        """
        if refresh:
            self.bigram = defaultdict(lambda : self.k_smooting)
        for text in texts:
            if pos:
                tokens = ['#S'] + self.pos_tokenize(text) + ['#E']
            else:
                tokens = ['#S'] + Corpus.tokenize(text, remove_stopwords=remove_stopwords) + ['#E']
            for (a, b) in nltk.ngrams(tokens, 2):
                self.bigram[(a, b)] += 1

    def trigram_frequencies(self, texts: Iterable[str], refresh: bool = True,
                            remove_stopwords: bool = False, pos: bool = False):
        """
        Performs tokenization and stores frequencies in self.trigram
        :param texts: iterable over texts
        :param refresh: if True, the trigram index will be drop before indexing
        :param remove_stopwords: if True perform stopwords removal
        :param pos: if True performs POS tokenization
        :return: None
        """
        if refresh:
            self.trigram = defaultdict(lambda : self.k_smooting)
        for text in texts:
            if pos:
                tokens = ['#S'] + self.pos_tokenize(text) + ['#E']
            else:
                tokens = ['#S'] + Corpus.tokenize(text, remove_stopwords=remove_stopwords) + ['#E']
            for (a, b, c) in nltk.ngrams(tokens, 3):
                self.trigram[(a, b, c)] += 1

    def movie_list(self, query: dict = None) -> list:
        """
        Returns a list of movies with id and titles
        :param query: an optional dictionary containing a MongoDb query
        :return: a list of (movie_id, title)
        """
        return [(record['id'], record['title']) for record in self.movies.find(query)]


class Movie(Corpus):

    def __init__(self, db_name: str, movie_id: str, k_smoothing: int = 0):
        super().__init__(db_name, k_smoothing=k_smoothing)
        self.movie_id = movie_id
        self.fields = ['title', 'year', 'rating', 'votes', 'genres']
        for field in self.fields:
            setattr(self, field, None)
        metadata = self.movies.find_one({'id': self.movie_id})
        if metadata is None:
            raise EntityNotFound(self.__class__.__name__, self.movie_id)
        else:
            for k, v in metadata.items():
                if k != '_id':
                    setattr(self, k, v)

    @property
    def size(self):
        return self.lines.count_documents({"character.movie.id": self.movie_id})

    def get_lines_text(self) -> list:
        q, p = {"character.movie.id": self.movie_id}, {"_id": 0, "id": 1, "text": 1}
        return [line['text'] for line in self.lines.find(q, p).sort([('id', pymongo.ASCENDING)])]