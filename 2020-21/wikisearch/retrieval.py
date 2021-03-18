# -*- coding: utf-8 -*-
__project__ = 'inforet'
__author__ = 'Alfio Ferrara'
__email__ = 'alfio.ferrara@unimi.it'
__institution__ = 'Università degli Studi di Milano'
__date__ = '17 mar 2021'
__comment__ = '''
    Classes and examples for implementing a naive search engine
    and evaluate its performances over the wikisearch dataset
    Data Structure:
        {'docid': 'Id of document',
         'entity': 'Wikidata entity associated with the document',
         'target': 'Textual description of the entity',
         'wikipage': 'Wikipedia URL associated with the document',
         'query': 'Query the document is relevant for',
         'document': 'Text of the document'}
    '''

import json
import urllib.request
from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Tokenizer(ABC):

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass


class Vectorizer(ABC):

    def __init__(self, documents: List[str], tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.documents = documents

    @abstractmethod
    def vectors(self) -> np.ndarray:
        pass

    @abstractmethod
    def vec(self, document) -> np.ndarray:
        pass


class SearchEngine(object):

    def __init__(self, vectorizer: Vectorizer):
        self.vectorizer = vectorizer
        self.corpus = self.vectorizer.vectors()

    def search(self, query: str) -> Tuple[List[int], List[float]]:
        q = self.vectorizer.vec(document=query)
        match = cosine_similarity(q, self.corpus)
        answers, scores = [], []
        for i, s in sorted(enumerate(match[0]), key=lambda x: -x[1]):
            answers.append(i)
            scores.append(s)
        return answers, scores


class NLTKTokenizer(Tokenizer):

    def tokenize(self, text: str) -> List[str]:
        return nltk.word_tokenize(text)


class TfIdfVectors(Vectorizer):

    def __init__(self, documents: List[str], tokenizer: Tokenizer):
        super().__init__(documents, tokenizer)
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer.tokenize)

    def vectors(self) -> np.ndarray:
        return self.vectorizer.fit_transform(self.documents)

    def vec(self, document) -> np.ndarray:
        return self.vectorizer.transform([document])


class WikiDataset(object):

    def __init__(self, url: str):
        data = urllib.request.urlopen(url).read()
        self.D = json.loads(data)
        self.documents = []
        self.queries = []
        self.entities = []
        for document in self.D:
            self.documents.append(document['document'])
            self.entities.append(document['entity'])
            self.queries.append(document['query'])
        self.query_map = dict([(q, self.entities[i]) for i, q in enumerate(self.queries)])

    def get_queries(self):
        return list(self.query_map.keys())

    def validate_answers(self, query_entity: str, answers: List[int]) -> List[int]:
        return [1 if self.entities[x] == query_entity else 0 for x in answers]

