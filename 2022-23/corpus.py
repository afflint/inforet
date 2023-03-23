from typing import List

import nltk
import numpy as np
import spacy
nlp = spacy.load('en_core_web_lg')
from gensim.corpora import Dictionary
from collections import defaultdict, Iterable


def tokenizer(text, pos_filters=None,
              lemma=False, lowercase=False):
    tokens = []
    for token in nlp(text):
        if pos_filters is None or token.pos_ in pos_filters:
            if lemma:
                x = token.lemma_
            else:
                x = token.text
            if lowercase:
                tokens.append(x.lower())
            else:
                tokens.append(x)
    return tokens


class Corpus(object):

    def __init__(self, documents: Iterable,
                 pos_filters: List[str] = None,
                 lemma: bool = False, lowercase: bool = False):
        self.pos_filters = pos_filters
        self.lemma = lemma
        self.lowercase = lowercase
        self.documents = [
            tokenizer(x,
                      pos_filters=self.pos_filters,
                      lemma=self.lemma,
                      lowercase=self.lowercase) for x in documents]
        self.bigram_mi = defaultdict(lambda: 0)
        self.trigram_mi = defaultdict(lambda: 0)
        self.bigrams_index, self.trigrams_index = defaultdict(list), defaultdict(list)
        self.dictionary = Dictionary()

    def add_bigrams(self, min_mi: float = 0.01):
        for bigram, mi in self.bigram_mi.items():
            if mi > min_mi:
                for d in self.bigrams_index[bigram]:
                    self.documents[d].append("{} {}".format(bigram[0], bigram[1]))

    def add_trigrams(self, min_mi: float = 0.01):
        for trigram, mi in self.trigram_mi.items():
            if mi > min_mi:
                for d in self.trigrams_index[trigram]:
                    self.documents[d].append(
                        "{} {} {}".format(trigram[0], trigram[1], trigram[2]))

    def create_dictionary(self):
        self.dictionary.add_documents(self.documents)

    def bag_of_words(self) -> List[List[int]]:
        return [self.dictionary.doc2idx(x) for x in self.documents]

    def query_to_bow(self, query: str, bigrams_mi: float = 0.01, trigrams_mi: float = 0.01):
        q_tokens = tokenizer(query, self.pos_filters, self.lemma, self.lowercase)
        for bigram in nltk.ngrams(q_tokens, n=2):
            if self.bigram_mi[bigram] > bigrams_mi:
                q_tokens.append("{} {}".format(bigram[0], bigram[1]))
        for trigram in nltk.ngrams(q_tokens, n=3):
            if self.trigram_mi[trigram] > trigrams_mi:
                q_tokens.append("{} {} {}".format(trigram[0], trigram[1], trigram[1]))
        return self.dictionary.doc2idx(q_tokens)


    def index_bigrams(self, threshold: int = 5):
        unigrams, bigrams, trigrams = defaultdict(lambda: 0), \
                                      defaultdict(lambda: 0), \
                                      defaultdict(lambda: 0)
        for i, document in enumerate(self.documents):
            for unigram in document:
                unigrams[unigram] += 1
            for bigram in nltk.ngrams(document, n=2):
                bigrams[bigram] += 1
                self.bigrams_index[bigram].append(i)
            for trigram in nltk.ngrams(document, n=3):
                trigrams[trigram] += 1
                self.trigrams_index[trigram].append(i)
        U, B, T = sum(unigrams.values()), sum(bigrams.values()), sum(trigrams.values())
        for bigram, count in bigrams.items():
            if unigrams[bigram[0]] > threshold and unigrams[bigram[1]] > 0:
                self.bigram_mi[bigram] = (count / B) * np.log((count / B) /
                                                         ((unigrams[bigram[0]] / U) *
                                                          (unigrams[bigram[1]] / U)))
        for trigram, count in trigrams.items():
            if unigrams[trigram[0]] > threshold and \
                    unigrams[trigram[1]] > threshold and \
                    unigrams[trigram[2]] > threshold:
                self.trigram_mi[trigram] = (count / T) * np.log((count / T) /
                                                           ((unigrams[trigram[0]] / U) *
                                                            (unigrams[trigram[1]] / U) *
                                                            (unigrams[trigram[2]] / U)))
