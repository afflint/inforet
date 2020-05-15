# -*- coding: utf-8 -*-
__project__ = 'inforet'
__author__ = 'Alfio Ferrara'
__email__ = 'alfio.ferrara@unimi.it'
__institution__ = 'Università degli Studi di Milano'
__date__ = '24 apr 2020'
__comment__ = '''
    Implements corpus classes needed to access the
    The Cornell Movie-Dialogs Corpus
    Available at https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html.
    Danescu-Niculescu-Mizil, C., & Lee, L. (2011, June). 
    Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs. 
    In Proceedings of the 2nd workshop on cognitive modeling and computational linguistics (pp. 76-87). 
    Association for Computational Linguistics.
    The dataset has been uploaded in MongoDb using the script available at 
    https://github.com/afflint/inforet/blob/master/thematic-studies/language-models/dataset/acquisition.ipynb.
    '''
from .corpus import Corpus
import pymongo


class MalformedCollection(Exception):

    def __init__(self, *args):
        if args > 0:
            self.collection = args[0]
        else:
            self.collection = None

    def __str__(self):
        if self.collection:
            message = """
            MalformedCollection: Collection `{}` is malformed or projection clause is not correct.
            The pipeline must project an `id` and a `text` field.
            """.format(self.collection)
            return message
        else:
            return "MalformedCollection"


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


class MovieDialogCollection(Corpus):

    def __init__(self, db_name: str, collection_name: str,
                 drop_stopwords: bool = False, use_pos: bool = False,
                 mix_pos: bool = False,
                 language='english', pipeline: list = None,
                 pos_filter: list = None, lemma: bool = False):
        super().__init__(drop_stopwords=drop_stopwords, use_pos=use_pos,
                         mix_pos=mix_pos,
                         language=language,
                         pos_filter=pos_filter, lemma=lemma)
        self.db = pymongo.MongoClient()[db_name]
        self.collection = self.db[collection_name]
        if pipeline is None:
            self.pipeline = []
        else:
            self.pipeline = pipeline

    def __iter__(self):
        cursor = self.collection.aggregate(self.pipeline, allowDiskUse=True, maxTimeMS=100000000)
        with cursor:
            for record in cursor:
                try:
                    if record['text'] is not None:
                        yield record['id'], record['text']
                    else:
                        pass
                except KeyError:
                    raise MalformedCollection(self.collection.name)

    def get_tokens(self) -> list:
        return [(id, self.tokenize(text)) for id, text in self]

    def get_skip_tokens(self, n=2, s=2):
        return [(id, Corpus.skip(self.tokenize(text), n=n, s=s)) for id, text in self]

    def get_text(self) -> list:
        return [(id, text) for id, text in self]

    @property
    def size(self):
        return len(self.get_text())

    @property
    def vocabulary(self) -> list:
        v = set()
        for id, tokens in self.get_tokens():
            for token in tokens:
                v.add(token)
        return list(v)

    @property
    def word_count(self) -> int:
        w = 0
        for id, tokens in self.get_tokens():
            w += len(tokens)
        return w


class MovieDialogMovie(MovieDialogCollection):

    def __init__(self, movie_id: str,
                 db_name: str, drop_stopwords: bool = False,
                 use_pos: bool = False,
                 mix_pos: bool = False,
                 language='english', pipeline: list = None,
                 pos_filter: list = None, lemma: bool = False):
        super().__init__(db_name, 'movies', drop_stopwords, use_pos,
                         mix_pos,
                         language, pipeline,
                         pos_filter, lemma)
        self.movie_id = movie_id
        self.fields = ['title', 'year', 'rating', 'votes', 'genres']
        for field in self.fields:
            setattr(self, field, None)
        metadata = self.collection.find_one({'id': self.movie_id})
        if metadata is None:
            raise EntityNotFound(self.__class__.__name__, self.movie_id)
        else:
            for k, v in metadata.items():
                if k != '_id':
                    setattr(self, k, v)
