from typing import List, Tuple, Callable, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfIdfSearchEngine(object):

    def __init__(self, search_base: List[Tuple[str, str]],
                 tokenizer: Callable = None):
        self.document_ids = []
        self.documents = []
        for doc_id, document in search_base:
            self.document_ids.append(doc_id)
            self.documents.append(document)
        self.vectorizer = TfidfVectorizer(tokenizer=tokenizer)
        self.X = self.vectorizer.fit_transform(self.documents)
        self.vocabulary = self.vectorizer.get_feature_names_out()

    def search(self, query: str):
        q = self.vectorizer.transform([query])
        s = cosine_similarity(q, self.X)
        ranking = [(i, self.document_ids[i], score) for i, score in
                   sorted(enumerate(s[0]), key=lambda x: -x[1])]
        return ranking

    @staticmethod
    def feedback(ranking: List[Tuple[int, str, str]], ground_truth: Set,
                 top_k: int):
        tp, fp, fn, tn = list(), list(), list(), list()
        retrieved = ranking[:top_k]
        non_retrieved = ranking[top_k:]
        for j, (i, doc_id, score) in enumerate(retrieved):
            if doc_id in ground_truth:
                tp.append(doc_id)
            else:
                fp.append(doc_id)
        for j, (i, doc_id, score) in enumerate(non_retrieved):
            if doc_id in ground_truth:
                fn.append(doc_id)
            else:
                tn.append(doc_id)
        return tp, fp, fn, tn