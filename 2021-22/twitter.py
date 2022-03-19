from pymongo.collection import Collection
import pandas as pd


ENTITY = 0
DOMAIN = 1


class TwitterDataset(object):

    def __init__(self, mongodb: Collection, query: dict = None):
        self.db = mongodb
        if query is None:
            query = {}
        self.tweets = list(self.db.find(query))
        self.corpus = dict([(tweet['id'], tweet['text']) for tweet in self.tweets])
        self.E = None
        self.D = None
        self.M = None
        self._get_annotations()

    def _get_annotations(self):
        entities, domains = {}, {}
        metadata = []
        for i, tweet in enumerate(self.tweets):
            if 'context_annotations' in tweet.keys():
                for annotation in tweet['context_annotations']:
                    domain, entity = annotation['domain'], annotation['entity']
                    if domain['id'] not in domains:
                        domains[domain['id']] = domain
                    if entity['id'] not in entities:
                        entities[entity['id']] = entity
                    metadata.append({
                        'entity': entity['id'], 'domain': domain['id'], 'tweet': tweet['id']
                    })
        self.M = pd.DataFrame(metadata)
        self.E = pd.DataFrame(entities).T
        self.D = pd.DataFrame(domains).T

    def ground_truth(self, query: str, query_type: int = ENTITY):
        if query_type == ENTITY:
            data = self.E
        elif query_type == DOMAIN:
            data = self.D
        else:
            return set()
        q_tweets = set()
        data_ids = data[data['name'] == query].id.values
        for did in data_ids:
            if query_type == ENTITY:
                q_tweets = q_tweets.union(set(self.M[self.M.entity == did].tweet.values))
            elif query_type == DOMAIN:
                q_tweets = q_tweets.union(set(self.M[self.M.domain == did].tweet.values))
            else:
                q_tweets = set()
        return q_tweets

    @property
    def search_base(self):
        return [(x, self.corpus[x]) for x in set(self.M.tweet.values)]

    @property
    def entity_queries(self):
        return list(set(self.E['name'].values))

    @property
    def domain_queries(self):
        return list(set(self.D['name'].values))

    @property
    def entity_queries_stats(self):
        e_size = self.M.groupby('entity').count()
        e_size['query'] = [self.E.loc[x]['name'] for x in e_size.index.values]
        return e_size.sort_values('tweet', ascending=False)

    @property
    def domain_queries_stats(self):
        d_size = pd.DataFrame(self.M.groupby('domain').tweet.nunique(), columns=['tweet'])
        d_size['query'] = [self.D.loc[x]['name'] for x in d_size.index.values]
        return d_size.sort_values('tweet', ascending=False)