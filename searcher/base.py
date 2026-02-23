from abc import ABC


class BaseSearcher(ABC):
    def search(self, query, threshold, indexer):
        raise NotImplementedError
