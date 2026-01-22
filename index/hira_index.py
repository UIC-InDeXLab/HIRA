from hira.index.hira_config import HiraConfig
from hira.index.hira_config import DeviceMode
from hira.index.indexer import Indexer
from hira.index.searcher import Searcher


class HiraIndex:
    def __init__(self, config: HiraConfig):
        self.config = config
        self.device_mode = self.config.device_mode
        self.indexer = None
        self.searcher = None

    def build(self):
        pass

    def search(self):
        pass

    def update(self):
        # TODO
        pass
