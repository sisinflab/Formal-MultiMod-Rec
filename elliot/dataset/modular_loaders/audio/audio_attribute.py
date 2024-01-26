import typing as t
import os
import numpy as np
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class AudioAttribute(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.audio_feature_folder_path = getattr(ns, "audio_features", None)

        self.item_mapping = {}
        self.audio_features_shape = None

        items = set(str(it) for it in items)
        inner_items = self.check_items_in_folder()

        self.users = users
        self.items = items & inner_items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items
        self.item_mapping = {item: val for val, item in enumerate(self.items)}

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "AudioAttribute"
        ns.object = self
        ns.audio_feature_folder_path = self.audio_feature_folder_path

        ns.item_mapping = self.item_mapping

        ns.audio_features_shape = self.audio_features_shape

        return ns

    def check_items_in_folder(self) -> t.Set[int]:
        items = set()
        if self.audio_feature_folder_path:
            items_folder = os.listdir(self.audio_feature_folder_path)
            items = items.union(set([f.split('.')[0] for f in items_folder]))
            self.audio_features_shape = np.load(os.path.join(self.audio_feature_folder_path,
                                                             items_folder[0])).shape[0]
        return items

    def get_all_features(self):
        return self.get_all_audio_features()

    def get_all_audio_features(self):
        all_features = np.empty((len(self.items), self.audio_features_shape))
        if self.audio_feature_folder_path:
            for key, value in self.item_mapping.items():
                all_features[value] = np.load(self.audio_feature_folder_path + '/' + str(key) + '.npy')
        return all_features
