import random


class Sampler:
    def __init__(self, indexed_ratings, transactions, batch_size, all_items, seed=42):
        self.transactions = transactions
        self.batch_size = batch_size
        self._ui_dict = {u: list(indexed_ratings[u].keys()) for u in indexed_ratings}
        self.all_items = all_items
        self.all_items.sort()
        random.shuffle(self.all_items)

    def step(self, edge_index):
        for batch_start in range(0, self.transactions, self.batch_size):
            start = batch_start
            stop = min(batch_start + self.batch_size, self.transactions)
            neg = []
            bui = edge_index[:, start: stop]
            for idx in range(bui.shape[1]):
                iid = random.sample(self.all_items, 1)[0]
                while iid in self._ui_dict[bui[0, idx]]:
                    iid = random.sample(self.all_items, 1)[0]
                neg.append(iid)
            yield bui[0].tolist(), bui[1].tolist(), neg
