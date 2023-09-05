class Sampler:
    def __init__(self, transactions, batch_size):
        self.transactions = transactions
        self.batch_size = batch_size

    def step(self, edge_index):
        for batch_start in range(0, self.transactions, self.batch_size):
            start = batch_start
            stop = min(batch_start + self.batch_size, self.transactions)
            bui = edge_index[:, start: stop]
            yield bui
