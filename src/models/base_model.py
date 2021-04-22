class BaseModel:
    """
    A model should encapsulate everything related to its training and inference.
    Only the most generic APIs should be exposed for maximal flexibility
    """
    def __init__(self):
        self.cfg = None
        self.net = None
        self.criteria = None
        self.optimizer = None
        self.scheduler = None
        self.log_writer = None
        self.total_train_epoch = -1
        self.curr_epoch = 0

        self.is_train = None

    def set_train_eval_state(self, to_train):
        assert to_train in (True, False)
        if self.is_train is None or self.is_train != to_train:
            if to_train:
                self.net.train()
            else:
                self.net.eval()
            self.is_train = to_train

    def ingest_train_input(self, input):
        raise NotImplementedError()

    def infer(self, input):
        raise NotImplementedError()

    def optimize_params(self):
        raise NotImplementedError()

    def advance_to_next_epoch(self):
        raise NotImplementedError()

    def load_latest_checkpoint_if_available(self, manager):
        raise NotImplementedError()

    def write_checkpoint(self, manager):
        raise NotImplementedError()

    def add_log_writer(self, log_writer):
        self.log_writer = log_writer

    def log_statistics(self, step, level=0):
        pass
