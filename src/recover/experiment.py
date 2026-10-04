class Experiment:
    def __init__(self, dataset, model, cv, metrics):
        self.dataset = dataset
        self.model = model
        self.cv = cv
        self.metrics = metrics

    def run(self):
        pass
