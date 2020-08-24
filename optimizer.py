from torch.optim.optimizer import Optimizer


class MultipleOptimizer(Optimizer):
    def __init__(self, params, op):
        super().__init__(params, {})
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self, **kwargs):
        for op in self.optimizers:
            op.step(**kwargs)

    def state_dict(self):
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dict):
        for dict_i, op in zip(state_dict, self.optimizers):
            op.load_state_dict(dict_i)