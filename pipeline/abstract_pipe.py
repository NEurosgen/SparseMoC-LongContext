from abc import ABC,abstractmethod
import torch


class Pipeline(ABC):
    def __init__(self,config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run_process(self, model, data,profiler = None):
        print(f'--- Starting method {self.__class__.__name__}---')

        result = self.process(model,data,profiler)
        self.log_result(result)
        self.teardown()
        return result

    @abstractmethod
    def process(self,model,data,profiler = None):
        pass
    @abstractmethod
    def log_result(self, result):
        pass

    def teardown(self):
        pass

