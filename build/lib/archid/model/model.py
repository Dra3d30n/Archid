import numpy as np 
class Model:
    def __init__(self,network,learning):
        self.network=network
        self.learning=learning

        self.learning.attach_network(self.network)
    def train(self, *args, **kwargs):
        return self.learning.train(*args, **kwargs)
    def predict(self,input):
        return self.network.forward(input)
    def eval(self):
        self.network.eval()
    def train_mode(self):
        self.network.train()