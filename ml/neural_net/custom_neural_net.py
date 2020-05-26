## API/Interface for the Neural Network ##
class DigitNeuralNetI:

    ### Train the Neural Net ###
    def train(self, path: str, epoch: int, lr: float):
        pass

    ### Test the trained Neural Net ###
    def test(self, path: str) -> ({}, float):
        pass

    def predict(self, img_path: str) -> int:
        pass

    ## Load Serialized Neural Net from disk ##
    ## DeSerialize Neural Net ##
    def load(self):
        pass

    ## Serialize Neural Net ##
    ## Save Serialized Neural Net to disk ##
    def save(self):
        pass

    ## Get Neural Network Parameters ##
    def parameters() -> {}:
        pass
