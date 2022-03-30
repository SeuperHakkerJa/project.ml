import numpy as np

class ActivationLayer():
    activation_types={}

    @classmethod
    def register_activation(cls, activation_name):
        def _register(activation_cls):
            cls.activation_types[activation_name]=activation_cls
            return activation_cls
        return _register

    @classmethod
    def build(cls, activation_name, **kwargs):
        if activation_name not in cls.activation_types:
            raise ValueError(f'Activation function type {activation_name} is not implemented')
        return cls.activation_types[activation_name](**kwargs)

    def __call__(self, *args, **kwargs):
        return NotImplementedError

    def gradient(self, x):
        return NotImplementedError

    def __repr__(self):
        return NotImplementedError


@ActivationLayer.register_activation('Sigmoid')
class Sigmoid(ActivationLayer):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

    def __repr__(self):
        return "Sigmoid()"


@ActivationLayer.register_activation('Softmax')
class Softmax(ActivationLayer):
    def __call__(self, x):
        e_x=np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p=self.__call__(x)
        return p * (1 - p)


@ActivationLayer.register_activation('Tanh')
class Tanh(ActivationLayer):
    def __call__(self, x):
        return 2 / (1 + np.exp(-2 * x)) - 1

    def gradient(self, x):
        return 1 - np.power(self.__call__(x), 2)


@ActivationLayer.register_activation('ReLU')
class ReLU(ActivationLayer):
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


@ActivationLayer.register_activation('LeakyReLU')
class LeakyReLU(ActivationLayer):
    def __init__(self, alpha=0.2):
        self.alpha=alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)

    def __repr__(self):
        return f"LeakyReLU(): alpha = {self.alpha}"


