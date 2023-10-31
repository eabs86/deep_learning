import numpy as np

#para problemas mais complexos não é recomendado usar a step function
def stepfunction(soma):
    if soma>=1:
        return 1
    return 0


#para problemas binários usa-se a função signmoide
def sigmoidFunction(soma):
    return 1/(1+np.exp(-soma))


#pode ser usada para classificação binária ou quando há variáveis negativas
def tanhFunction(soma):
    # return np.tanh(soma)
    return (np.exp(soma) - np.exp(-soma))/(np.exp(soma) + np.exp(-soma))


#função ReLU (rectified linear unit). Ruim para casos onde há muitos dados negativos
def reluFunction(soma):
    return np.maximum(0,soma)

#muito utilizada em problemas de regressao
def linearFunction(soma):
    return soma

#muito utilizada em problemas com mais de 2 classes. Retorna probabilidades
def softmaxFunction(valores):
    return np.exp(valores)/np.sum(np.exp(valores))
