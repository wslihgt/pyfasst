"""distances.py

distance/divergence functions

2013 Jean-Louis Durrieu
"""


def ISDistortion(X,Y):
    """
    value = ISDistortion(X, Y)
  
    Returns the value of the Itakura-Saito (IS) divergence between
    matrix X and matrix Y. X and Y should be two NumPy arrays with
    same dimension.
    """
    return sum((-np.log(X / Y) + (X / Y) - 1))
