"""
This file contains common utility functions.
"""

def auto_assign(func):
    """
    Automatically assigns the parameters.
    """
    def wrapper(self, *args, **kwargs):
        keys = func.__code__.co_varnames[1:func.__code__.co_argcount]
        for key, value in zip(keys, args):
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)
        
        func(self, *args, **kwargs)
        
    return wrapper