"""
Decorator that attaches analytical gradients to methods.
"""

import functools


class differentiable:
    """
    Method decorator exposing registered gradients as ``obj.method.grad_<variable>``.

    Usage::

        @differentiable
        def viscosity(self, T, P=101325.0): ...

        @viscosity.grad("T")
        def _viscosity_grad_T(self, T, P=101325.0): ...

    ``obj.viscosity(T)`` then calls the method and ``obj.viscosity.grad_T(T)``
    its gradient with respect to ``T``. Gradients must be named
    ``_<method>_grad_<variable>``; the docs template uses this name to list them
    under their method.

    Parameters
    ----------
    func : callable
        method returning the value
    """

    def __init__(self, func):
        self.func = func
        self.gradients = {}
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__

    def grad(self, variable: str):
        """
        Register the gradient with respect to ``variable``.

        Parameters
        ----------
        variable : str
            name of the variable; the gradient is exposed as ``grad_<variable>``

        Returns
        -------
        callable
            decorator storing the gradient method and returning it unchanged
        """

        def register(gradient):
            expected = f"_{self.__name__}_grad_{variable}"
            if gradient.__name__ != expected:
                raise NameError(
                    f"gradient of {self.__name__!r} w.r.t. {variable!r} must be named "
                    f"{expected!r}, got {gradient.__name__!r}"
                )
            self.gradients[variable] = gradient
            return gradient

        return register

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return _BoundDifferentiable(self, obj)


class _BoundDifferentiable:
    """
    ``differentiable`` bound to an instance.

    Parameters
    ----------
    descriptor : differentiable
        descriptor holding the method and its gradients
    obj : object
        instance the method is bound to
    """

    def __init__(self, descriptor: differentiable, obj):
        self._descriptor = descriptor
        self._obj = obj
        functools.update_wrapper(self, descriptor.func)

    def __call__(self, *args, **kwargs):
        return self._descriptor.func(self._obj, *args, **kwargs)

    def __getattr__(self, name: str):
        descriptor = self.__dict__.get("_descriptor")
        if descriptor is None:  # attribute lookups before __init__ (e.g. copy)
            raise AttributeError(name)
        variable = name.removeprefix("grad_")
        if name.startswith("grad_") and variable in descriptor.gradients:
            return functools.partial(descriptor.gradients[variable], self._obj)
        available = ", ".join(f"grad_{v}" for v in descriptor.gradients) or "none"
        raise AttributeError(
            f"{descriptor.func.__name__!r} has no attribute {name!r}; "
            f"available gradients: {available}"
        )
