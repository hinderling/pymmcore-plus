from itertools import chain
from typing import Type

from loguru import logger
from Pyro5 import errors
from Pyro5.api import behavior, expose, oneway

from ._mmcore_plus import CMMCorePlus


def wrap_for_pyro(cls: Type) -> Type:
    """Create proxy class compatible with pyro.

    Some classes, such as those autogenerated by SWIG, may be difficult
    to expose via `Pyro.api.expose`, because Pyro wants to add attributes
    directly to every class and class method that it exposes.  In some
    cases, this leads to an error like:

    AttributeError: 'method_descriptor' object has no attribute '_pyroExposed'

    This wrapper takes a class and returns a proxy class whose methods can
    all be modified.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._obj = cls(*args, **kwargs)

    def _proxy_method(name):
        def _f(self, *args, **kwargs):
            obj = getattr(self, "_obj")
            method = getattr(obj, name)
            return method(*args, **kwargs)

        _f.__name__ = name
        return _f

    _dict_ = {}
    for k, v in chain(*(c.__dict__.items() for c in reversed(cls.mro()))):
        if callable(v) and not k.startswith("_"):
            _dict_[k] = _proxy_method(k)
            for attr in dir(v):
                if attr.startswith("_pyro"):
                    setattr(_dict_[k], attr, getattr(v, attr))

    _dict_["__init__"] = __init__
    return type(f"{cls.__name__}Proxy", (), _dict_)


@expose
@behavior(instance_mode="single")
@wrap_for_pyro
class pyroCMMCore(CMMCorePlus):
    @oneway
    def run_mda(self, sequence) -> None:
        return super().run_mda(sequence)

    @oneway
    def emit_signal(self, signal_name, *args):
        logger.debug("{}: {}", signal_name, args)
        for handler in list(self._callback_handlers):
            try:
                handler._pyroClaimOwnership()
                # FIXME: magic name connection with RemoteMMCore.register_callback
                handler.receive_core_callback(signal_name, args)
            except errors.CommunicationError:
                self.disconnect_remote_callback(handler)
