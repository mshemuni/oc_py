from logging import Logger, getLogger

import numpy as np

from .errors import LengthCheckError


class Checker:
    @staticmethod
    def length_checker(data, reference):
        if len(reference) != len(data):
            raise LengthCheckError("length of data is not sufficient")


class Fixer:
    @staticmethod
    def length_fixer(data, reference):
        if reference is None:
            return data

        if isinstance(data, str):
            return np.array([data] * len(reference), dtype=object)

        if hasattr(data, "__len__"):
            Checker.length_checker(data, reference)
            if isinstance(data, list):
                return np.array(data)
            return data
        else:
            return np.array([data] * len(reference))

    @staticmethod
    def none_to_nan(data_frame):
        return data_frame.replace({None: np.nan})

    @staticmethod
    def logger(logger: Logger | None = None, name: str | None = None) -> Logger:
        """
        Checks if a logger is passed as an argument. If not, it returns a logger with the specified name
        or a default name.


        Parameters
        ----------
        logger: Logger, default = None
            An optional Logger instance.
        name: str, default = None
            An optional string representing the name of the logger.

        Returns
        -------
        units.Quantity
            The converted value in time.
        """
        if logger is None:
            if name is None:
                return getLogger("OCPY")
            else:
                return getLogger(name)
        else:
            return logger
