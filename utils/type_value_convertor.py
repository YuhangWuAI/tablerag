from collections import abc
from datetime import date, datetime, time
from decimal import Decimal


class TypedValueConverter:
    """
    Static class defined for type-value serialization and deserialization
    """

    converters: dict = {
        'none': None,
        'bool': None,
        'int': (int, lambda s: s),
        'float': (float, lambda s: s),
        'str': (str, lambda s: s),
        'date': (lambda x: x.isoformat(), date.fromisoformat),
        'datetime': (lambda x: x.isoformat(), datetime.fromisoformat),
        'time': (lambda x: x.isoformat(), time.fromisoformat),
        'decimal': (str, Decimal),
        'array': (str, lambda s: s),
        'dict': (str, lambda s: s),
    }

    @staticmethod
    def register(
        data_type: str, serialize_func: abc.Callable, deserialize_func: abc.Callable
    ):
        """
        Register or update a serialize and deserialize function for a data type
        """
        assert serialize_func is not None and deserialize_func is not None
        TypedValueConverter.converters[data_type] = (serialize_func, deserialize_func)

    @staticmethod
    def serialize(data_type: str, value: object, default_type: str = None) -> object:
        """
        Serialize a typed-value into json basic object
        """
        if value is None:
            return value

        data_type = data_type.lower()
        if data_type not in TypedValueConverter.converters:
            if default_type:
                data_type = default_type
            else:
                raise NotImplementedError(f"Data type `{data_type}`")

        converter = TypedValueConverter.converters[data_type]
        try:
            return converter[0](value)
        except:
            return str(value)

    @staticmethod
    def deserialize(data_type: str, value: object, default_type: str = None) -> object:
        """
        Deserialize a typed value from json basic object
        """
        if value is None:
            return value

        data_type = data_type.lower()
        if data_type not in TypedValueConverter.converters:
            if default_type:
                data_type = default_type
            else:
                raise NotImplementedError(f"Data type `{data_type}`")

        converter = TypedValueConverter.converters[data_type]
        if converter is None:
            return value

        try:
            return converter[1](value)
        except:
            return value
