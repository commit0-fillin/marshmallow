"""Field classes for various types of data."""
from __future__ import annotations
import collections
import copy
import datetime as dt
import decimal
import ipaddress
import math
import numbers
import typing
import uuid
import warnings
from collections.abc import Mapping as _Mapping
from enum import Enum as EnumType
from marshmallow import class_registry, types, utils, validate
from marshmallow.base import FieldABC, SchemaABC
from marshmallow.exceptions import FieldInstanceResolutionError, StringNotCollectionError, ValidationError
from marshmallow.utils import is_aware, is_collection, resolve_field_instance
from marshmallow.utils import missing as missing_
from marshmallow.validate import And, Length
from marshmallow.warnings import RemovedInMarshmallow4Warning
__all__ = ['Field', 'Raw', 'Nested', 'Mapping', 'Dict', 'List', 'Tuple', 'String', 'UUID', 'Number', 'Integer', 'Decimal', 'Boolean', 'Float', 'DateTime', 'NaiveDateTime', 'AwareDateTime', 'Time', 'Date', 'TimeDelta', 'Url', 'URL', 'Email', 'IP', 'IPv4', 'IPv6', 'IPInterface', 'IPv4Interface', 'IPv6Interface', 'Enum', 'Method', 'Function', 'Str', 'Bool', 'Int', 'Constant', 'Pluck']
_T = typing.TypeVar('_T')

class Field(FieldABC):
    """Basic field from which other fields should extend. It applies no
    formatting by default, and should only be used in cases where
    data does not need to be formatted before being serialized or deserialized.
    On error, the name of the field will be returned.

    :param dump_default: If set, this value will be used during serialization if the
        input value is missing. If not set, the field will be excluded from the
        serialized output if the input value is missing. May be a value or a callable.
    :param load_default: Default deserialization value for the field if the field is not
        found in the input data. May be a value or a callable.
    :param data_key: The name of the dict key in the external representation, i.e.
        the input of `load` and the output of `dump`.
        If `None`, the key will match the name of the field.
    :param attribute: The name of the key/attribute in the internal representation, i.e.
        the output of `load` and the input of `dump`.
        If `None`, the key/attribute will match the name of the field.
        Note: This should only be used for very specific use cases such as
        outputting multiple fields for a single attribute, or using keys/attributes
        that are invalid variable names, unsuitable for field names. In most cases,
        you should use ``data_key`` instead.
    :param validate: Validator or collection of validators that are called
        during deserialization. Validator takes a field's input value as
        its only parameter and returns a boolean.
        If it returns `False`, an :exc:`ValidationError` is raised.
    :param required: Raise a :exc:`ValidationError` if the field value
        is not supplied during deserialization.
    :param allow_none: Set this to `True` if `None` should be considered a valid value during
        validation/deserialization. If ``load_default=None`` and ``allow_none`` is unset,
        will default to ``True``. Otherwise, the default is ``False``.
    :param load_only: If `True` skip this field during serialization, otherwise
        its value will be present in the serialized data.
    :param dump_only: If `True` skip this field during deserialization, otherwise
        its value will be present in the deserialized object. In the context of an
        HTTP API, this effectively marks the field as "read-only".
    :param dict error_messages: Overrides for `Field.default_error_messages`.
    :param metadata: Extra information to be stored as field metadata.

    .. versionchanged:: 2.0.0
        Removed `error` parameter. Use ``error_messages`` instead.

    .. versionchanged:: 2.0.0
        Added `allow_none` parameter, which makes validation/deserialization of `None`
        consistent across fields.

    .. versionchanged:: 2.0.0
        Added `load_only` and `dump_only` parameters, which allow field skipping
        during the (de)serialization process.

    .. versionchanged:: 2.0.0
        Added `missing` parameter, which indicates the value for a field if the field
        is not found during deserialization.

    .. versionchanged:: 2.0.0
        ``default`` value is only used if explicitly set. Otherwise, missing values
        inputs are excluded from serialized output.

    .. versionchanged:: 3.0.0b8
        Add ``data_key`` parameter for the specifying the key in the input and
        output data. This parameter replaced both ``load_from`` and ``dump_to``.
    """
    _CHECK_ATTRIBUTE = True
    default_error_messages = {'required': 'Missing data for required field.', 'null': 'Field may not be null.', 'validator_failed': 'Invalid value.'}

    def __init__(self, *, load_default: typing.Any=missing_, missing: typing.Any=missing_, dump_default: typing.Any=missing_, default: typing.Any=missing_, data_key: str | None=None, attribute: str | None=None, validate: None | typing.Callable[[typing.Any], typing.Any] | typing.Iterable[typing.Callable[[typing.Any], typing.Any]]=None, required: bool=False, allow_none: bool | None=None, load_only: bool=False, dump_only: bool=False, error_messages: dict[str, str] | None=None, metadata: typing.Mapping[str, typing.Any] | None=None, **additional_metadata) -> None:
        if default is not missing_:
            warnings.warn("The 'default' argument to fields is deprecated. Use 'dump_default' instead.", RemovedInMarshmallow4Warning, stacklevel=2)
            if dump_default is missing_:
                dump_default = default
        if missing is not missing_:
            warnings.warn("The 'missing' argument to fields is deprecated. Use 'load_default' instead.", RemovedInMarshmallow4Warning, stacklevel=2)
            if load_default is missing_:
                load_default = missing
        self.dump_default = dump_default
        self.load_default = load_default
        self.attribute = attribute
        self.data_key = data_key
        self.validate = validate
        if validate is None:
            self.validators = []
        elif callable(validate):
            self.validators = [validate]
        elif utils.is_iterable_but_not_string(validate):
            self.validators = list(validate)
        else:
            raise ValueError("The 'validate' parameter must be a callable or a collection of callables.")
        self.allow_none = load_default is None if allow_none is None else allow_none
        self.load_only = load_only
        self.dump_only = dump_only
        if required is True and load_default is not missing_:
            raise ValueError("'load_default' must not be set for required fields.")
        self.required = required
        metadata = metadata or {}
        self.metadata = {**metadata, **additional_metadata}
        if additional_metadata:
            warnings.warn(f'Passing field metadata as keyword arguments is deprecated. Use the explicit `metadata=...` argument instead. Additional metadata: {additional_metadata}', RemovedInMarshmallow4Warning, stacklevel=2)
        messages = {}
        for cls in reversed(self.__class__.__mro__):
            messages.update(getattr(cls, 'default_error_messages', {}))
        messages.update(error_messages or {})
        self.error_messages = messages

    def __repr__(self) -> str:
        return f'<fields.{self.__class__.__name__}(dump_default={self.dump_default!r}, attribute={self.attribute!r}, validate={self.validate}, required={self.required}, load_only={self.load_only}, dump_only={self.dump_only}, load_default={self.load_default}, allow_none={self.allow_none}, error_messages={self.error_messages})>'

    def __deepcopy__(self, memo):
        return copy.copy(self)

    def get_value(self, obj, attr, accessor=None, default=missing_):
        """Return the value for a given key from an object.

        :param object obj: The object to get the value from.
        :param str attr: The attribute/key in `obj` to get the value from.
        :param callable accessor: A callable used to retrieve the value of `attr` from
            the object `obj`. Defaults to `marshmallow.utils.get_value`.
        """
        accessor_func = accessor or utils.get_value
        return accessor_func(obj, attr, default)

    def _validate(self, value):
        """Perform validation on ``value``. Raise a :exc:`ValidationError` if validation
        does not succeed.
        """
        errors = []
        for validator in self.validators:
            try:
                if validator(value) is False:
                    self.fail('validator_failed')
            except ValidationError as error:
                errors.extend(error.messages)
        if errors:
            raise ValidationError(errors)

    def make_error(self, key: str, **kwargs) -> ValidationError:
        """Helper method to make a `ValidationError` with an error message
        from ``self.error_messages``.
        """
        try:
            msg = self.error_messages[key]
        except KeyError:
            class_name = self.__class__.__name__
            msg = MISSING_ERROR_MESSAGE.format(class_name=class_name, key=key)
        if isinstance(msg, str):
            msg = msg.format(**kwargs)
        return ValidationError(msg)

    def fail(self, key: str, **kwargs):
        """Helper method that raises a `ValidationError` with an error message
        from ``self.error_messages``.

        .. deprecated:: 3.0.0
            Use `make_error <marshmallow.fields.Field.make_error>` instead.
        """
        warnings.warn(
            "Field.fail is deprecated. Use Field.make_error instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise self.make_error(key, **kwargs)

    def _validate_missing(self, value):
        """Validate missing values. Raise a :exc:`ValidationError` if
        `value` should be considered missing.
        """
        if value is missing_:
            if self.required:
                raise self.make_error('required')
        elif value is None:
            if self.allow_none is False:
                raise self.make_error('null')

    def serialize(self, attr: str, obj: typing.Any, accessor: typing.Callable[[typing.Any, str, typing.Any], typing.Any] | None=None, **kwargs):
        """Pulls the value for the given key from the object, applies the
        field's formatting and returns the result.

        :param attr: The attribute/key to get from the object.
        :param obj: The object to access the attribute/key from.
        :param accessor: Function used to access values from ``obj``.
        :param kwargs: Field-specific keyword arguments.
        """
        if self.dump_only:
            return self.dump_default

        value = self.get_value(obj, attr, accessor=accessor)
        if value is missing_:
            return self.dump_default if self.dump_default is not missing_ else None

        return self._serialize(value, attr, obj, **kwargs)

    def deserialize(self, value: typing.Any, attr: str | None=None, data: typing.Mapping[str, typing.Any] | None=None, **kwargs):
        """Deserialize ``value``.

        :param value: The value to deserialize.
        :param attr: The attribute/key in `data` to deserialize.
        :param data: The raw input data passed to `Schema.load`.
        :param kwargs: Field-specific keyword arguments.
        :raise ValidationError: If an invalid value is passed or if a required value
            is missing.
        """
        if self.load_only:
            return self.load_default

        self._validate_missing(value)
        if value is missing_:
            return self.load_default

        value = self._deserialize(value, attr, data, **kwargs)
        self._validate(value)
        return value

    def _bind_to_schema(self, field_name, schema):
        """Update field with values from its parent schema. Called by
        :meth:`Schema._bind_field <marshmallow.Schema._bind_field>`.

        :param str field_name: Field name set in schema.
        :param Schema|Field schema: Parent object.
        """
        self.parent = self.schema = schema
        self.name = field_name
        self.root = schema.root
        # Allow fields to override their data key
        if self.data_key is None:
            self.data_key = field_name
        self.metadata.setdefault('data_key', self.data_key)

    def _serialize(self, value: typing.Any, attr: str | None, obj: typing.Any, **kwargs):
        """Serializes ``value`` to a basic Python datatype. Noop by default.
        Concrete :class:`Field` classes should implement this method.

        Example: ::

            class TitleCase(Field):
                def _serialize(self, value, attr, obj, **kwargs):
                    if not value:
                        return ""
                    return str(value).title()

        :param value: The value to be serialized.
        :param str attr: The attribute or key on the object to be serialized.
        :param object obj: The object the value was pulled from.
        :param dict kwargs: Field-specific keyword arguments.
        :return: The serialized value
        """
        return value

    def _deserialize(self, value: typing.Any, attr: str | None, data: typing.Mapping[str, typing.Any] | None, **kwargs):
        """Deserialize value. Concrete :class:`Field` classes should implement this method.

        :param value: The value to be deserialized.
        :param attr: The attribute/key in `data` to be deserialized.
        :param data: The raw input data passed to the `Schema.load`.
        :param kwargs: Field-specific keyword arguments.
        :raise ValidationError: In case of formatting or validation failure.
        :return: The deserialized value.

        .. versionchanged:: 2.0.0
            Added ``attr`` and ``data`` parameters.

        .. versionchanged:: 3.0.0
            Added ``**kwargs`` to signature.
        """
        return value

    @property
    def context(self):
        """The context dictionary for the parent :class:`Schema`."""
        return self.parent.context if self.parent else {}

class Raw(Field):
    """Field that applies no formatting."""

class Nested(Field):
    """Allows you to nest a :class:`Schema <marshmallow.Schema>`
    inside a field.

    Examples: ::

        class ChildSchema(Schema):
            id = fields.Str()
            name = fields.Str()
            # Use lambda functions when you need two-way nesting or self-nesting
            parent = fields.Nested(lambda: ParentSchema(only=("id",)), dump_only=True)
            siblings = fields.List(fields.Nested(lambda: ChildSchema(only=("id", "name"))))


        class ParentSchema(Schema):
            id = fields.Str()
            children = fields.List(
                fields.Nested(ChildSchema(only=("id", "parent", "siblings")))
            )
            spouse = fields.Nested(lambda: ParentSchema(only=("id",)))

    When passing a `Schema <marshmallow.Schema>` instance as the first argument,
    the instance's ``exclude``, ``only``, and ``many`` attributes will be respected.

    Therefore, when passing the ``exclude``, ``only``, or ``many`` arguments to `fields.Nested`,
    you should pass a `Schema <marshmallow.Schema>` class (not an instance) as the first argument.

    ::

        # Yes
        author = fields.Nested(UserSchema, only=("id", "name"))

        # No
        author = fields.Nested(UserSchema(), only=("id", "name"))

    :param nested: `Schema` instance, class, class name (string), dictionary, or callable that
        returns a `Schema` or dictionary. Dictionaries are converted with `Schema.from_dict`.
    :param exclude: A list or tuple of fields to exclude.
    :param only: A list or tuple of fields to marshal. If `None`, all fields are marshalled.
        This parameter takes precedence over ``exclude``.
    :param many: Whether the field is a collection of objects.
    :param unknown: Whether to exclude, include, or raise an error for unknown
        fields in the data. Use `EXCLUDE`, `INCLUDE` or `RAISE`.
    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """
    default_error_messages = {'type': 'Invalid type.'}

    def __init__(self, nested: SchemaABC | type | str | dict[str, Field | type] | typing.Callable[[], SchemaABC | type | dict[str, Field | type]], *, dump_default: typing.Any=missing_, default: typing.Any=missing_, only: types.StrSequenceOrSet | None=None, exclude: types.StrSequenceOrSet=(), many: bool=False, unknown: str | None=None, **kwargs):
        if only is not None and (not is_collection(only)):
            raise StringNotCollectionError('"only" should be a collection of strings.')
        if not is_collection(exclude):
            raise StringNotCollectionError('"exclude" should be a collection of strings.')
        if nested == 'self':
            warnings.warn("Passing 'self' to `Nested` is deprecated. Use `Nested(lambda: MySchema(...))` instead.", RemovedInMarshmallow4Warning, stacklevel=2)
        self.nested = nested
        self.only = only
        self.exclude = exclude
        self.many = many
        self.unknown = unknown
        self._schema = None
        super().__init__(default=default, dump_default=dump_default, **kwargs)

    @property
    def schema(self):
        """The nested Schema object.

        .. versionchanged:: 1.0.0
            Renamed from `serializer` to `schema`.
        """
        if not self._schema:
            if isinstance(self.nested, SchemaABC):
                self._schema = self.nested
            elif isinstance(self.nested, type) and issubclass(self.nested, SchemaABC):
                self._schema = self.nested(many=self.many, only=self.only, exclude=self.exclude,
                                           context=self.context, load_only=self._nested_normalized_option('load_only'),
                                           dump_only=self._nested_normalized_option('dump_only'))
            elif isinstance(self.nested, dict):
                self._schema = Schema.from_dict(self.nested,
                                                many=self.many, only=self.only, exclude=self.exclude,
                                                context=self.context, load_only=self._nested_normalized_option('load_only'),
                                                dump_only=self._nested_normalized_option('dump_only'))
            elif isinstance(self.nested, str):
                if self.nested == 'self':
                    parent_class = self.parent.__class__
                    self._schema = parent_class(many=self.many, only=self.only, exclude=self.exclude,
                                                context=self.context, load_only=self._nested_normalized_option('load_only'),
                                                dump_only=self._nested_normalized_option('dump_only'))
                else:
                    schema_class = class_registry.get_class(self.nested)
                    self._schema = schema_class(many=self.many, only=self.only, exclude=self.exclude,
                                                context=self.context, load_only=self._nested_normalized_option('load_only'),
                                                dump_only=self._nested_normalized_option('dump_only'))
            else:
                raise ValueError("Nested fields must be passed a Schema, not {}.".format(self.nested.__class__))
        return self._schema

    def _deserialize(self, value, attr, data, partial=None, **kwargs):
        """Same as :meth:`Field._deserialize` with additional ``partial`` argument.

        :param bool|tuple partial: For nested schemas, the ``partial``
            parameter passed to `Schema.load`.

        .. versionchanged:: 3.0.0
            Add ``partial`` parameter.
        """
        if self.many and not utils.is_collection(value):
            self.fail('type', input=value, type=value.__class__.__name__)
        return self.schema.load(value, unknown=self.unknown, partial=partial)

class Pluck(Nested):
    """Allows you to replace nested data with one of the data's fields.

    Example: ::

        from marshmallow import Schema, fields


        class ArtistSchema(Schema):
            id = fields.Int()
            name = fields.Str()


        class AlbumSchema(Schema):
            artist = fields.Pluck(ArtistSchema, "id")


        in_data = {"artist": 42}
        loaded = AlbumSchema().load(in_data)  # => {'artist': {'id': 42}}
        dumped = AlbumSchema().dump(loaded)  # => {'artist': 42}

    :param Schema nested: The Schema class or class name (string)
        to nest, or ``"self"`` to nest the :class:`Schema` within itself.
    :param str field_name: The key to pluck a value from.
    :param kwargs: The same keyword arguments that :class:`Nested` receives.
    """

    def __init__(self, nested: SchemaABC | type | str | typing.Callable[[], SchemaABC], field_name: str, **kwargs):
        super().__init__(nested, only=(field_name,), **kwargs)
        self.field_name = field_name

class List(Field):
    """A list field, composed with another `Field` class or
    instance.

    Example: ::

        numbers = fields.List(fields.Float())

    :param cls_or_instance: A field class or instance.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionchanged:: 2.0.0
        The ``allow_none`` parameter now applies to deserialization and
        has the same semantics as the other fields.

    .. versionchanged:: 3.0.0rc9
        Does not serialize scalar values to single-item lists.
    """
    default_error_messages = {'invalid': 'Not a valid list.'}

    def __init__(self, cls_or_instance: Field | type, **kwargs):
        super().__init__(**kwargs)
        try:
            self.inner = resolve_field_instance(cls_or_instance)
        except FieldInstanceResolutionError as error:
            raise ValueError('The list elements must be a subclass or instance of marshmallow.base.FieldABC.') from error
        if isinstance(self.inner, Nested):
            self.only = self.inner.only
            self.exclude = self.inner.exclude

class Tuple(Field):
    """A tuple field, composed of a fixed number of other `Field` classes or
    instances

    Example: ::

        row = Tuple((fields.String(), fields.Integer(), fields.Float()))

    .. note::
        Because of the structured nature of `collections.namedtuple` and
        `typing.NamedTuple`, using a Schema within a Nested field for them is
        more appropriate than using a `Tuple` field.

    :param Iterable[Field] tuple_fields: An iterable of field classes or
        instances.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionadded:: 3.0.0rc4
    """
    default_error_messages = {'invalid': 'Not a valid tuple.'}

    def __init__(self, tuple_fields, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not utils.is_collection(tuple_fields):
            raise ValueError('tuple_fields must be an iterable of Field classes or instances.')
        try:
            self.tuple_fields = [resolve_field_instance(cls_or_instance) for cls_or_instance in tuple_fields]
        except FieldInstanceResolutionError as error:
            raise ValueError('Elements of "tuple_fields" must be subclasses or instances of marshmallow.base.FieldABC.') from error
        self.validate_length = Length(equal=len(self.tuple_fields))

class String(Field):
    """A string field.

    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """
    default_error_messages = {'invalid': 'Not a valid string.', 'invalid_utf8': 'Not a valid utf-8 string.'}

class UUID(String):
    """A UUID field."""
    default_error_messages = {'invalid_uuid': 'Not a valid UUID.'}

    def _validated(self, value) -> uuid.UUID | None:
        """Format the value or raise a :exc:`ValidationError` if an error occurs."""
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        try:
            return uuid.UUID(value)
        except (ValueError, AttributeError, TypeError):
            self.fail('invalid_uuid')

class Number(Field):
    """Base class for number fields.

    :param bool as_string: If `True`, format the serialized value as a string.
    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """
    num_type = float
    default_error_messages = {'invalid': 'Not a valid number.', 'too_large': 'Number too large.'}

    def __init__(self, *, as_string: bool=False, **kwargs):
        self.as_string = as_string
        super().__init__(**kwargs)

    def _format_num(self, value) -> typing.Any:
        """Return the number value for value, given this field's `num_type`."""
        if value is None:
            return None
        return self.num_type(value)

    def _validated(self, value) -> _T | None:
        """Format the value or raise a :exc:`ValidationError` if an error occurs."""
        if value is None:
            return None
        try:
            return self.num_type(value)
        except (TypeError, ValueError) as error:
            self.fail('invalid')

    def _serialize(self, value, attr, obj, **kwargs) -> str | _T | None:
        """Return a string if `self.as_string=True`, otherwise return this field's `num_type`."""
        if value is None:
            return None
        num = self._format_num(value)
        if self.as_string:
            return str(num)
        return num

class Integer(Number):
    """An integer field.

    :param strict: If `True`, only integer types are valid.
        Otherwise, any value castable to `int` is valid.
    :param kwargs: The same keyword arguments that :class:`Number` receives.
    """
    num_type = int
    default_error_messages = {'invalid': 'Not a valid integer.'}

    def __init__(self, *, strict: bool=False, **kwargs):
        self.strict = strict
        super().__init__(**kwargs)

class Float(Number):
    """A double as an IEEE-754 double precision string.

    :param bool allow_nan: If `True`, `NaN`, `Infinity` and `-Infinity` are allowed,
        even though they are illegal according to the JSON specification.
    :param bool as_string: If `True`, format the value as a string.
    :param kwargs: The same keyword arguments that :class:`Number` receives.
    """
    num_type = float
    default_error_messages = {'special': 'Special numeric values (nan or infinity) are not permitted.'}

    def __init__(self, *, allow_nan: bool=False, as_string: bool=False, **kwargs):
        self.allow_nan = allow_nan
        super().__init__(as_string=as_string, **kwargs)

class Decimal(Number):
    """A field that (de)serializes to the Python ``decimal.Decimal`` type.
    It's safe to use when dealing with money values, percentages, ratios
    or other numbers where precision is critical.

    .. warning::

        This field serializes to a `decimal.Decimal` object by default. If you need
        to render your data as JSON, keep in mind that the `json` module from the
        standard library does not encode `decimal.Decimal`. Therefore, you must use
        a JSON library that can handle decimals, such as `simplejson`, or serialize
        to a string by passing ``as_string=True``.

    .. warning::

        If a JSON `float` value is passed to this field for deserialization it will
        first be cast to its corresponding `string` value before being deserialized
        to a `decimal.Decimal` object. The default `__str__` implementation of the
        built-in Python `float` type may apply a destructive transformation upon
        its input data and therefore cannot be relied upon to preserve precision.
        To avoid this, you can instead pass a JSON `string` to be deserialized
        directly.

    :param places: How many decimal places to quantize the value. If `None`, does
        not quantize the value.
    :param rounding: How to round the value during quantize, for example
        `decimal.ROUND_UP`. If `None`, uses the rounding value from
        the current thread's context.
    :param allow_nan: If `True`, `NaN`, `Infinity` and `-Infinity` are allowed,
        even though they are illegal according to the JSON specification.
    :param as_string: If `True`, serialize to a string instead of a Python
        `decimal.Decimal` type.
    :param kwargs: The same keyword arguments that :class:`Number` receives.

    .. versionadded:: 1.2.0
    """
    num_type = decimal.Decimal
    default_error_messages = {'special': 'Special numeric values (nan or infinity) are not permitted.'}

    def __init__(self, places: int | None=None, rounding: str | None=None, *, allow_nan: bool=False, as_string: bool=False, **kwargs):
        self.places = decimal.Decimal((0, (1,), -places)) if places is not None else None
        self.rounding = rounding
        self.allow_nan = allow_nan
        super().__init__(as_string=as_string, **kwargs)
        if self.places is not None and 'precision' in kwargs:
            raise ValueError("Cannot specify both `places` and `precision`.")
        self.precision = kwargs.get('precision', None)

class Boolean(Field):
    """A boolean field.

    :param truthy: Values that will (de)serialize to `True`. If an empty
        set, any non-falsy value will deserialize to `True`. If `None`,
        `marshmallow.fields.Boolean.truthy` will be used.
    :param falsy: Values that will (de)serialize to `False`. If `None`,
        `marshmallow.fields.Boolean.falsy` will be used.
    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """
    truthy = {'t', 'T', 'true', 'True', 'TRUE', 'on', 'On', 'ON', 'y', 'Y', 'yes', 'Yes', 'YES', '1', 1}
    falsy = {'f', 'F', 'false', 'False', 'FALSE', 'off', 'Off', 'OFF', 'n', 'N', 'no', 'No', 'NO', '0', 0}
    default_error_messages = {'invalid': 'Not a valid boolean.'}

    def __init__(self, *, truthy: set | None=None, falsy: set | None=None, **kwargs):
        super().__init__(**kwargs)
        if truthy is not None:
            self.truthy = set(truthy)
        if falsy is not None:
            self.falsy = set(falsy)
        if self.load_default is missing_:
            self.load_default = False

class DateTime(Field):
    """A formatted datetime string.

    Example: ``'2014-12-22T03:12:58.019077+00:00'``

    :param format: Either ``"rfc"`` (for RFC822), ``"iso"`` (for ISO8601),
        ``"timestamp"``, ``"timestamp_ms"`` (for a POSIX timestamp) or a date format string.
        If `None`, defaults to "iso".
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionchanged:: 3.0.0rc9
        Does not modify timezone information on (de)serialization.
    .. versionchanged:: 3.19
        Add timestamp as a format.
    """
    SERIALIZATION_FUNCS = {'iso': utils.isoformat, 'iso8601': utils.isoformat, 'rfc': utils.rfcformat, 'rfc822': utils.rfcformat, 'timestamp': utils.timestamp, 'timestamp_ms': utils.timestamp_ms}
    DESERIALIZATION_FUNCS = {'iso': utils.from_iso_datetime, 'iso8601': utils.from_iso_datetime, 'rfc': utils.from_rfc, 'rfc822': utils.from_rfc, 'timestamp': utils.from_timestamp, 'timestamp_ms': utils.from_timestamp_ms}
    DEFAULT_FORMAT = 'iso'
    OBJ_TYPE = 'datetime'
    SCHEMA_OPTS_VAR_NAME = 'datetimeformat'
    default_error_messages = {'invalid': 'Not a valid {obj_type}.', 'invalid_awareness': 'Not a valid {awareness} {obj_type}.', 'format': '"{input}" cannot be formatted as a {obj_type}.'}

    def __init__(self, format: str | None=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.format = format or self.DEFAULT_FORMAT
        if self.format not in self.SERIALIZATION_FUNCS:
            raise ValueError(
                f"Invalid format '{self.format}'. Supported formats are: "
                f"{', '.join(self.SERIALIZATION_FUNCS.keys())}"
            )

class NaiveDateTime(DateTime):
    """A formatted naive datetime string.

    :param format: See :class:`DateTime`.
    :param timezone: Used on deserialization. If `None`,
        aware datetimes are rejected. If not `None`, aware datetimes are
        converted to this timezone before their timezone information is
        removed.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionadded:: 3.0.0rc9
    """
    AWARENESS = 'naive'

    def __init__(self, format: str | None=None, *, timezone: dt.timezone | None=None, **kwargs) -> None:
        super().__init__(format=format, **kwargs)
        self.timezone = timezone

class AwareDateTime(DateTime):
    """A formatted aware datetime string.

    :param format: See :class:`DateTime`.
    :param default_timezone: Used on deserialization. If `None`, naive
        datetimes are rejected. If not `None`, naive datetimes are set this
        timezone.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. versionadded:: 3.0.0rc9
    """
    AWARENESS = 'aware'

    def __init__(self, format: str | None=None, *, default_timezone: dt.tzinfo | None=None, **kwargs) -> None:
        super().__init__(format=format, **kwargs)
        self.default_timezone = default_timezone

class Time(DateTime):
    """A formatted time string.

    Example: ``'03:12:58.019077'``

    :param format: Either ``"iso"`` (for ISO8601) or a date format string.
        If `None`, defaults to "iso".
    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """
    SERIALIZATION_FUNCS = {'iso': utils.to_iso_time, 'iso8601': utils.to_iso_time}
    DESERIALIZATION_FUNCS = {'iso': utils.from_iso_time, 'iso8601': utils.from_iso_time}
    DEFAULT_FORMAT = 'iso'
    OBJ_TYPE = 'time'
    SCHEMA_OPTS_VAR_NAME = 'timeformat'

class Date(DateTime):
    """ISO8601-formatted date string.

    :param format: Either ``"iso"`` (for ISO8601) or a date format string.
        If `None`, defaults to "iso".
    :param kwargs: The same keyword arguments that :class:`Field` receives.
    """
    default_error_messages = {'invalid': 'Not a valid date.', 'format': '"{input}" cannot be formatted as a date.'}
    SERIALIZATION_FUNCS = {'iso': utils.to_iso_date, 'iso8601': utils.to_iso_date}
    DESERIALIZATION_FUNCS = {'iso': utils.from_iso_date, 'iso8601': utils.from_iso_date}
    DEFAULT_FORMAT = 'iso'
    OBJ_TYPE = 'date'
    SCHEMA_OPTS_VAR_NAME = 'dateformat'

class TimeDelta(Field):
    """A field that (de)serializes a :class:`datetime.timedelta` object to an
    integer or float and vice versa. The integer or float can represent the
    number of days, seconds or microseconds.

    :param precision: Influences how the integer or float is interpreted during
        (de)serialization. Must be 'days', 'seconds', 'microseconds',
        'milliseconds', 'minutes', 'hours' or 'weeks'.
    :param serialization_type: Whether to (de)serialize to a `int` or `float`.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    Integer Caveats
    ---------------
    Any fractional parts (which depends on the precision used) will be truncated
    when serializing using `int`.

    Float Caveats
    -------------
    Use of `float` when (de)serializing may result in data precision loss due
    to the way machines handle floating point values.

    Regardless of the precision chosen, the fractional part when using `float`
    will always be truncated to microseconds.
    For example, `1.12345` interpreted as microseconds will result in `timedelta(microseconds=1)`.

    .. versionchanged:: 2.0.0
        Always serializes to an integer value to avoid rounding errors.
        Add `precision` parameter.
    .. versionchanged:: 3.17.0
        Allow (de)serialization to `float` through use of a new `serialization_type` parameter.
        `int` is the default to retain previous behaviour.
    """
    DAYS = 'days'
    SECONDS = 'seconds'
    MICROSECONDS = 'microseconds'
    MILLISECONDS = 'milliseconds'
    MINUTES = 'minutes'
    HOURS = 'hours'
    WEEKS = 'weeks'
    default_error_messages = {'invalid': 'Not a valid period of time.', 'format': '{input!r} cannot be formatted as a timedelta.'}

    def __init__(self, precision: str=SECONDS, serialization_type: type[int | float]=int, **kwargs):
        precision = precision.lower()
        units = (self.DAYS, self.SECONDS, self.MICROSECONDS, self.MILLISECONDS, self.MINUTES, self.HOURS, self.WEEKS)
        if precision not in units:
            msg = 'The precision must be {} or "{}".'.format(', '.join([f'"{each}"' for each in units[:-1]]), units[-1])
            raise ValueError(msg)
        if serialization_type not in (int, float):
            raise ValueError('The serialization type must be one of int or float')
        self.precision = precision
        self.serialization_type = serialization_type
        super().__init__(**kwargs)

class Mapping(Field):
    """An abstract class for objects with key-value pairs.

    :param keys: A field class or instance for dict keys.
    :param values: A field class or instance for dict values.
    :param kwargs: The same keyword arguments that :class:`Field` receives.

    .. note::
        When the structure of nested data is not known, you may omit the
        `keys` and `values` arguments to prevent content validation.

    .. versionadded:: 3.0.0rc4
    """
    mapping_type = dict
    default_error_messages = {'invalid': 'Not a valid mapping type.'}

    def __init__(self, keys: Field | type | None=None, values: Field | type | None=None, **kwargs):
        super().__init__(**kwargs)
        if keys is None:
            self.key_field = None
        else:
            try:
                self.key_field = resolve_field_instance(keys)
            except FieldInstanceResolutionError as error:
                raise ValueError('"keys" must be a subclass or instance of marshmallow.base.FieldABC.') from error
        if values is None:
            self.value_field = None
        else:
            try:
                self.value_field = resolve_field_instance(values)
            except FieldInstanceResolutionError as error:
                raise ValueError('"values" must be a subclass or instance of marshmallow.base.FieldABC.') from error
            if isinstance(self.value_field, Nested):
                self.only = self.value_field.only
                self.exclude = self.value_field.exclude

class Dict(Mapping):
    """A dict field. Supports dicts and dict-like objects. Extends
    Mapping with dict as the mapping_type.

    Example: ::

        numbers = fields.Dict(keys=fields.Str(), values=fields.Float())

    :param kwargs: The same keyword arguments that :class:`Mapping` receives.

    .. versionadded:: 2.1.0
    """
    mapping_type = dict

class Url(String):
    """An URL field.

    :param default: Default value for the field if the attribute is not set.
    :param relative: Whether to allow relative URLs.
    :param require_tld: Whether to reject non-FQDN hostnames.
    :param schemes: Valid schemes. By default, ``http``, ``https``,
        ``ftp``, and ``ftps`` are allowed.
    :param kwargs: The same keyword arguments that :class:`String` receives.
    """
    default_error_messages = {'invalid': 'Not a valid URL.'}

    def __init__(self, *, relative: bool=False, absolute: bool=True, schemes: types.StrSequenceOrSet | None=None, require_tld: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.relative = relative
        self.absolute = absolute
        self.require_tld = require_tld
        validator = validate.URL(relative=self.relative, absolute=self.absolute, schemes=schemes, require_tld=self.require_tld, error=self.error_messages['invalid'])
        self.validators.insert(0, validator)

class Email(String):
    """An email field.

    :param args: The same positional arguments that :class:`String` receives.
    :param kwargs: The same keyword arguments that :class:`String` receives.
    """
    default_error_messages = {'invalid': 'Not a valid email address.'}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        validator = validate.Email(error=self.error_messages['invalid'])
        self.validators.insert(0, validator)

class IP(Field):
    """A IP address field.

    :param bool exploded: If `True`, serialize ipv6 address in long form, ie. with groups
        consisting entirely of zeros included.

    .. versionadded:: 3.8.0
    """
    default_error_messages = {'invalid_ip': 'Not a valid IP address.'}
    DESERIALIZATION_CLASS = None

    def __init__(self, *args, exploded=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploded = exploded

class IPv4(IP):
    """A IPv4 address field.

    .. versionadded:: 3.8.0
    """
    default_error_messages = {'invalid_ip': 'Not a valid IPv4 address.'}
    DESERIALIZATION_CLASS = ipaddress.IPv4Address

class IPv6(IP):
    """A IPv6 address field.

    .. versionadded:: 3.8.0
    """
    default_error_messages = {'invalid_ip': 'Not a valid IPv6 address.'}
    DESERIALIZATION_CLASS = ipaddress.IPv6Address

class IPInterface(Field):
    """A IPInterface field.

    IP interface is the non-strict form of the IPNetwork type where arbitrary host
    addresses are always accepted.

    IPAddress and mask e.g. '192.168.0.2/24' or '192.168.0.2/255.255.255.0'

    see https://python.readthedocs.io/en/latest/library/ipaddress.html#interface-objects

    :param bool exploded: If `True`, serialize ipv6 interface in long form, ie. with groups
        consisting entirely of zeros included.
    """
    default_error_messages = {'invalid_ip_interface': 'Not a valid IP interface.'}
    DESERIALIZATION_CLASS = None

    def __init__(self, *args, exploded: bool=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploded = exploded

class IPv4Interface(IPInterface):
    """A IPv4 Network Interface field."""
    default_error_messages = {'invalid_ip_interface': 'Not a valid IPv4 interface.'}
    DESERIALIZATION_CLASS = ipaddress.IPv4Interface

class IPv6Interface(IPInterface):
    """A IPv6 Network Interface field."""
    default_error_messages = {'invalid_ip_interface': 'Not a valid IPv6 interface.'}
    DESERIALIZATION_CLASS = ipaddress.IPv6Interface

class Enum(Field):
    """An Enum field (de)serializing enum members by symbol (name) or by value.

    :param enum Enum: Enum class
    :param boolean|Schema|Field by_value: Whether to (de)serialize by value or by name,
        or Field class or instance to use to (de)serialize by value. Defaults to False.

    If `by_value` is `False` (default), enum members are (de)serialized by symbol (name).
    If it is `True`, they are (de)serialized by value using :class:`Field`.
    If it is a field instance or class, they are (de)serialized by value using this field.

    .. versionadded:: 3.18.0
    """
    default_error_messages = {'unknown': 'Must be one of: {choices}.'}

    def __init__(self, enum: type[EnumType], *, by_value: bool | Field | type=False, **kwargs):
        super().__init__(**kwargs)
        self.enum = enum
        self.by_value = by_value
        if by_value is False:
            self.field: Field = String()
            self.choices_text = ', '.join((str(self.field._serialize(m, None, None)) for m in enum.__members__))
        else:
            if by_value is True:
                self.field = Field()
            else:
                try:
                    self.field = resolve_field_instance(by_value)
                except FieldInstanceResolutionError as error:
                    raise ValueError('"by_value" must be either a bool or a subclass or instance of marshmallow.base.FieldABC.') from error
            self.choices_text = ', '.join((str(self.field._serialize(m.value, None, None)) for m in enum))

class Method(Field):
    """A field that takes the value returned by a `Schema` method.

    :param str serialize: The name of the Schema method from which
        to retrieve the value. The method must take an argument ``obj``
        (in addition to self) that is the object to be serialized.
    :param str deserialize: Optional name of the Schema method for deserializing
        a value The method must take a single argument ``value``, which is the
        value to deserialize.

    .. versionchanged:: 2.0.0
        Removed optional ``context`` parameter on methods. Use ``self.context`` instead.

    .. versionchanged:: 2.3.0
        Deprecated ``method_name`` parameter in favor of ``serialize`` and allow
        ``serialize`` to not be passed at all.

    .. versionchanged:: 3.0.0
        Removed ``method_name`` parameter.
    """
    _CHECK_ATTRIBUTE = False

    def __init__(self, serialize: str | None=None, deserialize: str | None=None, **kwargs):
        kwargs['dump_only'] = bool(serialize) and (not bool(deserialize))
        kwargs['load_only'] = bool(deserialize) and (not bool(serialize))
        super().__init__(**kwargs)
        self.serialize_method_name = serialize
        self.deserialize_method_name = deserialize
        self._serialize_method = None
        self._deserialize_method = None

class Function(Field):
    """A field that takes the value returned by a function.

    :param serialize: A callable from which to retrieve the value.
        The function must take a single argument ``obj`` which is the object
        to be serialized. It can also optionally take a ``context`` argument,
        which is a dictionary of context variables passed to the serializer.
        If no callable is provided then the ```load_only``` flag will be set
        to True.
    :param deserialize: A callable from which to retrieve the value.
        The function must take a single argument ``value`` which is the value
        to be deserialized. It can also optionally take a ``context`` argument,
        which is a dictionary of context variables passed to the deserializer.
        If no callable is provided then ```value``` will be passed through
        unchanged.

    .. versionchanged:: 2.3.0
        Deprecated ``func`` parameter in favor of ``serialize``.

    .. versionchanged:: 3.0.0a1
        Removed ``func`` parameter.
    """
    _CHECK_ATTRIBUTE = False

    def __init__(self, serialize: None | typing.Callable[[typing.Any], typing.Any] | typing.Callable[[typing.Any, dict], typing.Any]=None, deserialize: None | typing.Callable[[typing.Any], typing.Any] | typing.Callable[[typing.Any, dict], typing.Any]=None, **kwargs):
        kwargs['dump_only'] = bool(serialize) and (not bool(deserialize))
        kwargs['load_only'] = bool(deserialize) and (not bool(serialize))
        super().__init__(**kwargs)
        self.serialize_func = serialize and utils.callable_or_raise(serialize)
        self.deserialize_func = deserialize and utils.callable_or_raise(deserialize)

class Constant(Field):
    """A field that (de)serializes to a preset constant.  If you only want the
    constant added for serialization or deserialization, you should use
    ``dump_only=True`` or ``load_only=True`` respectively.

    :param constant: The constant to return for the field attribute.

    .. versionadded:: 2.0.0
    """
    _CHECK_ATTRIBUTE = False

    def __init__(self, constant: typing.Any, **kwargs):
        super().__init__(**kwargs)
        self.constant = constant
        self.load_default = constant
        self.dump_default = constant

class Inferred(Field):
    """A field that infers how to serialize, based on the value type.

    .. warning::

        This class is treated as private API.
        Users should not need to use this class directly.
    """

    def __init__(self):
        super().__init__()
        self._field_cache = {}
URL = Url
Str = String
Bool = Boolean
Int = Integer
