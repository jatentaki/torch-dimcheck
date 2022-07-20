import re
import inspect
import functools
from dataclasses import dataclass, field
from typing import (
    Union,
    Optional,
    Tuple,
    Dict,
    Set,
    List,
    OrderedDict,
    Any,
    get_origin,
)

@dataclass
class ConstError:
    tensor_name: str
    expected: int
    found: int
    
    def __str__(self) -> str:
        return f'{self.tensor_name}: {self.expected} != {self.found}'

@dataclass
class Inconsistency:
    label: str
    values: Set[int]
    
    def __str__(self) -> str:
        return f'Inconsistency: {self.label} = {list(self.values)}'

@dataclass
class EmptyPlusWildcard:
    label: str

    def __str__(self) -> str:
        return f'Label {self.label} requires at least 1 dimension, got 0'

class DimcheckError(TypeError):
    def __init__(self, text: str):
        self.text = text

    def set_func_data(self, func):
        if not hasattr(self, 'function_name'):
            self.function_name = get_function_name(func)

        if not hasattr(self, 'class_name'):
            cls = get_defining_class(func)
            if cls is not None:
                self.class_name = cls.__name__
            else:
                self.class_name = None

    def render_context(self) -> str:
        if not hasattr(self, 'class_name'):
            offender = '<unknown>'
        elif self.class_name is None:
            return f'Error in function `{self.function_name}`'
        else:
            return (f'Error in method `{self.class_name}.{self.function_name}`')

    def __str__(self):
        context = self.render_context()
        return f'{context}: {self.text}'

class ParseError(DimcheckError):
    def __init__(self, text: str):
        super(ParseError, self).__init__(text)
        self.tensor_name = None

    def render_context(self) -> str:
        if not hasattr(self, 'class_name'):
            offender = '<unknown>'
        elif self.class_name is None:
            offender = f'function `{self.function_name}`'
        else:
            offender = f'method `{self.class_name}.{self.function_name}`'

        return (f'Error parsing argument `{self.tensor_name}` of {offender}: '
                f'{self.text}.')

    __str__ = render_context
        
class ShapeError(DimcheckError):
    def __init__(
        self,
        issues: List[Union[ConstError, Inconsistency]],
        context: List[str],
    ):
        self.issues = issues
        self.context = context
    
    def __str__(self) -> str:
        func_data = self.render_context()
        issues = '\n\t'.join(str(i) for i in self.issues)
        context = '\n\t'.join(self.context)
        return f'{func_data}\nIssues:\n\t{issues}\nContext:\n\t{context}'

LABEL_RE = re.compile('[a-zA-Z]([a-zA-Z]|\d)*')
FIXED_RE = re.compile('\d+')
WCARD_PLUS_RE = re.compile('([a-zA-Z]([a-zA-Z]|\d)*)?\+')
WCARD_STAR_RE = re.compile('([a-zA-Z]([a-zA-Z]|\d)*)?\*')

@dataclass(unsafe_hash=True)
class Token:
    label: Union[int, str]
    
    @classmethod
    def from_str(cls, string: str) -> 'Token':
        match = FIXED_RE.match(string)
        if match:
            return cls(int(match.group(0)))

        match = LABEL_RE.match(string)
        if match:
            return cls(string)

        match = WCARD_PLUS_RE.match(string)
        if match:
            return cls(string)

        match = WCARD_STAR_RE.match(string)
        if match:
            return cls(string)

        raise DimcheckError(f'`{string}` is not a valid token')

    @property
    def is_star(self):
        return isinstance(self.label, str) and '*' in self.label

    @property
    def is_plus(self):
        return isinstance(self.label, str) and '+' in self.label

    @property
    def is_wildcard(self):
        return self.is_star or self.is_plus

    @classmethod
    def tokenize(cls, annotation: str) -> Tuple['Token']:
        tokens = tuple(cls.from_str(s) for s in annotation.split())
        # there can be at most one wildcard
        if sum((1 if t.is_wildcard else 0) for t in tokens) > 1:
            raise DimcheckError(f'Annotation `{annotation}` cannot '
                             'contain more than 1 wildcard.')
        return tokens

ParseDict = Dict[Union[str, int], Union[int, Tuple[int]]]

@dataclass(unsafe_hash=True)
class A:
    raw: str
    tokens: Tuple[Token]
    
    def __call__(self):
        # this is just to fake the callable interface for typing
        return NotImplemented()
    
    @classmethod
    def __class_getitem__(cls, annotation: str) -> 'A':
        return A(annotation, Token.tokenize(annotation))

    def __repr__(self) -> str:
        return f'A[{self.raw}]'

    def parse_shape(self, shape: Tuple[int]) -> ParseDict:
        parse_dict: ParseDict = {}
            
        n_shape = len(shape)
        n_token = len(self.tokens)
        has_wild = any(t.is_wildcard for t in self.tokens)

        if has_wild and (n_shape < n_token - 1):
            raise ParseError(f'Got shape {shape} with an annotation {self.raw}')

        if not has_wild and n_shape != n_token:
            raise ParseError(f'Got shape of length {len(shape)} with an '
                            f'annotation of length {(len(self.tokens))} '
                            f'({self.raw}) vs ({shape}).')
        
        if not has_wild:
            # no wildcards -> one token corresponds to one dim -> just zip together
            for dim, token in zip(shape, self.tokens):
                parse_dict[token.label] = dim
        else:
            # one wildcard present
            s_t, s_s = 0, 0
            e_t, e_s = n_token-1, n_shape-1

            # associate tokens with dims from the front
            while not self.tokens[s_t].is_wildcard:
                parse_dict[self.tokens[s_t].label] = shape[s_s]
                s_t += 1
                s_s += 1

            # then from the back
            while not self.tokens[e_t].is_wildcard:
                parse_dict[self.tokens[e_t].label] = shape[e_s]
                e_t -= 1
                e_s -= 1

            assert s_t == e_t
            # the remainder belongs to the wildcard
            parse_dict[self.tokens[s_t].label] = shape[s_s:e_s+1]
        
        return parse_dict
    
    def render(self, parse_dict: ParseDict) -> str:
        binds = []
        for token in self.tokens:
            value = parse_dict.get(token.label, '?')
            binds.append(f'{token.label}={value}')
        return ' '.join(binds)

def get_function_name(func):
    if isinstance(func, functools.partial):
        return get_function_name(func.func)
    return func.__name__

def get_defining_class(meth):
    '''taken from https://stackoverflow.com/questions/3589311 '''
    if isinstance(meth, functools.partial):
        return get_defining_class(meth.func)
    if inspect.ismethod(meth) \
        or (inspect.isbuiltin(meth) \
            and getattr(meth, '__self__', None) is not None \
            and getattr(meth.__self__, '__class__', None
           )):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, '__func__', meth)  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                      None)
        if isinstance(cls, type):
            return cls
    return getattr(meth, '__objclass__', None)  # handle special descriptor objects

def check_consistency(parses: Dict[str, ParseDict]) -> Optional[ShapeError]:
    issues = []
    
    bindings: Dict[str, Set[int]] = {}
        
    for tensor_name, tensor_parses in parses.items():
        for label, value in tensor_parses.items():
            if isinstance(label, int):
                if label != value:
                    issues.append(ConstError(
                        tensor_name=tensor_name,
                        expected=label,
                        found=value,
                    ))
            elif isinstance(label, str):
                bindings.setdefault(label, set()).add(value)

                if '+' in label and len(value) == 0:
                    issues.append(EmptyPlusWildcard(label))

            else:
                raise AssertionError('Unreachable')
    
    for label, values in bindings.items():
        if label == '...':
            continue
        if len(values) > 1:
            issues.append(Inconsistency(label=label, values=values))
        else:
            assert len(values) == 1
    
    if not issues:
        return None
    
    return ShapeError(issues, context=[])
    
def _is_optional_annotation(type_) -> bool:
    return hasattr(type_, '__origin__') \
        and type_.__origin__ == Union \
        and len(type_.__args__) == 2 \
        and type_.__args__[1] == type(None) \
        and isinstance(type_.__args__[0], A)

def get_shape(tensorlike, name):
    if not hasattr(tensorlike, 'shape'):
        raise DimcheckError(f'Expected {name} to have a an attribute `shape` '
                            f'(type({name})={type(tensorlike).__name__}).')

    try:
        shape = tuple(tensorlike.shape)
    except TypeError:
        raise DimcheckError(f'Expected {name}.shape to return a tuple of int '
                            f'but {name}.shape is not iterable '
                            f'(type({name})={type(tensorlike).__name__}).')
    
    if not all(isinstance(dim, int) for dim in shape):
        types = ', '.join(type(e).__name__ for e in shape)
        raise DimcheckError(f'Expected {name}.shape to return a tuple of int '
                            f'but the result is a tuple of {types} '
                            f'(type({name})={type(tensorlike).__name__}.')

    return shape

@dataclass
class CheckerState:
    parses: Dict[str, ParseDict] = field(default_factory=dict)
    annotations: Dict[str, A] = field(default_factory=dict)

    def update(self, name, value, annotation) -> None:
        if _is_optional_annotation(annotation):
            if value is None:
                return
            else:
                annotation = annotation.__args__[0]

        if not isinstance(annotation, A):
            return
        
        shape = get_shape(value, name) 
        try:
            self.parses[name] = annotation.parse_shape(shape)
        except ParseError as e:
            e.tensor_name = name
            raise e

        self.annotations[name] = annotation

    def check(self) -> Optional[ShapeError]:
        maybe_error = check_consistency(self.parses)
        if maybe_error is not None:
            for name, annotation in self.annotations.items():
                rendered = annotation.render(self.parses[name])
                maybe_error.context.append(f'{name}: {rendered}')
        return maybe_error
 
@dataclass
class Signature:
    args: OrderedDict[str, A]
    returns: Optional[List[Optional[A]]]

    @staticmethod
    def _to_A(annotation):
        ''' Convert strings to A, throw on non-types '''

        if isinstance(annotation, str):
            return A[annotation]
        if isinstance(annotation, A):
            return annotation

        # unwrap decorated objects. this is important for dataclasses which
        # have themselves been wrapped in @dimchecked. Without this, they'd
        # be treated as a function object and thus rejected
        if hasattr(annotation, '__wrapped__'):
            annotation = annotation.__wrapped__

        if isinstance(annotation, type) or \
            getattr(annotation, '__module__', None) == 'typing' \
            or annotation is None:
           return annotation

        raise TypeError(f'Annotations used with @dimchecked can only '
                        f'be types, strings, A, None or std::typing '
                        f'objects, found {annotation=} '
                        f'({type(annotation)=}).')

    @classmethod
    def from_func(cls, func):
        sig = inspect.signature(func)

        args = OrderedDict()
        for parameter in sig.parameters.values():
            args[parameter.name] = cls._to_A(parameter.annotation)

        if sig.return_annotation in (inspect.Signature.empty, Any):
            return cls(args, None)

        returns = []
        if get_origin(sig.return_annotation) == tuple: # typing.Tuple
            return_annotation = sig.return_annotation.__args__
        elif not isinstance(sig.return_annotation, tuple): # regularize a singular item -> 1-tuple
            return_annotation = (sig.return_annotation, )
        else: # an actual tuple
            return_annotation = sig.return_annotation

        for subannotation in return_annotation:
            returns.append(cls._to_A(subannotation))

        return cls(args, returns)

    def zip_args(self, args: List[Any], kwargs: Dict[str, Any]):
        for value, (name, annotation) in zip(args, self.args.items()):
            yield name, value, annotation

        for key, value in kwargs.items():
            if key not in self.args:
                continue

            yield key, value, self.args[key]

    def zip_returns(self, returns: List[Any]):
        if self.returns is None:
            return

        if len(self.returns) != len(returns):
            raise DimcheckError(f'Return should have {len(self.returns)} '
                            f'elements, found {len(returns)}.')

        for i, (value, annotation) in enumerate(zip(returns, self.returns)):
            yield f'<return {i}>', value, annotation

def dimchecked(wrapped):
    if isinstance(wrapped, type):
        if issubclass(wrapped, tuple) and hasattr(wrapped, '_asdict'):
            # looks like a NamedTuple, which doesn't use __init__
            wrapped.__new__ = dimchecked(wrapped.__new__)
            return wrapped

        # this is the case for dataclasses
        wrapped.__init__ = dimchecked(wrapped.__init__)
        return wrapped

    signature = Signature.from_func(wrapped)
    
    @functools.wraps(wrapped)
    def wrapper(*args, **kwargs):
        try:
            checker_state = CheckerState()
            
            for name, value, annotation in signature.zip_args(args, kwargs):
                checker_state.update(name, value, annotation)
            
            maybe_error = checker_state.check()
            if maybe_error is not None:
                raise maybe_error

            result = wrapped(*args, **kwargs)

            if isinstance(result, (list, tuple)) and \
               not hasattr(result, '_asdict'): # don't unpack NamedTuple
                tupled_result = result
            else:
                tupled_result = (result, )

            for name, value, annotation in signature.zip_returns(tupled_result):
                checker_state.update(name, value, annotation)

            maybe_error = checker_state.check()
            if maybe_error is not None:
                raise maybe_error

            return result
        except DimcheckError as e:
            e.set_func_data(wrapped)
            raise e

    return wrapper
