import re
import inspect
import torch
from dataclasses import dataclass, field
from typing import Union, Optional, Tuple, Tuple, Dict, Set, List

LABEL_RE = re.compile('[a-zA-Z]([a-zA-Z]|\d)*')
FIXED_RE = re.compile('\d+')
WCARD_RE = re.compile('[a-zA-Z]([a-zA-Z]|\d)*\.\.\.')

@dataclass(unsafe_hash=True)
class Token:
    label: Union[int, str]
    
    @classmethod
    def from_str(cls, string: str) -> 'Token':
        if match := FIXED_RE.match(string):
            return cls(int(match.group(0)))
        elif match := LABEL_RE.match(string):
            return cls(string)
        elif match := WCARD_RE.match(string):
            return cls(string)
        else:
            raise TypeError(f'`{string}` is not a valid token')

    @property
    def is_wildcard(self):
        return isinstance(self.label, str) and '...' in self.label 

    @classmethod
    def tokenize(cls, annotation: str) -> Tuple['Token']:
        return tuple(cls.from_str(s) for s in annotation.split())

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

    def parse_shape(self, shape: Tuple[int]) -> ParseDict:
        parse_dict: ParseDict = {}
            
        if len(shape) != len(self.tokens):
            raise TypeError(f'Got shape of length {len(shape)} with an '
                            f'annotation of length {(len(self.tokens))} '
                            f'({self.raw}) vs ({shape}).')
        
        for dim, token in zip(shape, self.tokens):
            parse_dict[token.label] = dim
        
        return parse_dict
    
    def render(self, parse_dict: ParseDict) -> str:
        binds = []
        for token in self.tokens:
            value = parse_dict.get(token.label, '?')
            binds.append(f'{token.label}={value}')
        return ' '.join(binds)

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

class ShapeError(TypeError):
    def __init__(self, issues: List[Union[ConstError, Inconsistency]], context: List[str]):
        self.issues = issues
        self.context = context
    
    def __str__(self) -> str:
        issues = '\n\t'.join(str(i) for i in self.issues)
        context = '\n\t'.join(self.context)
        return f'Issues:\n\t{issues}\nContext:\n\t{context}'

def check_consistency(parses: Dict[str, ParseDict]) -> Optional[ShapeError]:
    issues = []
    
    bindings: Dict[str, Set[int]] = {}
        
    for tensor_name, tensor_parses in parses.items():
        for label, value in tensor_parses.items():
            if isinstance(label, int):
                if label != value:
                    issues.append(ConstError(tensor_name=tensor_name, expected=label, found=value))
            elif isinstance(label, str):
                bindings.setdefault(label, set()).add(value)
            else:
                raise AssertionError('Unreachable')
    
    for label, values in bindings.items():
        if len(values) > 1:
            issues.append(Inconsistency(label=label, values=values))
        else:
            assert len(values) == 1
    
    if not issues:
        return None
    
    return ShapeError(issues, context=[])
    
def _zip_args_and_labels(args, signature):
    for arg, parameter in zip(args, signature.parameters.values()):
        yield parameter.name, arg, parameter.annotation

def _is_optional_annotation(type_) -> bool:
    return hasattr(type_, '__origin__') \
        and type_.__origin__ == Union \
        and len(type_.__args__) == 2 \
        and type_.__args__[1] == type(None) \
        and isinstance(type_.__args__[0], A)

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
        
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'Expected {name} to be a torch.Tensor, found {type(value)}.')
        
        self.parses[name] = annotation.parse_shape(value.shape)
        self.annotations[name] = annotation

    def check(self) -> Optional[ShapeError]:
        maybe_error = check_consistency(self.parses)
        if maybe_error is not None:
            for name, annotation in self.annotations.items():
                rendered = annotation.render(self.parses[name])
                maybe_error.context.append(f'{name}: {rendered}')
        return maybe_error
         
def dimchecked(func):
    signature = inspect.signature(func)
    
    def wrapped(*args, **kwargs):
        checker_state = CheckerState()
        
        for name, value, annotation in _zip_args_and_labels(args, signature):
            checker_state.update(name, value, annotation)
        
        maybe_error = checker_state.check()
        if maybe_error is not None:
            raise maybe_error

        result = func(*args, **kwargs)

        if not isinstance(signature.return_annotation, tuple):
            checker_state.update('<return>', result, signature.return_annotation)
        else:
            if len(signature.return_annotation) != len(result):
                raise TypeError(f'Return should have {len(signature.return_annotation)} elements, found {len(result)}.')

            for i, (value, annotation) in enumerate(zip(result, signature.return_annotation)):
                checker_state.update(f'<return {i}>', value, annotation)

        maybe_error = checker_state.check()
        if maybe_error is not None:
            raise maybe_error

        return result
    return wrapped
