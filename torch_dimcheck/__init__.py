import torch, functools, inspect


class Binding:
    def __init__(self, label, value, tensor_name, tensor_shape):
        self.label = label
        self.value = value
        self.tensor_name = tensor_name
        self.tensor_shape = tensor_shape

class ShapeChecker:
    def __init__(self):
        self.d = dict()

    def update(self, other):
        if isinstance(other, ShapeChecker):
            other = other.d

        for label in other.keys():
            if label in self.d:
                binding = self.d[label]
                new_binding = other[label]

                if not binding.value == new_binding.value:
                    raise LabeledShapeError(label, binding, new_binding)
            else:
                self.d[label] = other[label]
                

class ShapeError(Exception):
    pass


class SizeMismatch(ShapeError):
    def __init__(self, dim, expected, found, tensor_name):
        self.dim = dim
        self.expected = expected
        self.found = found
        self.tensor_name = tensor_name

    def __str__(self):
        fmt = "Size mismatch on dimension {} of argument `{}` (found {}, expected {})"
        msg = fmt.format(self.dim, self.tensor_name, self.found, self.expected)
        return msg


class LabeledShapeError(ShapeError):
    def __init__(self, label, prev_binding, new_binding):
        self.label = label
        self.prev_binding = prev_binding
        self.new_binding = new_binding

    def __str__(self):
        fmt = ("Label `{}` already had dimension {} bound to it (based on tensor {}"
               "of shape {}), but it appears with dimension {} in tensor {}")
        msg = fmt.format(
            self.label, self.prev_binding.value, self.prev_binding.tensor_name,
            self.prev_binding.shape, self.new_binding.value, self.new_binding.tensor_name
        )
        return msg


def get_bindings(tensor, annotation, tensor_name=None):
    n_ellipsis = annotation.count(...)
    if n_ellipsis > 1:
        # TODO: check this condition earlier
        raise ValueError("Only one ellipsis can be used per annotation")

    if len(annotation) != len(tensor.shape) and n_ellipsis == 0:
        # no ellipsis, dimensionality mismatch
        fmt = "Annotation {} differs in size from tensor shape {} ({} vs {})"
        msg = fmt.format(annotation, tensor.shape, len(annotation), len(tensor.shape))
        raise ShapeError(msg)

    bindings = ShapeChecker()
    # check if dimensions match, one by one
    for i, (dim, anno) in enumerate(zip(tensor.shape, annotation)):
        if isinstance(anno, str):
            # named wildcard, add to dict
            bindings.update({anno: Binding(anno, dim, tensor_name, tensor.shape)})
        elif anno == ...:
            # ellipsis - done checking from the front, skip to checking in reverse
            break
        elif isinstance(anno, int) and anno != dim:
            if anno == -1:
                # anonymous wildcard dimension, continue
                continue
            else:
                raise SizeMismatch(i, anno, dim, tensor_name)

    if n_ellipsis == 0:
        # no ellipsis - we don't have to go in reverse
        return bindings

    # there was an ellipsis, we have to check in reverse
    for i, (dim, anno) in enumerate(zip(tensor.shape[::-1], annotation[::-1])):
        if isinstance(anno, str):
            # named wildcard, add to dict
            bindings.update({anno: Binding(anno, dim, tensor_name, tensor.shape)})
        elif anno == ...:
            # ellipsis - done checking from the back, return
            return bindings
        elif isinstance(anno, int) and anno != dim:
            if anno == -1:
                # anonymous wildcard dimension, continue
                continue
            else:
                raise SizeMismatch(len(annotation) - i, anno, dim)

    raise AssertionError("Arrived at the end of procedure")


def dimchecked(func):
    sig = inspect.signature(func)

    checked_parameters = dict()
    for i, parameter in enumerate(sig.parameters.values()):
        if isinstance(parameter.annotation, list):
            checked_parameters[i] = parameter

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # check input
        shape_bindings = ShapeChecker()
        for i, arg in enumerate(args):
            if i in checked_parameters:
                param = checked_parameters[i]
                shapes = get_bindings(arg, param.annotation, tensor_name=param.name)
                shape_bindings.update(shapes)

        result = func(*args, **kwargs)

        if isinstance(sig.return_annotation, list):
            # single tensor output like f() -> [3, 6]
            shapes = get_bindings(
                result, sig.return_annotation, tensor_name='<return value>'
            )
            shape_bindings.update(shapes)
        elif isinstance(sig.return_annotation, tuple):
            # tuple output like f() -> ([3, 6], ..., [6, 5])
            for i, anno in enumerate(sig.return_annotation):
                if tensor == ...:
                    # skip
                    continue

                shapes = get_bindings(
                    result, anno, param_name='<return value {}>'.format(i)
                )
                shape_bindings.update(shapes)

        return result

    return wrapped

if __name__ == '__main__':
    import unittest
    
    class ShapeCheckedTests(unittest.TestCase):
        def test_wrap_no_anno(self):
            def f(t1, t2): # t1: [3, 5], t2: [5, 3] -> [3]
                return (t1.transpose(0, 1) * t2).sum(dim=0)
                 
            t1 = torch.randn(3, 5)
            t2 = torch.randn(5, 3)

            self.assertTrue((f(t1, t2) == dimchecked(f)(t1, t2)).all())

        def test_wrap_correct(self):
            def f(t1: [3, 5], t2: [5, 3]) -> [3]:
                return (t1.transpose(0, 1) * t2).sum(dim=0)
                 
            t1 = torch.randn(3, 5)
            t2 = torch.randn(5, 3)

            self.assertTrue((f(t1, t2) == dimchecked(f)(t1, t2)).all())

        def test_fails_wrong_parameter(self):
            def f(t1: [3, 3], t2: [5, 3]) -> [3]:
                return (t1.transpose(0, 1) * t2).sum(dim=0)
                 
            t1 = torch.randn(3, 5)
            t2 = torch.randn(5, 3)

            msg = "Size mismatch on dimension 1 of argument `t1` (found 5, expected 3)"
            with self.assertRaises(ShapeError) as ex:
                dimchecked(f)(t1, t2)
            self.assertEqual(str(ex.exception), msg)

        def test_fails_wrong_return(self):
            def f(t1: [3, 5], t2: [5, 3]) -> [5]:
                return (t1.transpose(0, 1) * t2).sum(dim=0)
                 
            t1 = torch.randn(3, 5)
            t2 = torch.randn(5, 3)

            msg = ("Size mismatch on dimension 0 of argument "
                   "`<return value>` (found 3, expected 5)")
            with self.assertRaises(ShapeError) as ex:
                dimchecked(f)(t1, t2)
            self.assertEqual(str(ex.exception), msg)

        def test_fails_parameter_label_mismatch(self):
            def f(t1: [3, 'a'], t2: ['a', 3]) -> [3]:
                return (t1.transpose(0, 1) * t2).sum(dim=0)
                 
            t1 = torch.randn(3, 4)
            t2 = torch.randn(5, 3)

            with self.assertRaises(ShapeError):
                dimchecked(f)(t1, t2)

        def test_fails_return_label_mismatch(self):
            def f(t1: [5, 'a'], t2: ['a', 5]) -> ['a']:
                return (t1.transpose(0, 1) * t2).sum(dim=0)
                 
            t1 = torch.randn(3, 5)
            t2 = torch.randn(5, 3)

            with self.assertRaises(ShapeError):
                dimchecked(f)(t1, t2)

        def test_succeeds_ellipsis(self):
            def f(t1: [5, ..., 'a'], t2: ['a', ..., 5]) -> ['a']:
                return (t1.transpose(0, 3) * t2).sum(dim=(1, 2, 3))
                 
            t1 = torch.randn(5, 1, 2, 3)
            t2 = torch.randn(3, 1, 2, 5)

            self.assertTrue((f(t1, t2) == dimchecked(f)(t1, t2)).all())

    unittest.main(failfast=True)
