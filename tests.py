import unittest, torch
from torch_dimcheck import dimchecked, ShapeError, A

class ShapeCheckedTests(unittest.TestCase):
    def test_wrap_no_anno(self):
        def f(t1, t2): # t1: [3, 5], t2: [5, 3] -> [3]
            return (t1.transpose(0, 1) * t2).sum(dim=0)
             
        t1 = torch.randn(3, 5)
        t2 = torch.randn(5, 3)

        self.assertTrue((f(t1, t2) == dimchecked(f)(t1, t2)).all())

    def test_wrap_correct(self):
        def f(t1: A['3 5'], t2: A['5 3']) -> A['3']:
            return (t1.transpose(0, 1) * t2).sum(dim=0)
             
        t1 = torch.randn(3, 5)
        t2 = torch.randn(5, 3)

        self.assertTrue((f(t1, t2) == dimchecked(f)(t1, t2)).all())

    def test_fails_wrong_parameter(self):
        def f(t1: A['3 3'], t2: A['5, 3']) -> A['3']:
            return (t1.transpose(0, 1) * t2).sum(dim=0)
             
        t1 = torch.randn(3, 5)
        t2 = torch.randn(5, 3)

        with self.assertRaises(ShapeError) as ex:
            dimchecked(f)(t1, t2)

    def test_fails_parameter_label_mismatch(self):
        def f(t1: A['3 a'], t2: A['a 3']) -> A['3']:
            return (t1.transpose(0, 1) * t2).sum(dim=0)
             
        t1 = torch.randn(3, 4)
        t2 = torch.randn(5, 3)

        with self.assertRaises(ShapeError):
            dimchecked(f)(t1, t2)

    def test_succeeds_with_labels(self):
        def f(t1: A['3 a'], t2: A['a 3']) -> A['3']:
            return (t1.transpose(0, 1) * t2).sum(dim=0)
             
        t1 = torch.randn(3, 4)
        t2 = torch.randn(4, 3)

        dimchecked(f)(t1, t2)

    def test_fails_wrong_return(self):
        def f(t1: A['3 5'], t2: A['5 3']) -> A['5']:
            return (t1.transpose(0, 1) * t2).sum(dim=0)
             
        t1 = torch.randn(3, 5)
        t2 = torch.randn(5, 3)

        with self.assertRaises(ShapeError) as ex:
            dimchecked(f)(t1, t2)

    def test_fails_return_label_mismatch(self):
        def f(t1: A['5 a'], t2: A['a 5']) -> A['a']:
            return (t1.transpose(0, 1) * t2).sum(dim=0)
             
        t1 = torch.randn(5, 3)
        t2 = torch.randn(3, 5)

        with self.assertRaises(ShapeError):
            dimchecked(f)(t1, t2)

    def test_fails_tuple_return_label_mismatch(self):
        def f(t1: A['a b'], t2: A['b a']) -> (A['b b'], A['a a']):
            ab = t1 @ t2
            ba = t1.T @ t2.T
            return ab, ba
             
        t1 = torch.randn(5, 3)
        t2 = torch.randn(3, 5)

        with self.assertRaises(ShapeError):
            dimchecked(f)(t1, t2)

    def test_succeeds_tuple_return(self):
        def f(t1: A['a b'], t2: A['b a']) -> (A['a a'], A['b b']):
            ab = t1 @ t2
            ba = t1.T @ t2.T
            return ab, ba
             
        t1 = torch.randn(5, 3)
        t2 = torch.randn(3, 5)

        dimchecked(f)(t1, t2)

    def test_fails_two_wildcards(self):
        with self.assertRaises(TypeError) as ex:
            def f(t1: A['3 ...a ...b 2'], t2: A['5 3']):
                pass

    def test_succeeds_consistent_wildcards(self):
        def f(t1: A['b... 3'], t2: A['b... 3']):
            pass
             
        t1 = torch.randn(3, 2, 3)
        t2 = torch.randn(3, 2, 3)

        dimchecked(f)(t1, t2)

        def g(t1: A['b... a 3'], t2: A['b... a 3']):
            pass

        dimchecked(g)(t1, t2)

    def test_succeeds_anonymous_wildcards(self):
        def f(t1: A['... 3'], t2: A['... 3']):
            pass
             
        t1 = torch.randn(3, 2, 3)
        t2 = torch.randn(3, 4, 3)

        dimchecked(f)(t1, t2)

    def test_fails_inconsistent_wildcards(self):
        def f(t1: A['b... 3'], t2: A['b... 3']):
            pass
             
        t1 = torch.randn(3, 3, 3)
        t2 = torch.randn(3, 5, 3)

        with self.assertRaises(ShapeError) as ex:
            dimchecked(f)(t1, t2)

    def test_fails_non_type_annotation(self):
        def f(t1: A['b... 3'], t2: ['b... 3']):
            pass
             
        t1 = torch.randn(3, 3, 3)
        t2 = torch.randn(3, 5, 3)

        with self.assertRaises(TypeError) as ex:
            dimchecked(f)(t1, t2)
        self.assertTrue('std::typing' in str(ex.exception))

#    def test_fails_backward_ellipsis_wildcard(self):
#        def f(t1: [3, ..., 'a'], t2: [5, ..., 'a']):
#            pass
#             
#        t1 = torch.randn(3, 3, 5)
#        t2 = torch.randn(5, 3, 3)
#
#        msg = ("Label `a` already had dimension 5 bound to it "
#               "(based on tensor t1 of shape (3, 3, 5)), but it "
#               "appears with dimension 3 in tensor t2")
#        with self.assertRaises(ShapeError) as ex:
#            dimchecked(f)(t1, t2)
#        self.assertEqual(str(ex.exception), msg)
#
#    def test_fails_backward_just_ellipsis(self):
#        def f(t1: [..., 2], t2: [..., 2]):
#            pass
#             
#        t1 = torch.randn(3, 3, 3, 2)
#        t2 = torch.randn(5, 3, 1, 5)
#
#        msg = "Size mismatch on dimension 3 of argument `t2` (found 5, expected 2)"
#        with self.assertRaises(ShapeError) as ex:
#            dimchecked(f)(t1, t2)
#        self.assertEqual(str(ex.exception), msg)
#
#    def test_succeeds_ellipsis(self):
#        def f(t1: [5, ..., 'a'], t2: ['a', ..., 5]) -> ['a']:
#            return (t1.transpose(0, 3) * t2).sum(dim=(1, 2, 3))
#             
#        t1 = torch.randn(5, 1, 2, 3)
#        t2 = torch.randn(3, 1, 2, 5)
#
#        self.assertTrue((f(t1, t2) == dimchecked(f)(t1, t2)).all())
