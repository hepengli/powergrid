import numpy as np
import pytest

from powergrid.core.action import Action

rng = np.random.default_rng(12345)

def test_set_specs_initializes_shapes():
    a = Action().set_specs(dim_c=2, dim_d=1, ncats=4)
    assert a.c.shape == (2,)
    assert a.c.dtype == np.float32
    assert a.d.shape == (1,)
    assert a.d.dtype == np.int32
    assert a.ncats == 4


def test_sample_continuous_and_multidiscrete(rng):
    lb = np.array([-1.0, 0.0], np.float32)
    ub = np.array([+1.0, 2.0], np.float32)
    a = Action().set_specs(dim_c=2, dim_d=3, ncats=[2, 3, 4], range=(lb, ub))
    a.sample(rng)

    # c within bounds
    assert np.all(a.c >= lb) and np.all(a.c <= ub)
    # d within 0..K-1
    Ks = np.array([2, 3, 4])
    assert np.all((a.d >= 0) & (a.d < Ks))


def test_sampling_respects_per_head_masks(rng):
    # dim_d=3 with categories [2,3,5]
    masks = [
        np.array([True, False], dtype=bool),            # only {0}
        np.array([True, False, True], dtype=bool),      # {0,2} allowed
        np.array([True, False, False, True, False], dtype=bool),  # {0,3}
    ]
    a = Action().set_specs(dim_c=0, dim_d=3, ncats=[2, 3, 5], masks=masks)

    seen = set()
    for _ in range(200):
        a.sample(rng)
        seen.add(tuple(a.d.tolist()))

    # each coordinate must obey its mask
    # head0 -> only 0
    assert all(t[0] == 0 for t in seen)
    # head1 -> only 0 or 2
    assert all(t[1] in (0, 2) for t in seen)
    # head2 -> only 0 or 3
    assert all(t[2] in (0, 3) for t in seen)


def test_sampling_reproducible_seed_and_rng():
    a1 = Action().set_specs(dim_c=0, dim_d=2, ncats=[3, 4])
    a2 = Action().set_specs(dim_c=0, dim_d=2, ncats=[3, 4])

    a1.sample(2025)                                  # seed
    a2.sample(np.random.default_rng(2025))           # explicit Generator with same seed
    np.testing.assert_array_equal(a1.d, a2.d)


def test_scale_unscale_roundtrip_with_zero_span():
    lb = np.array([-1.0, 0.0, 2.0], np.float32)
    ub = np.array([+1.0, 4.0, 2.0], np.float32)  # zero-span on last axis
    a = Action().set_specs(dim_c=3, dim_d=0, ncats=0, range=(lb, ub))

    a.c[...] = np.array([0.25, 1.0, 2.0], np.float32)
    x = a.scale()
    np.testing.assert_allclose(x, np.array([0.25, -0.5, 0.0], np.float32), atol=1e-6)

    # unscale should ignore normalized value on zero-span axis
    a.unscale(x)
    np.testing.assert_allclose(a.c, np.array([0.25, 1.0, 2.0], np.float32), atol=1e-6)


def test_clip_in_place():
    lb = np.array([-1.0, -2.0], np.float32)
    ub = np.array([+1.0, +2.0], np.float32)
    a = Action().set_specs(dim_c=2, dim_d=0, range=(lb, ub))
    a.c[...] = np.array([5.0, -9.0], np.float32)
    a.clip_()
    np.testing.assert_allclose(a.c, np.array([1.0, -2.0], np.float32))


def test_as_vector_and_from_vector():
    lb = np.array([-1.0, -1.0], np.float32)
    ub = np.array([+1.0, +1.0], np.float32)
    a = Action().set_specs(dim_c=2, dim_d=3, ncats=[2, 3, 4], range=(lb, ub))

    a.c[...] = np.array([0.2, -0.3], np.float32)
    a.d[...] = np.array([1, 0, 3], np.int32)
    vec = a.as_vector()
    assert vec.dtype == np.float32
    assert vec.shape == (2 + 3,)

    b = Action.from_vector(vec, dim_c=2, dim_d=3, ncats=[2, 3, 4], range=(lb, ub))
    np.testing.assert_allclose(b.c, a.c)
    np.testing.assert_array_equal(b.d, a.d)


def test_reset():
    a = Action().set_specs(dim_c=3, dim_d=2, ncats=[2, 5])
    a.c[...] = np.array([1.0, -1.0, 0.5], np.float32)
    a.d[...] = np.array([1, 4], np.int32)
    a.reset()
    np.testing.assert_allclose(a.c, np.zeros(3, np.float32))
    np.testing.assert_allclose(a.d, np.zeros(2, np.int32))


def test_error_when_ncats_seq_len_mismatch():
    with pytest.raises(ValueError):
        Action().set_specs(dim_c=0, dim_d=2, ncats=[3])  # len 1 != dim_d 2


def test_error_when_ncats_invalid_with_dim_d_zero():
    with pytest.raises(ValueError):
        Action().set_specs(dim_c=0, dim_d=0, ncats=[2])  # should be 0 or []


def test_mask_len_mismatch_raises():
    with pytest.raises(ValueError):
        Action().set_specs(dim_c=0, dim_d=2, ncats=[2, 2], masks=[np.array([True, True])])  # len 1 != dim_d


def test_mask_shape_mismatch_raises():
    with pytest.raises(ValueError):
        Action().set_specs(
            dim_c=0, dim_d=2, ncats=[2, 3],
            masks=[np.array([True, True]), np.array([True, False])]  # second should be len 3
        )


def test_mask_all_false_raises():
    with pytest.raises(ValueError):
        Action().set_specs(
            dim_c=0, dim_d=1, ncats=[3],
            masks=[np.array([False, False, False])]
        )