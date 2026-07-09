"""Module C tests: frontier-window selection and derived sector dimensions."""

from math import comb

import numpy as np
import pytest

from qthermal.active_space import select_active


def test_default_window():
    eps = np.linspace(-20.0, 5.0, 24)
    aspace = select_active(eps, nocc=9)  # defaults (4, 4)
    np.testing.assert_array_equal(aspace.active_idx, np.arange(5, 13))
    np.testing.assert_array_equal(aspace.core_idx, np.arange(5))
    assert aspace.ncas == 8
    assert aspace.ncore == 5
    assert aspace.nelecas == 8
    assert aspace.nalpha == aspace.nbeta == 4
    assert aspace.dim == comb(8, 4) ** 2 == 4900


def test_parametric_dimensions():
    eps = np.zeros(30)
    for n_occ_act, n_virt_act, nocc in [(3, 3, 7), (5, 5, 10), (2, 4, 6)]:
        aspace = select_active(eps, nocc=nocc, n_act_occ=n_occ_act,
                               n_act_virt=n_virt_act)
        ncas = n_occ_act + n_virt_act
        assert aspace.ncas == ncas
        assert aspace.nelecas == 2 * n_occ_act
        assert aspace.dim == comb(ncas, n_occ_act) ** 2
        assert aspace.active_idx[0] == nocc - n_occ_act
        assert aspace.active_idx[-1] == nocc + n_virt_act - 1
        assert len(aspace.core_idx) == nocc - n_occ_act


def test_scaling_reference_dims():
    """Spec scaling reference: ncas=6 -> 400, 8 -> 4900, 10 -> 63504, 12 -> 853776."""
    eps = np.zeros(40)
    for half, dim in [(3, 400), (4, 4900), (5, 63_504), (6, 853_776)]:
        aspace = select_active(eps, nocc=half + 3, n_act_occ=half, n_act_virt=half)
        assert aspace.dim == dim


def test_guards():
    eps = np.zeros(12)
    with pytest.raises(ValueError, match="core"):
        select_active(eps, nocc=4, n_act_occ=4, n_act_virt=4)
    with pytest.raises(ValueError, match="virtual"):
        select_active(eps, nocc=9, n_act_occ=4, n_act_virt=4)
    with pytest.raises(ValueError, match=">= 1"):
        select_active(eps, nocc=6, n_act_occ=0, n_act_virt=4)
