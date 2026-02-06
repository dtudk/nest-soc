import pytest

from nest.properties import BasicSpecies, Mixture


def test_specie_a():
    H2 = BasicSpecies.H2
    a_low = H2.a(T=298)
    assert a_low[0] == 4.078323210e04
    a_high = H2.a(T=1500)
    assert a_high[0] == 5.608128010e05
    with pytest.raises(ValueError):
        H2.a(T=-100)


def test_specie_cp():
    H2 = BasicSpecies.H2
    cp_0 = H2.cp(T=298.15)
    assert abs(cp_0 - 28.836) <= 0.001


def test_specie_h():
    H2 = BasicSpecies.H2
    hf_0 = H2.h(T=298.15)
    assert abs(hf_0 - 0.0) <= 0.001


def test_specie_s():
    H2 = BasicSpecies.H2
    s_0 = H2.s(T=298.15, P=1e5)
    assert abs(s_0 - 130.681) <= 0.001


def test_D_ij():
    """
    Based on experimental data and
    model error reported by [1]

    Reference
    ---------
    1. https://doi-org.proxy.findit.cvt.dk/10.1021/j100845a020
    """
    H2 = BasicSpecies.H2
    H2O = BasicSpecies.H2O
    mix = Mixture((H2, H2O))
    D_H2_H2O = mix.D_ij(0, 1, 307.3, 101325)
    assert abs(1 - D_H2_H2O / 1.02e-4 <= 9.19 / 100)
    D_H2_H2O = mix.D_ij(0, 1, 328.6, 101325)
    assert abs(1 - D_H2_H2O / 1.1210e-4 <= 7.09 / 100)
    D_H2_H2O = mix.D_ij(0, 1, 352.7, 101325)
    assert abs(1 - D_H2_H2O / 1.20e-4 <= 1.76 / 100)
