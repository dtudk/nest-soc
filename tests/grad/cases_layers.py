"""
Gradient test cases for the transport and conductivity classes in ``nest.grad.layers``.

Merged into ``test_gradients.py``; lists of ``INSTANCES`` and ``CLASS_INPUTS``
extend the ones already there.
"""

import numpy as np

from nest import properties as base_properties
from nest.grad import layers, properties

species = properties.BasicSpecies
H2_H2O = properties.Mixture((species.H2, species.H2O))
O2 = properties.Mixture((species.O2,))
# Durability-example air electrode: O2-only gas, but 2-species (O2, N2) partial pressures.
AIR_O2 = dict(
    gas=O2, nu=[-0.5], n_e=2, alpha=0.65, beta=0.35, theta=1, E_act=1.496e5,
    p=[0.22], gamma=10.0,
)
POROUS = dict(dp=1e-6, epsilon=0.3, tau=3.0)

INSTANCES = {
    layers.Kinetic: [layers.Kinetic(**AIR_O2)],
    layers.ButlerVolmer: [layers.ButlerVolmer(**AIR_O2)],
    layers.Conductivity: [
        layers.Conductivity(sigma0=3.34e4, E_act=8.6e4, theta=-1),
        layers.Conductivity(sigma0=1.0e5, E_act=1.0e4, theta=0),
        layers.Conductivity(),  # sigma0 = inf: no resistance
    ],
    layers.PorousTransport: [layers.PorousTransport(**POROUS)],
    layers.BinaryFick: [layers.BinaryFick(**POROUS)],
    layers.StefanMaxwell: [layers.StefanMaxwell(**POROUS)],
}

# One location: scalar T, 1-D flux and partial pressures (P is partial pressures for dP_dl
# and total pressures for D_eff).
_BINARY = {
    "T": 950.0, "P": np.array([4.0e4, 6.0e4]), "gas": H2_H2O,
    "mol_flux": np.array([-0.02, 0.02]), "delta": 1e-3,
}
CLASS_INPUTS = {
    layers.PorousTransport: [
        {**_BINARY, "T": np.array([600.0, 950.0, 1400.0]), "specie": species.H2},
    ],
    layers.BinaryFick: [
        _BINARY,
        # nest.properties gas (converted internally for the gradients)
        {**_BINARY, "gas": base_properties.Mixture(
            (base_properties.BasicSpecies.H2, base_properties.BasicSpecies.H2O))},
        # O2-only gas with (O2, N2) partial pressures: flux and D broadcast
        {**_BINARY, "P": np.array([2.0e4, 8.0e4]), "gas": O2, "mol_flux": np.array([-0.005])},
    ],
    layers.StefanMaxwell: [_BINARY],
}

# sigma = inf for sigma0 = inf: no finite differences (checked in test_grad_layers.py).
EXCLUDE = {"Conductivity.sigma[2]"}
