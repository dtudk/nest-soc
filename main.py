from nest.examples.iv_curve import iv_curve
from nest.examples.iv_curve_ce import iv_curve_ce
from nest.examples.iv_curve_co2 import iv_curve_co2
from nest.examples.durability import durability_test
from nest.examples.eis import eis


if __name__ == "__main__":
    # Plotting result examples
    iv_curve()
    iv_curve_ce()
    durability_test()
    iv_curve_co2()
    eis()