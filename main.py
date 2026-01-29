from nest.examples.iv_curve import iv_curve
from nest.examples.durability import durability_test

from nest.tests.test_imports import test_import_package,test_import_submodules

if __name__ == "__main__":
    # Run tests
    test_import_package()
    test_import_submodules()
    # Plotting result examples
    iv_curve()
    durability_test()