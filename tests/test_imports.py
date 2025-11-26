def test_import_package():
    import nest  # noqa: F401


def test_import_submodules():
    # Import key submodules to ensure packaging metadata is correct
    from nest import cell, ideal_gas, layers, problem  # noqa: F401

    assert cell is not None
    assert ideal_gas is not None
    assert layers is not None
    assert problem is not None
