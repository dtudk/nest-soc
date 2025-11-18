def test_import_package():
    import nest  # noqa: F401


def test_import_submodules():
    # Import key submodules to ensure packaging metadata is correct
    from nest import cell, idealGas, layers, problem  # noqa: F401

    assert cell is not None
    assert idealGas is not None
    assert layers is not None
    assert problem is not None
