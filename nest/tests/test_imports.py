def test_import_package():
    import nest  # noqa: F401


def test_import_submodules():
    # Import key submodules to ensure packaging metadata is correct
    from nest import cell, constants, degradation, layers, properties  # noqa: F401

    assert cell is not None
    assert constants is not None
    assert degradation is not None
    assert layers is not None
    assert properties is not None
