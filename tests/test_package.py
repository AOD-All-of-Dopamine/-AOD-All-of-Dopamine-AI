def test_package_importable():
    import aod_ai
    assert aod_ai.__name__ == "aod_ai"
    assert aod_ai.__version__ == "0.1.0"
