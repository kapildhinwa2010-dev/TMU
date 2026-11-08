import tmu


def test_version():
    assert tmu.__version__
    assert isinstance(tmu.__version__, str)
