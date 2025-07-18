import pytest

@pytest.fixture(scope='session')
def sample_fixture():
    return "Hello, World!"