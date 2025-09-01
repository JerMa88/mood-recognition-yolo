import resnet as le


def test_import_and_version():
    assert isinstance(le.__version__, str) and le.__version__, "Version should be a non-empty string"


def test_ping():
    assert le.ping() == "resnet"
