from io import BytesIO
from pathlib import Path

import pytest

from model_arena import ModelArena
from model_arena.aws import publish_s3, materialize_publication, s3_parts
from model_arena.storage import verify_manifest, digest


class MemoryS3:
    def __init__(self, fail=False):
        self.objects = {}
        self.metadata = {}
        self.fail = fail

    def upload_file(self, path, bucket, key, ExtraArgs):
        if self.fail:
            raise OSError("Upload interrupted")
        self.objects[key] = Path(path).read_bytes()
        self.metadata[key] = ExtraArgs["Metadata"]

    def head_object(self, *, Bucket, Key):
        return {"Metadata": self.metadata[Key]}

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body

    def get_object(self, *, Bucket, Key):
        return {"Body": BytesIO(self.objects[Key])}

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[key])


def test_publication_s3_roundtrip_and_atomic_discovery(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "table.json").write_text('{"data":[]}')
    client = MemoryS3()
    publication = publish_s3(source, "s3://example/base", client=client)
    result = materialize_publication(publication.path, tmp_path / "download", client=client)
    verify_manifest(result)
    assert digest(source / "table.json") == digest(result / "table.json")
    assert "base/latest.json" in client.objects


def test_incomplete_s3_upload_never_publishes_pointer(tmp_path):
    (tmp_path / "table.json").write_text("{}")
    client = MemoryS3(fail=True)
    with pytest.raises(OSError):
        publish_s3(tmp_path, "s3://example/base", client=client)
    assert "base/latest.json" not in client.objects


@pytest.mark.parametrize("uri", ["https://example/a", "s3:///missing", "s3://bucket/a?token=secret"])
def test_invalid_s3_uri_is_rejected(uri):
    with pytest.raises(ValueError):
        s3_parts(uri)


def test_scaffold_renders_publication_and_views(panel, spec, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from importlib.util import find_spec
    import sys
    df, static = panel
    arena = ModelArena(spec).fit(df=df, static_df=static)
    release = arena.postprocess(arena.predict())
    publication = arena.publish(release, destination=tmp_path / "publications")
    monkeypatch.setattr(sys, "argv", ["dashboard.py", publication.path])
    app = AppTest.from_file(find_spec("model_arena.dashboard").origin).run(timeout=20)
    assert not app.exception
    app.sidebar.radio[0].set_value("Pronósticos y alertas").run()
    assert not app.exception
    assert len(app.selectbox) >= 1
    app.sidebar.radio[0].set_value("Model Arena").run()
    assert not app.exception

