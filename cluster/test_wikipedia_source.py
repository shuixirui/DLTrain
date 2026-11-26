"""Check download integrity/resume and article provenance without internet or NPU access."""

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fetch_wikipedia_zh import download_attempt, select_articles


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


class WikipediaSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / "fixture.bin"
        self.data = b"explicit test fixture, not a real dataset shard"
        self.digest = hashlib.sha256(self.data).hexdigest()

    def download(self, response):
        with patch("urllib.request.urlopen", return_value=response) as open_url:
            download_attempt("https://example.invalid/fixture", self.target, len(self.data), self.digest)
        return open_url

    def test_full_file_must_match_pinned_digest(self):
        self.download(Response(self.data))
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_resume_uses_verified_range(self):
        partial = self.target.with_name(self.target.name + ".part")
        partial.write_bytes(self.data[:7])
        call = self.download(Response(self.data[7:], 206, {"Content-Range": f"bytes 7-{len(self.data)-1}/{len(self.data)}"}))
        self.assertEqual(call.call_args.args[0].get_header("Range"), "bytes=7-")
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_server_ignoring_range_restarts_file(self):
        self.target.with_name(self.target.name + ".part").write_bytes(b"old-prefix")
        self.download(Response(self.data))
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_wrong_range_cannot_corrupt_partial_file(self):
        partial = self.target.with_name(self.target.name + ".part")
        partial.write_bytes(self.data[:7])
        with self.assertRaisesRegex(ValueError, "Content-Range"):
            self.download(Response(self.data[7:], 206, {"Content-Range": f"bytes 0-{len(self.data)-1}/{len(self.data)}"}))
        self.assertEqual(partial.read_bytes(), self.data[:7])
        self.assertFalse(self.target.exists())

    def test_wrong_digest_never_promoted_to_source_file(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.download(Response(b"x" * len(self.data)))
        self.assertFalse(self.target.exists())
        self.assertEqual(len(list(self.target.parent.glob("*.rejected-*"))), 1)

    def test_html_download_rejected(self):
        with self.assertRaisesRegex(ValueError, "HTML"):
            self.download(Response(self.data, headers={"Content-Type": "text/html"}))
        self.assertFalse(self.target.exists())

    def test_article_identity_and_complete_text_preserved(self):
        row = dict(id="test", url="https://zh.wikipedia.org/wiki/Fixture", title="fixture", text="x" * 500)
        second = dict(row, id="second")
        result = list(select_articles([dict(row, text="short"), row, second], max_characters=400))
        self.assertEqual(result, [row])
        self.assertEqual(len(result[0]["text"]), 500)

    def test_non_wikipedia_provenance_rejected(self):
        row = dict(id="test", url="https://example.invalid/", title="fixture", text="x" * 500)
        with self.assertRaisesRegex(ValueError, "provenance"):
            list(select_articles([row]))


if __name__ == "__main__":
    unittest.main()
