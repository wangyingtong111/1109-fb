# -*- coding: utf-8 -*-

"""
Tests for MosesTokenizer
"""

import hashlib
import os
import tempfile
import unittest
import urllib.error
import urllib.request

from sacremoses.truecase import (
    MosesTruecaser,
    MosesDetruecaser,
    LegacyModelError,
    MODEL_HEADER,
)

#: Norvig's big.txt, the corpus these tests train a truecase model on. It is
#: fetched over the network and cached on disk -- and in CI, under a static
#: cache key -- so without an integrity check the suite trains on whatever
#: bytes the remote host happens to return, and a truecase model shapes the
#: capitalisation of everything downstream of it. Pinned by digest, not by URL.
BIG_TXT_SHA256 = "fa066c7d40f0f201ac4144e652aa62430e58a6b3805ec70650f678da5804e87b"
BIG_TXT_SIZE = 6488666


def get_content(url):
    with urllib.request.urlopen(url) as response:
        return response.read()  # Returns http.client.HTTPResponse.


def verify_big_txt(data):
    """Return ``data`` if it is the expected corpus, else raise.

    Raising rather than warning is deliberate: a silent fallback to unverified
    bytes is exactly the failure this check exists to prevent.
    """
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != BIG_TXT_SIZE or digest != BIG_TXT_SHA256:
        raise ValueError(
            "big.txt failed verification: expected %d bytes / sha256 %s, "
            "got %d bytes / sha256 %s. Delete any cached big.txt and retry; "
            "if it persists, do not trust the source."
            % (BIG_TXT_SIZE, BIG_TXT_SHA256, len(data), digest)
        )
    return data


class TestTruecaser(unittest.TestCase):
    def test_moses_truecase_documents(self):
        moses = MosesTruecaser()
        # Train the model from documents.
        docs = [line.split() for line in self.big_txt.split("\n")]
        moses.train(docs)
        # Test all self.input_output test cases.
        for _input, _output in self.input_output.items():
            self.assertEqual(moses.truecase(_input), _output)

    def test_moses_truecase_file(self):
        moses = MosesTruecaser()
        # Train the model from file.
        moses.train_from_file("big.txt")
        # Test all self.input_output test cases.
        for _input, _output in self.input_output.items():
            self.assertEqual(moses.truecase(_input), _output)

    def setUp(self):
        # Check if the Norvig's big.txt file exists. A cached copy is verified
        # too -- it may be left over from an earlier unverified run, or from CI
        # restoring its cache.
        if os.path.isfile("big.txt"):
            with open("big.txt", "rb") as fin:
                raw = verify_big_txt(fin.read())
        else:  # Otherwise, download the big.txt.
            try:  # Download from the original norvig.com
                raw = verify_big_txt(get_content("https://norvig.com/big.txt"))
            except (urllib.error.URLError, OSError, ValueError):
                # Named exceptions, not a bare `except:` -- that swallowed
                # KeyboardInterrupt and, worse, turned a failed integrity check
                # into a silent fallback to the mirror.
                big_text_url = str(
                    "https://gist.githubusercontent.com/alvations/"
                    "6e878bab0eda2624167aa7ec13fc3e94/raw/"
                    "4fb3bac1da1ba7a172ff1936e96bee3bc8892931/"
                    "big.txt"
                )
                raw = verify_big_txt(get_content(big_text_url))
            # newline="" so Windows does not rewrite \n to \r\n and change the
            # digest of the file we just verified.
            with open("big.txt", "w", encoding="utf-8", newline="") as fout:
                fout.write(raw.decode("utf8"))
        self.big_txt = raw.decode("utf8")

        # Test case where inputs are all caps.
        caps_input = "THE ADVENTURES OF SHERLOCK HOLMES"
        expected_caps_output = ["the", "adventures", "of", "Sherlock", "Holmes"]

        # Test normal input to truecase.
        normal_input = str(
            "You can also find out about how to make a donation "
            "to Project Gutenberg, and how to get involved."
        )
        expecte_normal_output = [
            "you",
            "can",
            "also",
            "find",
            "out",
            "about",
            "how",
            "to",
            "make",
            "a",
            "donation",
            "to",
            "Project",
            "Gutenberg,",
            "and",
            "how",
            "to",
            "get",
            "involved.",
        ]

        # Keep a key-value pairs of in/outputs.
        self.input_output = {
            caps_input: expected_caps_output,
            normal_input: expecte_normal_output,
        }

    def test_use_known(self):
        moses = MosesTruecaser()
        moses.train([
            ['Start', 'a', 'a', 'A', 'A', 'a', 'a', '.'], # 'a' is best, but 'A' is also known.
        ])
        self.assertEqual(moses.truecase('Start A .', use_known=False), ['Start', 'a', '.'])
        self.assertEqual(moses.truecase('Start A .', use_known=True), ['Start', 'A', '.'])

    def test_delayed_sentence_start(self):
        """Test that first word after delayed sentence start still holds, and
        thus ignores use_known for that particular token."""
        moses = MosesTruecaser()
        moses.train([
            ['Start', 'Start', 'start'] # 'Start' is best, but 'start' is okay.
        ])
        self.assertEqual(moses.truecase('" start start', use_known=True), ['"', 'Start', 'start'])


class TestTruecaseModelMerging(unittest.TestCase):
    """Batch-produced truecase models must merge on their true per-variant
    counts, without reassembling the original corpus, and the result must be
    independent of batch count and merge order."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def path(self, name):
        return os.path.join(self.tmpdir.name, name)

    def train_shard(self, name, sentences):
        moses = MosesTruecaser()
        moses.train(sentences)
        path = self.path(name)
        moses.save_model(path)
        return path

    def test_written_model_carries_version_header_and_true_counts(self):
        path = self.train_shard("a.model", [["x", "Oak", "Oak", "oak"]])
        with open(path, encoding="utf8") as fin:
            lines = [line.rstrip("\n") for line in fin if line.strip()]
        self.assertEqual(lines[0], MODEL_HEADER)
        # Both the best and the non-best variant carry their real counts, not a
        # placeholder (1): Oak=2, oak=1.
        self.assertIn("Oak (2/3)", lines[1])
        self.assertIn("oak (1)", lines[1])

    def test_merge_sums_true_variant_counts(self):
        # Shard 1: Oak 4, oak 3. Shard 2: OAK 4, oak 3. Leading 'x' keeps the
        # target tokens out of the sentence-initial position so they get full
        # weight, matching the worked example in the feature request.
        p1 = self.train_shard("b1.model", [["x"] + ["Oak"] * 4 + ["oak"] * 3])
        p2 = self.train_shard("b2.model", [["x"] + ["OAK"] * 4 + ["oak"] * 3])

        moses = MosesTruecaser()
        model = moses.merge_models([p1, p2])
        counts = model["casing"]["oak"]
        # oak (6) is the most frequent form, but Oak and OAK stay known writes.
        self.assertEqual(dict(counts), {"oak": 6, "Oak": 4, "OAK": 4})
        self.assertEqual(model["best"]["oak"], "oak")
        self.assertEqual(set(counts), {"oak", "Oak", "OAK"})

    def test_merge_equals_training_on_the_whole_corpus(self):
        sentences = [
            ["x"] + ["Oak"] * 4 + ["oak"] * 3,
            ["x"] + ["OAK"] * 4 + ["oak"] * 3,
        ]
        p1 = self.train_shard("b1.model", sentences[:1])
        p2 = self.train_shard("b2.model", sentences[1:])
        merged = MosesTruecaser().merge_models([p1, p2])

        whole = MosesTruecaser()
        whole.train(sentences)
        self.assertEqual(
            {k: dict(v) for k, v in merged["casing"].items()},
            {k: dict(v) for k, v in whole.model["casing"].items()},
        )
        self.assertEqual(merged["best"], whole.model["best"])

    def test_merge_is_order_independent(self):
        p1 = self.train_shard("b1.model", [["x"] + ["Oak"] * 4 + ["oak"] * 3])
        p2 = self.train_shard("b2.model", [["x"] + ["OAK"] * 4 + ["oak"] * 3])
        out_ab, out_ba = self.path("ab.model"), self.path("ba.model")
        MosesTruecaser().merge_models([p1, p2], save_to=out_ab)
        MosesTruecaser().merge_models([p2, p1], save_to=out_ba)
        # Byte-identical output, not just equal counts: ordering is deterministic.
        with open(out_ab, "rb") as fin:
            ab = fin.read()
        with open(out_ba, "rb") as fin:
            ba = fin.read()
        self.assertEqual(ab, ba)

    def test_merge_is_associative_across_more_batches(self):
        # The same totals split two ways: two shards vs three shards, merged in
        # different orders, must converge on the identical model.
        two_a = self.train_shard("two_a.model", [["x"] + ["Oak"] * 4 + ["oak"] * 3])
        two_b = self.train_shard("two_b.model", [["x"] + ["OAK"] * 4 + ["oak"] * 3])
        two_out = self.path("two.model")
        MosesTruecaser().merge_models([two_a, two_b], save_to=two_out)

        three_a = self.train_shard("three_a.model", [["x"] + ["Oak"] * 2 + ["oak"] * 1])
        three_b = self.train_shard("three_b.model", [["x"] + ["Oak"] * 2 + ["oak"] * 2])
        three_c = self.train_shard("three_c.model", [["x"] + ["OAK"] * 4 + ["oak"] * 3])
        for order in ([three_a, three_b, three_c], [three_c, three_a, three_b]):
            out = self.path("three_%d.model" % len(order))
            model = MosesTruecaser().merge_models(order, save_to=out)
            self.assertEqual(
                dict(model["casing"]["oak"]), {"oak": 6, "Oak": 4, "OAK": 4}
            )
            self.assertEqual(model["best"]["oak"], "oak")
            with open(out, "rb") as fin:
                three_bytes = fin.read()
            with open(two_out, "rb") as fin:
                two_bytes = fin.read()
            self.assertEqual(three_bytes, two_bytes)

    def test_merged_model_round_trips_through_a_file(self):
        p1 = self.train_shard("b1.model", [["x"] + ["Oak"] * 4 + ["oak"] * 3])
        p2 = self.train_shard("b2.model", [["x"] + ["OAK"] * 4 + ["oak"] * 3])
        out = self.path("merged.model")
        MosesTruecaser().merge_models([p1, p2], save_to=out)
        reloaded = MosesTruecaser(load_from=out)
        self.assertEqual(reloaded.model["best"]["oak"], "oak")
        self.assertEqual(
            dict(reloaded.model["casing"]["oak"]),
            {"oak": 6, "Oak": 4, "OAK": 4},
        )

    def test_exact_count_tie_breaks_deterministically(self):
        # tie:2 vs Tie:2 -> the lexicographically smaller surface form wins,
        # regardless of the order the variants were inserted.
        p1 = self.train_shard("t1.model", [["x", "tie", "tie"]])
        p2 = self.train_shard("t2.model", [["x", "Tie", "Tie"]])
        m1 = MosesTruecaser().merge_models([p1, p2])
        m2 = MosesTruecaser().merge_models([p2, p1])
        self.assertEqual(m1["best"]["tie"], "Tie")
        self.assertEqual(m2["best"]["tie"], "Tie")
        # The losing variant is still a known surface form.
        self.assertEqual(set(m1["casing"]["tie"]), {"tie", "Tie"})

    def test_merge_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            MosesTruecaser().merge_models([])

    def test_legacy_model_still_loads_but_refuses_to_merge(self):
        # A headerless file in the original Moses/Perl layout: non-best variants
        # use the placeholder (1), so their counts are not recoverable.
        legacy = self.path("legacy.model")
        with open(legacy, "w", encoding="utf8") as fout:
            fout.write("cat (5/7) Cat (1) CAt (1)\n")
            fout.write("dog (2/2)\n")

        # Still readable for inference.
        loaded = MosesTruecaser(load_from=legacy)
        self.assertEqual(loaded.model["best"]["cat"], "cat")
        self.assertEqual(loaded.truecase("CAT ."), ["cat", "."])

        good = self.train_shard("good.model", [["x", "cat", "cat"]])
        # Merging it must fail loudly and name the offending file.
        with self.assertRaises(LegacyModelError) as ctx:
            MosesTruecaser().merge_models([good, legacy])
        self.assertIn("legacy.model", str(ctx.exception))
        self.assertIn(MODEL_HEADER, str(ctx.exception))

    def test_genuine_count_of_one_in_v2_is_mergeable(self):
        # A real (1) under the v2 header must NOT be mistaken for a legacy
        # placeholder: the header is what distinguishes the two.
        v2 = self.path("v2.model")
        with open(v2, "w", encoding="utf8") as fout:
            fout.write(MODEL_HEADER + "\n")
            fout.write("foo (2/3) Foo (1)\n")
        good = self.train_shard("good.model", [["x", "foo", "foo", "foo"]])
        model = MosesTruecaser().merge_models([v2, good])
        self.assertEqual(dict(model["casing"]["foo"]), {"foo": 5, "Foo": 1})

    def test_malformed_model_still_reports_location(self):
        bad = self.path("bad.model")
        with open(bad, "w", encoding="utf8") as fout:
            fout.write(MODEL_HEADER + "\n")
            fout.write("cat (2/3) Cat\n")  # odd number of fields
        with self.assertRaises(ValueError) as ctx:
            MosesTruecaser(load_from=bad)
        self.assertIn("line 2", str(ctx.exception))


class TestDetruecaser(unittest.TestCase):
    def test_moses_detruecase_str(self):
        moses = MosesDetruecaser()
        text = "the adventures of Sherlock Holmes"
        expected = ["The", "adventures", "of", "Sherlock", "Holmes"]
        expected_str = "The adventures of Sherlock Holmes"
        assert moses.detruecase(text) == expected
        assert moses.detruecase(text, return_str=True) == expected_str

    def test_moses_detruecase_headline(self):
        moses = MosesDetruecaser()
        text = "the adventures of Sherlock Holmes"
        expected = ["The", "Adventures", "of", "Sherlock", "Holmes"]
        expected_str = "The Adventures of Sherlock Holmes"
        assert moses.detruecase(text, is_headline=True) == expected
        assert moses.detruecase(text, is_headline=True, return_str=True) == expected_str

    def test_moses_detruecase_allcaps(self):
        moses = MosesDetruecaser()
        text = "MLB Baseball standings"
        expected = ["MLB", "Baseball", "standings"]
        expected_str = "MLB Baseball standings"
        assert moses.detruecase(text) == expected
        assert moses.detruecase(text, return_str=True) == expected_str
