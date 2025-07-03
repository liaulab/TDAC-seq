from tdac_seq.crispr import find_cutsite, slice_cs, slice_cs_left, slice_cs_right, clean_cs
import pytest

@pytest.mark.parametrize("guide, expected", [
    ("GCCGCCGGGGAGAAGGATGA", 47),
    ("TCCTTCATCCTTCTCCCCGG", 37),
    ("AAAAAAAAAAAAAAAAAAAA", None),
    ])
def test_find_cutsite(guide, expected):
    cutsites = find_cutsite(
        ref_seq="AGCCCGGAGGTCCCGCGTGGAGCTGCCGCCGCCGCCGGGGAGAAGGATGAAGGACAAACAGAAGAAGAAGAAGGAGCGCACGTGGGCCGAGGCCGCGCGC",
        guide=guide,
        )
    assert cutsites == expected

@pytest.mark.parametrize("cs, pos, start, expected", [
    ("*ct", 0, 0, "*ct"),
    ("*ct*ag", 0, 0, "*ct*ag"),
    ("*ct*ag", 1, 0, "*ag"),
    ("*ct*ag", 2, 0, None),
    (":10", 0, 0, ":10"),
    (":10", 1, 0, ":9"),
    (":10", 2, 0, ":8"),
    (":10", 1, 1, ":10"),
    (":10", 1, 2, None),
    (":10", 10, 2, ":2"),
    (":3-ct:3", 0, 0, ":3-ct:3"),
    (":3-ct:3", 1, 0, ":2-ct:3"),
    (":3-ct:3", 2, 0, ":1-ct:3"),
    (":3-ct:3", 3, 0, "-ct:3"),
    (":3-ct:3", 4, 0, "-t:3"),
    (":3-ct:3", 5, 0, ":3"),
    (":3-ct:3", 6, 0, ":2"),
    (":3-ct:3", 7, 0, ":1"),
    (":3-ct:3", 8, 0, None),
    (":1*ct:1", 0, 0, ":1*ct:1"),
    (":1*ct:1", 1, 0, "*ct:1"),
    (":1*ct:1", 2, 0, ":1"),
])
def test_slice_cs_left(cs, pos, start, expected):
    out = slice_cs_left(cs, pos, start)
    assert out == expected

@pytest.mark.parametrize("cs, pos, start, expected", [
    ("*ct*ag", 0, 0, None),
    ("*ct*ag", 1, 0, "*ct"),
    ("*ct*ag", 2, 0, "*ct*ag"),
    ("*ct*ag", 3, 0, None),
    (":10", 0, 0, None),
    (":10", 1, 0, ":1"),
    (":10", 2, 0, ":2"),
    (":10", 10, 0, ":10"),
    (":10", 11, 0, None),
    (":10", 1, 1, None),
    (":10", 1, 2, None),
    (":3-ct:3", 0, 0, None),
    (":3-ct:3", 1, 0, ":1"),
    (":3-ct:3", 2, 0, ":2"),
    (":3-ct:3", 3, 0, ":3"),
    (":3-ct:3", 4, 0, ":3-c"),
    (":3-ct:3", 5, 0, ":3-ct"),
    (":3-ct:3", 6, 0, ":3-ct:1"),
    (":3-ct:3", 7, 0, ":3-ct:2"),
    (":3-ct:3", 8, 0, ":3-ct:3"),
    (":1*ct:1", 0, 0, None),
    (":1*ct:1", 1, 0, ":1"),
    (":1*ct:1", 2, 0, ":1*ct"),
    (":1*ct:1", 3, 0, ":1*ct:1"),
    (":1*ct:1", 4, 0, None),
])
def test_slice_cs_right(cs, pos, start, expected):
    out = slice_cs_right(cs, pos, start)
    assert out == expected

@pytest.mark.parametrize("cs, window, start, expected", [
    ("*ct", slice(0, 1), 0, "*ct"),
    ("*ct*ag", slice(0, 1), 0, "*ct"),
    ("*ct*ag", slice(1, 2), 0, "*ag"),
    ("*ct*ag", slice(0, 2), 0, "*ct*ag"),
    ("*ct*ag", slice(0, 3), 0, None),
    ("*ct*ag", slice(1, 3), 0, None),
    ("*ct*ag", slice(1, 2), 1, "*ct"),
    ("*ct*ag", slice(0, 2), 1, None),
    (":10", slice(0, 1), 0, ":1"),
    (":10", slice(1, 2), 0, ":1"),
])
def test_slice_cs(cs, window, start, expected):
    out = slice_cs(cs, window, start)
    assert out == expected

@pytest.mark.parametrize("cs, cutsite, expected", [
    (":10", 5, ":10"),
    (":5*ct:4", 5, ":5*ct:4"),
    ("*ct:9", 5, ":10"),
    (":4*ct*ct:4", 5, ":4*ct*ct:4"),
    (":3*ct-ct:4", 5, ":3*ct-ct:4"),
    (":2*ct:1-ct:4", 5, ":4-ct:4"),
    ("-cccccccccccc", 5, "-cccccccccccc"),
])
def test_clean_cs(cs, cutsite, expected):
    out = clean_cs(cs, cutsite)
    assert out == expected
