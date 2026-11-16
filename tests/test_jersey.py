from sportsvision.jersey import vote_number


def test_consistent_reads_win():
    assert vote_number([("10", 0.9), ("10", 0.8), ("10", 0.7), ("16", 0.4)]) == "10"


def test_two_reads_are_not_enough():
    assert vote_number([("7", 0.99), ("7", 0.99)]) is None


def test_invalid_reads_are_ignored():
    reads = [("123", 0.9), ("", 0.9), ("A1", 0.9), ("9", 0.6), ("9", 0.7), ("9", 0.8)]
    assert vote_number(reads) == "9"


def test_leading_zero_numbers_are_valid():
    assert vote_number([("08", 1.0)] * 3) == "08"  # Punjab's #08 in the basketball clip


def test_partial_single_digit_reads_support_the_two_digit_number():
    # a folded shirt or the locator clipping one digit turns 32 into "3" or "2"
    reads = [("32", 0.9)] * 3 + [("3", 0.9), ("3", 0.9), ("2", 0.9)]
    assert vote_number(reads) == "32"


def test_split_vote_returns_nothing():
    assert vote_number([("4", 0.8)] * 3 + [("14", 0.8)] * 3) == "14"  # "4" is part of 14
    assert vote_number([("5", 0.8)] * 3 + [("17", 0.8)] * 3) is None
