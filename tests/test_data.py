from tmu.data import estimate_text_difficulty, parse_timestamp


def test_parse_unix_timestamp():
    assert parse_timestamp(1767229200) == 1767229200.0


def test_parse_iso_timestamp():
    ts = parse_timestamp("2026-01-01T09:30:00")
    assert isinstance(ts, float)
    assert ts > 0


def test_estimate_text_difficulty_range():
    d = estimate_text_difficulty("Explain why transformer attention works in detail.")
    assert 0.0 <= d <= 1.0
