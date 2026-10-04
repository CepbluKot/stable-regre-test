import pandas as pd

from anomaly_explorer.replay import chronological_prefix, initial_cursor, next_cursor


def test_replay_starts_with_enough_history_and_only_seen_rows():
    frame = pd.DataFrame(
        {"timestamp": [4000, 1000, 3000, 2000], "value_0": [4, 1, 3, 2]}
    )
    assert initial_cursor(total=4020, minimum_context=61, tail=240) == 3780
    assert initial_cursor(total=120, minimum_context=61, tail=240) == 61
    assert initial_cursor(total=61, minimum_context=61, tail=240) is None
    prefix = chronological_prefix(frame, cursor=2)
    assert prefix.timestamp.tolist() == [1000, 2000]
    assert prefix.value_0.tolist() == [1, 2]


def test_replay_tick_pauses_and_loops_without_exceeding_source():
    assert next_cursor(3780, total=4020, start=3780, step=5, playing=True) == 3785
    assert next_cursor(4019, total=4020, start=3780, step=5, playing=True) == 4020
    assert next_cursor(4020, total=4020, start=3780, step=5, playing=True) == 3780
    assert next_cursor(3900, total=4020, start=3780, step=5, playing=False) == 3900
