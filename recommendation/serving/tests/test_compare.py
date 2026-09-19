from aod_serving.tools.compare import pages_equal, tied_positions


def test_exact_match():
    assert pages_equal([1, 2, 3], [1, 2, 3])
    assert not pages_equal([1, 2, 3], [1, 3, 2])


def test_tie_swap_is_allowed_only_with_flag_and_only_where_scores_tie():
    scores = [0.9, 0.7, 0.7, 0.5]
    assert tied_positions(scores) == {1, 2, 3}                       # 마지막 자리는 페이지 밖과 동점일 수 있다
    assert not pages_equal([1, 2, 3, 4], [1, 3, 2, 4], scores)
    assert pages_equal([1, 2, 3, 4], [1, 3, 2, 4], scores, allow_ties=True)
    assert not pages_equal([1, 2, 3, 4], [2, 1, 3, 4], scores, allow_ties=True)   # 0번 자리는 동점이 아니다
    assert pages_equal([1, 2, 3, 4], [1, 2, 3, 9], scores, allow_ties=True)       # 경계 자리


def test_length_mismatch_is_never_equal():
    assert not pages_equal([1, 2], [1], [0.5, 0.5], allow_ties=True)
