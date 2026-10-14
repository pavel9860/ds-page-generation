from dspages.layout import indic


def test_prefix_vowel_moves_after_cluster():
    assert indic.visual_order("सािहतय")
    assert indic.to_logical("सािहतय") == "साहितय"


def test_logical_text_is_left_alone():
    assert not indic.visual_order("साहित्य हिंदी")


def test_reph_moves_before_cluster():
    assert indic.to_logical("धमर्") == "धर्म"


def test_two_part_vowel_recomposed():
    assert indic.to_logical("ெகா") == "கொ"
