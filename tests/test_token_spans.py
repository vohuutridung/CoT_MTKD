from cot_mtkd.data.schema import CharacterSegment, TokenRegion
from cot_mtkd.data.token_spans import assign_token_regions


def test_short_step_merged_with_delimiter_keeps_content_token():
    segments = [
        CharacterSegment(0, 1, TokenRegion.REASONING, 0),
        CharacterSegment(1, 3, TokenRegion.DELIMITER, 0),
    ]
    regions, steps = assign_token_regions([(0, 3)], segments)
    assert regions == [int(TokenRegion.REASONING)]
    assert steps == [0]
