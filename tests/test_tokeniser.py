"""Tokeniser unit tests."""

from __future__ import annotations

import pytest

from sql_engine.errors import TokeniseError
from sql_engine.tokeniser import TokenType, tokenise


def kinds(sql: str) -> list[TokenType]:
    return [token.type for token in tokenise(sql)]


def values(sql: str) -> list[str]:
    return [token.value for token in tokenise(sql)[:-1]]


def test_keywords_are_upper_cased_and_classified():
    tokens = tokenise("select Distinct FROM")
    assert [t.type for t in tokens[:-1]] == [TokenType.KEYWORD] * 3
    assert [t.value for t in tokens[:-1]] == ["SELECT", "DISTINCT", "FROM"]


def test_destructive_words_are_identifiers_not_keywords():
    """The security argument starts here.

    DROP, INSERT, UPDATE and DELETE are not keywords in this language, so they cannot
    begin a statement the parser recognises. They tokenise as ordinary identifiers.
    """
    for word in ("DROP", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "PRAGMA"):
        token = tokenise(word)[0]
        assert token.type is TokenType.IDENTIFIER, word


def test_identifiers_and_qualified_names():
    assert values("a.b_1") == ["a", ".", "b_1"]


def test_quoted_identifier_keeps_case_and_allows_keywords():
    token = tokenise('"select"')[0]
    assert token.type is TokenType.IDENTIFIER
    assert token.value == "select"


def test_quoted_identifier_doubles_to_escape():
    assert tokenise('"a""b"')[0].value == 'a"b'


def test_string_literal_doubles_to_escape():
    token = tokenise("'it''s'")[0]
    assert token.type is TokenType.STRING
    assert token.value == "it's"


@pytest.mark.parametrize(
    "source,expected",
    [("1", "1"), ("1.5", "1.5"), (".5", ".5"), ("1e3", "1e3"), ("1.5e-3", "1.5e-3")],
)
def test_number_shapes(source: str, expected: str):
    token = tokenise(source)[0]
    assert token.type is TokenType.NUMBER
    assert token.value == expected


def test_number_followed_by_letters_is_an_error():
    """`12abc` must not silently split into NUMBER(12) IDENTIFIER(abc)."""
    with pytest.raises(TokeniseError):
        tokenise("SELECT 12abc")


def test_multi_character_operators_win_over_single():
    assert values("a <= b >= c <> d != e") == [
        "a", "<=", "b", ">=", "c", "<>", "d", "!=", "e"
    ]


def test_line_comment_is_skipped():
    assert values("SELECT a -- everything here is ignored\nFROM t") == [
        "SELECT", "a", "FROM", "t"
    ]


def test_block_comment_is_skipped():
    assert values("SELECT /* mid */ a FROM t") == ["SELECT", "a", "FROM", "t"]


def test_unterminated_block_comment_reports_its_start():
    with pytest.raises(TokeniseError) as error:
        tokenise("SELECT a /* never closed")
    assert error.value.line == 1


def test_unterminated_string_is_an_error():
    with pytest.raises(TokeniseError):
        tokenise("SELECT 'unterminated")


def test_illegal_character_reports_position():
    with pytest.raises(TokeniseError) as error:
        tokenise("SELECT a\nFROM ~t")
    assert error.value.line == 2
    assert error.value.column == 6


def test_positions_track_newlines():
    tokens = tokenise("SELECT\n  a")
    assert (tokens[1].line, tokens[1].column) == (2, 3)


def test_eof_token_is_always_last():
    assert tokenise("SELECT 1")[-1].type is TokenType.EOF
    assert tokenise("")[-1].type is TokenType.EOF
