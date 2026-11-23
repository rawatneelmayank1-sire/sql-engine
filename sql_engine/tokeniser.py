"""Hand-written SQL tokeniser.

A generated lexer would be shorter, but the error positions are the product here: the
repair loop feeds "line 1, column 34" back to the model, and a model given a precise
location fixes the right token. So the scanner tracks line/column explicitly rather
than reconstructing it from a byte offset after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

from .errors import TokeniseError


class TokenType(Enum):
    """Lexical class of a token.

    KEYWORD and IDENTIFIER are separated at scan time (not parse time) so the parser
    never has to re-check a word list; a quoted identifier stays an IDENTIFIER even
    when it spells a reserved word.
    """

    KEYWORD = auto()
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    OPERATOR = auto()
    PUNCTUATION = auto()
    EOF = auto()


#: Every word the grammar recognises. Anything not listed is an identifier, which is
#: why DROP, INSERT, UPDATE and DELETE are absent: they are not keywords in this
#: language, so ``DROP TABLE t`` tokenises as three identifiers and dies in the parser.
KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "SELECT", "DISTINCT", "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER",
        "LIMIT", "OFFSET", "AS", "AND", "OR", "NOT", "IN", "BETWEEN", "LIKE",
        "IS", "NULL", "JOIN", "INNER", "LEFT", "OUTER", "ON", "ASC", "DESC",
        "TRUE", "FALSE",
    }
)

#: Multi-character operators must be attempted before single-character ones, so this
#: tuple is ordered longest-first and matched by prefix.
_OPERATORS: Final[tuple[str, ...]] = (
    "<=", ">=", "<>", "!=", "=", "<", ">", "+", "-", "*", "/", "%",
)

_PUNCTUATION: Final[frozenset[str]] = frozenset("(),.;")

_SINGLE_QUOTE: Final[str] = chr(39)
_DOUBLE_QUOTE: Final[str] = chr(34)


@dataclass(frozen=True, slots=True)
class Token:
    """A single lexical unit plus the source position it came from."""

    type: TokenType
    #: For KEYWORD this is upper-cased; for every other type it is the source text
    #: (with quotes stripped and escapes resolved for STRING).
    value: str
    line: int
    column: int

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


class Tokeniser:
    """Converts SQL source text into a flat list of :class:`Token`."""

    def __init__(self, source: str) -> None:
        self._src = source
        self._pos = 0
        self._line = 1
        self._col = 1

    # -- character helpers -------------------------------------------------------

    def _peek(self, offset: int = 0) -> str:
        index = self._pos + offset
        return self._src[index] if index < len(self._src) else ""

    def _advance(self, count: int = 1) -> str:
        chunk = self._src[self._pos : self._pos + count]
        for char in chunk:
            if char == "\n":
                self._line += 1
                self._col = 1
            else:
                self._col += 1
        self._pos += count
        return chunk

    # -- main loop ---------------------------------------------------------------

    def tokenise(self) -> list[Token]:
        """Scan the whole input. Raises :class:`TokeniseError` on an illegal character."""
        tokens: list[Token] = []
        while True:
            self._skip_trivia()
            if self._pos >= len(self._src):
                break
            tokens.append(self._scan_token())
        tokens.append(Token(TokenType.EOF, "", self._line, self._col))
        return tokens

    def _skip_trivia(self) -> None:
        """Consume whitespace, ``--`` line comments and ``/* */`` block comments."""
        while self._pos < len(self._src):
            char = self._peek()
            if char in " \t\r\n":
                self._advance()
            elif char == "-" and self._peek(1) == "-":
                while self._pos < len(self._src) and self._peek() != "\n":
                    self._advance()
            elif char == "/" and self._peek(1) == "*":
                start_line, start_col = self._line, self._col
                self._advance(2)
                while True:
                    if self._pos >= len(self._src):
                        raise TokeniseError(
                            "unterminated block comment", self._pos, start_line, start_col
                        )
                    if self._peek() == "*" and self._peek(1) == "/":
                        self._advance(2)
                        break
                    self._advance()
            else:
                return

    def _scan_token(self) -> Token:
        line, col = self._line, self._col
        char = self._peek()

        if char.isalpha() or char == "_":
            return self._scan_word(line, col)
        if char == _DOUBLE_QUOTE:
            return self._scan_quoted_identifier(line, col)
        if char.isdigit() or (char == "." and self._peek(1).isdigit()):
            return self._scan_number(line, col)
        if char == _SINGLE_QUOTE:
            return self._scan_string(line, col)
        if char in _PUNCTUATION:
            self._advance()
            return Token(TokenType.PUNCTUATION, char, line, col)
        for op in _OPERATORS:
            if self._src.startswith(op, self._pos):
                self._advance(len(op))
                return Token(TokenType.OPERATOR, op, line, col)
        raise TokeniseError(f"unexpected character {char!r}", self._pos, line, col)

    def _scan_word(self, line: int, col: int) -> Token:
        start = self._pos
        while self._peek().isalnum() or self._peek() == "_":
            self._advance()
        word = self._src[start : self._pos]
        upper = word.upper()
        if upper in KEYWORDS:
            return Token(TokenType.KEYWORD, upper, line, col)
        return Token(TokenType.IDENTIFIER, word, line, col)

    def _scan_delimited(self, delimiter: str, what: str, line: int, col: int) -> str:
        """Scan a delimited run, treating a doubled delimiter as one literal character.

        Both SQL string literals and SQL quoted identifiers escape their delimiter by
        doubling it, so the two scanners differ only in which character they look for.
        """
        self._advance()  # opening delimiter
        parts: list[str] = []
        while True:
            if self._pos >= len(self._src):
                raise TokeniseError(f"unterminated {what}", self._pos, line, col)
            char = self._peek()
            if char == delimiter:
                if self._peek(1) == delimiter:
                    parts.append(delimiter)
                    self._advance(2)
                    continue
                self._advance()
                break
            parts.append(char)
            self._advance()
        return "".join(parts)

    def _scan_quoted_identifier(self, line: int, col: int) -> Token:
        text = self._scan_delimited(_DOUBLE_QUOTE, "quoted identifier", line, col)
        return Token(TokenType.IDENTIFIER, text, line, col)

    def _scan_string(self, line: int, col: int) -> Token:
        text = self._scan_delimited(_SINGLE_QUOTE, "string literal", line, col)
        return Token(TokenType.STRING, text, line, col)

    def _scan_number(self, line: int, col: int) -> Token:
        start = self._pos
        seen_dot = False
        seen_exp = False
        while True:
            char = self._peek()
            if char.isdigit():
                self._advance()
            elif char == "." and not seen_dot and not seen_exp:
                seen_dot = True
                self._advance()
            elif (
                char in "eE"
                and not seen_exp
                and (
                    self._peek(1).isdigit()
                    or (self._peek(1) in "+-" and self._peek(2).isdigit())
                )
            ):
                seen_exp = True
                self._advance(2 if self._peek(1) in "+-" else 1)
            else:
                break
        text = self._src[start : self._pos]
        # ``12abc`` must be an error rather than NUMBER(12) IDENTIFIER(abc); silently
        # splitting it would let a typo through as a valid-looking query.
        if self._peek().isalpha() or self._peek() == "_":
            raise TokeniseError(
                f"malformed number literal {text + self._peek()!r}", self._pos, line, col
            )
        return Token(TokenType.NUMBER, text, line, col)


def tokenise(source: str) -> list[Token]:
    """Convenience wrapper: tokenise ``source`` and return the token list."""
    return Tokeniser(source).tokenise()
