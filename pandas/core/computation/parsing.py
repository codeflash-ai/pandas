"""
:func:`~pandas.eval` source string parsing functions
"""

from __future__ import annotations

from enum import Enum
from io import StringIO
from keyword import iskeyword
import token
import tokenize
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterator,
    )

_ascii_escape_cache = dict()

# A token value Python's tokenizer probably will never use.
BACKTICK_QUOTED_STRING = 100


def create_valid_python_identifier(name: str) -> str:
    """
    Create valid Python identifiers from any string.

    Check if name contains any special characters. If it contains any
    special characters, the special characters will be replaced by
    a special string and a prefix is added.

    Raises
    ------
    SyntaxError
        If the returned name is not a Python valid identifier, raise an exception.
    """
    if name.isidentifier() and not iskeyword(name):
        return name

    # Escape characters that fall outside the ASCII range (U+0001..U+007F).
    # GH 49633
    # Precompute the escaped replacements and use a local variable for replacements for speed
    gen = []
    append_gen = gen.append
    for c in name:
        if c in _ascii_escape_cache:
            c_escaped = _ascii_escape_cache[c]
        else:
            c_escaped = "".join(chr(b) for b in c.encode("ascii", "backslashreplace"))
            _ascii_escape_cache[c] = c_escaped
        append_gen((c, c_escaped))

    name = []
    append_name = name.append
    for c, c_escaped in gen:
        if c != c_escaped:
            append_name(c_escaped.replace("\\", "_UNICODE_"))
        else:
            append_name(c_escaped.replace("\\", "_BACKSLASH_"))
    name = "".join(name)

    # Construct the replacement dictionary only once and reuse
    # Use a tuple to store in the closure for subsequent calls
    if not hasattr(create_valid_python_identifier, "_special_characters_replacements"):
        replacements = {
            char: f"_{token.tok_name[tokval]}_"
            for char, tokval in tokenize.EXACT_TOKEN_TYPES.items()
        }
        replacements.update(
            {
                " ": "_",
                "?": "_QUESTIONMARK_",
                "!": "_EXCLAMATIONMARK_",
                "$": "_DOLLARSIGN_",
                "€": "_EUROSIGN_",
                "°": "_DEGREESIGN_",
                "'": "_SINGLEQUOTE_",
                '"': "_DOUBLEQUOTE_",
                "#": "_HASH_",
                "`": "_BACKTICK_",
            }
        )
        create_valid_python_identifier._special_characters_replacements = replacements
    special_characters_replacements = (
        create_valid_python_identifier._special_characters_replacements
    )

    # Use list comprehension for high performance and memory efficiency

    name = "".join([special_characters_replacements.get(char, char) for char in name])
    name = f"BACKTICK_QUOTED_STRING_{name}"

    if not name.isidentifier():
        raise SyntaxError(f"Could not convert '{name}' to a valid Python identifier.")

    return name


def clean_backtick_quoted_toks(tok: tuple[int, str]) -> tuple[int, str]:
    """
    Clean up a column name if surrounded by backticks.

    Backtick quoted string are indicated by a certain tokval value. If a string
    is a backtick quoted token it will processed by
    :func:`_create_valid_python_identifier` so that the parser can find this
    string when the query is executed.
    In this case the tok will get the NAME tokval.

    Parameters
    ----------
    tok : tuple of int, str
        ints correspond to the all caps constants in the tokenize module

    Returns
    -------
    tok : Tuple[int, str]
        Either the input or token or the replacement values
    """
    toknum, tokval = tok
    if toknum == BACKTICK_QUOTED_STRING:
        return tokenize.NAME, create_valid_python_identifier(tokval)
    return toknum, tokval


def clean_column_name(name: Hashable) -> Hashable:
    """
    Function to emulate the cleaning of a backtick quoted name.

    The purpose for this function is to see what happens to the name of
    identifier if it goes to the process of being parsed a Python code
    inside a backtick quoted string and than being cleaned
    (removed of any special characters).

    Parameters
    ----------
    name : hashable
        Name to be cleaned.

    Returns
    -------
    name : hashable
        Returns the name after tokenizing and cleaning.

    Notes
    -----
        For some cases, a name cannot be converted to a valid Python identifier.
        In that case :func:`tokenize_string` raises a SyntaxError.
        In that case, we just return the name unmodified.

        If this name was used in the query string (this makes the query call impossible)
        an error will be raised by :func:`tokenize_backtick_quoted_string` instead,
        which is not caught and propagates to the user level.
    """
    try:
        # Escape backticks
        if isinstance(name, str) and "`" in name:
            name = name.replace("`", "``")

        tokenized = tokenize_string(f"`{name}`")
        tokval = next(tokenized)[1]
        return create_valid_python_identifier(tokval)
    except SyntaxError:
        return name


def tokenize_backtick_quoted_string(
    token_generator: Iterator[tokenize.TokenInfo], source: str, string_start: int
) -> tuple[int, str]:
    """
    Creates a token from a backtick quoted string.

    Moves the token_generator forwards till right after the next backtick.

    Parameters
    ----------
    token_generator : Iterator[tokenize.TokenInfo]
        The generator that yields the tokens of the source string (Tuple[int, str]).
        The generator is at the first token after the backtick (`)

    source : str
        The Python source code string.

    string_start : int
        This is the start of backtick quoted string inside the source string.

    Returns
    -------
    tok: Tuple[int, str]
        The token that represents the backtick quoted string.
        The integer is equal to BACKTICK_QUOTED_STRING (100).
    """
    for _, tokval, start, _, _ in token_generator:
        if tokval == "`":
            string_end = start[1]
            break

    return BACKTICK_QUOTED_STRING, source[string_start:string_end]


class ParseState(Enum):
    DEFAULT = 0
    IN_BACKTICK = 1
    IN_SINGLE_QUOTE = 2
    IN_DOUBLE_QUOTE = 3


def _split_by_backtick(s: str) -> list[tuple[bool, str]]:
    """
    Splits a str into substrings along backtick characters (`).

    Disregards backticks inside quotes.

    Parameters
    ----------
    s : str
        The Python source code string.

    Returns
    -------
    substrings: list[tuple[bool, str]]
        List of tuples, where each tuple has two elements:
        The first is a boolean indicating if the substring is backtick-quoted.
        The second is the actual substring.
    """
    substrings = []
    substr: list[str] = []  # Will join into a string before adding to `substrings`
    i = 0
    parse_state = ParseState.DEFAULT
    while i < len(s):
        char = s[i]

        match char:
            case "`":
                # start of a backtick-quoted string
                if parse_state == ParseState.DEFAULT:
                    if substr:
                        substrings.append((False, "".join(substr)))

                    substr = [char]
                    i += 1
                    parse_state = ParseState.IN_BACKTICK
                    continue

                elif parse_state == ParseState.IN_BACKTICK:
                    # escaped backtick inside a backtick-quoted string
                    next_char = s[i + 1] if (i != len(s) - 1) else None
                    if next_char == "`":
                        substr.append(char)
                        substr.append(next_char)
                        i += 2
                        continue

                    # end of the backtick-quoted string
                    else:
                        substr.append(char)
                        substrings.append((True, "".join(substr)))

                        substr = []
                        i += 1
                        parse_state = ParseState.DEFAULT
                        continue
            case "'":
                # start of a single-quoted string
                if parse_state == ParseState.DEFAULT:
                    parse_state = ParseState.IN_SINGLE_QUOTE
                # end of a single-quoted string
                elif (parse_state == ParseState.IN_SINGLE_QUOTE) and (s[i - 1] != "\\"):
                    parse_state = ParseState.DEFAULT
            case '"':
                # start of a double-quoted string
                if parse_state == ParseState.DEFAULT:
                    parse_state = ParseState.IN_DOUBLE_QUOTE
                # end of a double-quoted string
                elif (parse_state == ParseState.IN_DOUBLE_QUOTE) and (s[i - 1] != "\\"):
                    parse_state = ParseState.DEFAULT
        substr.append(char)
        i += 1

    if substr:
        substrings.append((False, "".join(substr)))

    return substrings


def tokenize_string(source: str) -> Iterator[tuple[int, str]]:
    """
    Tokenize a Python source code string.

    Parameters
    ----------
    source : str
        The Python source code string.

    Returns
    -------
    tok_generator : Iterator[Tuple[int, str]]
        An iterator yielding all tokens with only toknum and tokval (Tuple[ing, str]).
    """
    # GH 59285
    # Escape characters, including backticks
    fragments = []
    append_fragments = fragments.append
    for is_backtick_quoted, substring in _split_by_backtick(source):
        if is_backtick_quoted:
            # Use slice only if substring has >2 characters (performance, but safe for correctness)
            # The substring will always have at least two backticks (start/end) so slicing [1:-1] is correct
            core = substring[1:-1]
            processed = create_valid_python_identifier(core)
            append_fragments(processed)
        else:
            append_fragments(substring)
    source = "".join(fragments)

    line_reader = StringIO(source).readline
    token_generator = tokenize.generate_tokens(line_reader)

    # Use local variable in for loop for performance
    for token_tuple in token_generator:
        yield token_tuple[0], token_tuple[1]
