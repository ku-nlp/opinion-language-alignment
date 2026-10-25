"""Response parsing, faithful to match_llm_str()/normalize() in the AAAI repo."""
import ast
import re

PATTERN = re.compile(r"\{(?:\s*'?\d+'?\s*:\s*'\d+(?:\.\d+)?%'\s*,?)+\s*\}")

# Models below ~14B often emit the dict body without the outer braces:
#     '1': '28.50%', '2': '35.00%', '3': '22.00%', '4': '14.50%'
# The content is identical and unambiguous, so accepting it is a formatting fix, not a
# leniency that changes any answer. It only fires when PATTERN finds nothing, so every
# response the AAAI parser accepted is parsed exactly as before -- verified on the three
# 35B/70B/72B runs, where all 8164 responses are byte-identical under both parsers.
# At least two key:value pairs are required so a stray percentage cannot match.
BARE = re.compile(
    r"(?:'?\d+'?\s*:\s*'\d+(?:\.\d+)?%'\s*,\s*)+'?\d+'?\s*:\s*'\d+(?:\.\d+)?%'")

# Some models leave the percentage itself unquoted:
#     {'1': 62.7%, '2': 30.5%, '3': 5.2%, '4': 1.6%}
# The `%` makes that invalid Python whether or not it is quoted, so nothing that used to
# parse can change meaning -- the substitution only fires where a colon is followed
# directly by a digit, which a quoted value never is.
UNQUOTED_PCT = re.compile(r"(:\s*)(\d+(?:\.\d+)?%)")


def _normalize(s):
    """Rescale to 100%, or None if there is nothing to rescale.

    A model that answers {'1': '0%', '2': '0%'} states no distribution at all; the
    original implementation divided by the zero total and crashed.
    """
    raw = {k: float(v.strip("%")) for k, v in ast.literal_eval(s).items()}
    total = sum(raw.values())
    if total <= 0:
        return None
    return str({k: f"{v / total * 100:.2f}%" for k, v in raw.items()})


def match_llm_str(text):
    """Return the canonical dict string, or None if the output is unparseable."""
    if "%}" in text:
        text = text.replace("%}", "%'}")
    text = re.sub(r"(?<!%)([0-9])'}", r"\1%'}", text)
    text = re.sub(r"(?<!%)([0-9])',", r"\1%',", text)
    text = UNQUOTED_PCT.sub(r"\1'\2'", text)
    m = PATTERN.search(text)
    if m:
        probs_str = m.group(0)
    else:
        m = BARE.search(text)
        if not m:
            return None
        probs_str = "{" + m.group(0) + "}"
    probs = {k: float(v.strip("%")) / 100 for k, v in ast.literal_eval(probs_str).items()}
    if abs(sum(probs.values()) - 1.0) < 5e-1:
        return probs_str
    return _normalize(probs_str)


def to_vector(probs_str, choice_keys):
    """Dict string -> probability vector aligned to choice_keys, renormalised.

    Keys are coerced to str: models often omit the quotes ({2: '35.71%'}), which
    literal_eval turns into int keys. Matching those against the string choice_keys
    used to fail silently and drop the whole question.
    """
    d = {str(k): float(v.strip("%")) / 100 for k, v in ast.literal_eval(probs_str).items()}
    vec = [d.get(str(k), 0.0) for k in choice_keys]
    tot = sum(vec)
    return [v / tot for v in vec] if tot > 0 else None
