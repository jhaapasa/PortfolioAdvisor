from __future__ import annotations


def slugify(text: str) -> str:
    """Convert arbitrary text into a lowercase, dash-delimited slug.

    Rules:
    - Lowercase
    - Non-alphanumeric -> '-'
    - Collapse repeated dashes
    - Trim leading/trailing dashes
    - Return 'none' when empty
    """
    s = (text or "").lower()
    out: list[str] = []
    prev_dash = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("-")
            prev_dash = True
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "none"


def instrument_id_to_slug(instrument_id: str) -> str:
    """Slug for instrument identity used in file/folder names.

    Example: 'cid:stocks:us:XNAS:AAPL' -> 'cid-stocks-us-xnas-aapl'
    """
    return slugify(instrument_id)


