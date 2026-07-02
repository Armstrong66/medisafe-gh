"""
loader.py — Load and filter G-MASS probe JSONL files.
Owner: D  |  MediSafe-GH · Africa AI Safety Prize 2026
"""

from core.utils import load_jsonl
from core.logger import get_logger

logger = get_logger(__name__)

# Default probe file locations per language
PROBE_PATHS = {
    "english":      "data/probes/probes_en.jsonl",
    "twi":          "data/probes/probes_twi.jsonl",
    "ghanaian_en":  "data/probes/probes_gh_en.jsonl",
}


def load_probes(language: str = "english") -> list[dict]:
    """
    Load the full probe set for a given language condition.

    Args:
        language : "english", "twi", or "ghanaian_en"

    Returns:
        List of probe dicts with fields:
        probe_id, disease_domain, failure_category, prompt,
        language, validator, validation_status, notes
    """
    path = PROBE_PATHS.get(language)
    if not path:
        raise ValueError(
            f"Unknown language: '{language}'. "
            f"Valid options: {list(PROBE_PATHS.keys())}"
        )
    probes = load_jsonl(path)
    logger.info(f"Loaded {len(probes)} probes [{language}] from {path}")
    return probes


def load_probes_from_path(path: str) -> list[dict]:
    """Load probes from a custom JSONL path (e.g. pilot set)."""
    probes = load_jsonl(path)
    logger.info(f"Loaded {len(probes)} probes from {path}")
    return probes


def load_bilingual_probes(path: str) -> list[dict]:
    """
    Load probes that contain both english_prompt and twi_prompt fields
    side by side (e.g. GMASS_150-A_probes_twi.jsonl).

    Returns the raw bilingual records, unchanged.
    Each record has: probe_id, disease_domain, failure_category,
    english_prompt, twi_prompt, validation_status, notes.
    """
    probes = load_jsonl(path)
    logger.info(f"Loaded {len(probes)} bilingual probes from {path}")
    return probes


def expand_bilingual_probes(bilingual_probes: list[dict]) -> dict[str, list[dict]]:
    """
    Split a bilingual probe list into two separate single-language probe lists
    that share the same probe_id — needed for SDS comparison.

    IMPORTANT: every expanded record carries BOTH `prompt` (the language-specific
    text actually sent to the model) AND `english_prompt` (always English).
    This is required because GMassScorer.score_one() needs probe_prompt_en for
    LlamaGuard3 context even when scoring a Twi response — LlamaGuard3 judges
    response safety IN THE CONTEXT of what was asked, and that context prompt
    must be English regardless of which language the model was actually queried in.

    Per §1 of GMASS_Team_Clarifications.md: the Twi text used for `prompt` is
    resolved via resolve_twi_prompt() — preferring the human-validated version
    when available, falling back to the machine-translated draft otherwise.
    Both raw fields (`prompt_twi_draft`, `prompt_twi_validated`) are preserved
    unchanged on the expanded record for reproducibility, even though only
    one of them is selected into `prompt`.

    Backward compatible: if a record only has the older flat `twi_prompt`
    field (pre-clarifications schema), that value is used directly.

    Args:
        bilingual_probes : records with english_prompt + twi fields
                           (either the new prompt_twi_draft/prompt_twi_validated
                           pair, or the older flat twi_prompt field)

    Returns:
        {
            "english": [ {probe_id, disease_domain, failure_category,
                           prompt, english_prompt, language: "english"}, ... ],
            "twi":     [ {probe_id, disease_domain, failure_category,
                           prompt, english_prompt, language: "twi",
                           prompt_twi_draft, prompt_twi_validated,
                           translation_status}, ... ],
        }
    """
    english_probes = []
    twi_probes     = []

    for p in bilingual_probes:
        base = {
            "probe_id":         p["probe_id"],
            "disease_domain":   p["disease_domain"],
            "failure_category": p["failure_category"],
            "english_prompt":   p["english_prompt"],  # always carried, for scorer context
        }
        english_probes.append({**base, "language": "english", "prompt": p["english_prompt"]})

        twi_resolved, twi_status = resolve_twi_prompt(p)
        twi_probes.append({
            **base,
            "language":             "twi",
            "prompt":               twi_resolved,
            "prompt_twi_draft":     p.get("prompt_twi_draft", p.get("twi_prompt")),
            "prompt_twi_validated": p.get("prompt_twi_validated"),
            "translation_status":   twi_status,
        })

    logger.info(
        f"Expanded {len(bilingual_probes)} bilingual probes → "
        f"{len(english_probes)} english + {len(twi_probes)} twi"
    )
    return {"english": english_probes, "twi": twi_probes}


def resolve_twi_prompt(probe: dict) -> tuple[str, str]:
    """
    Resolve which Twi text to actually send to a model, per §1 of
    GMASS_Team_Clarifications.md.

    Resolution order:
        1. prompt_twi_validated  — human-corrected, used if present and non-empty
        2. prompt_twi_draft      — raw GhanaNLP/Khaya machine translation
        3. twi_prompt            — backward-compat flat field (older schema,
                                   e.g. GMASS_150-A_probes_twi.jsonl), treated
                                   as an unvalidated draft

    CRITICAL per §1: this function NEVER mutates or overwrites the source
    fields — it only SELECTS which one to use for the model call. Both the
    draft and the validated correction remain intact on the probe record as
    separate, permanent data points (lets the team measure how often machine
    translation needed correction, and keeps the dataset reproducible).

    Args:
        probe : a single bilingual probe record

    Returns:
        (resolved_text, status) where status is one of:
            "validated"        — human-corrected version was used
            "draft_unreviewed" — no validated version yet, draft used as-is
            "legacy_flat"      — older schema with only twi_prompt, no
                                  draft/validated distinction recorded

    Example:
        text, status = resolve_twi_prompt(probe)
        if status == "draft_unreviewed":
            logger.warning(f"{probe['probe_id']}: using unreviewed Twi draft")
    """
    validated = probe.get("prompt_twi_validated")
    if validated:
        return validated, "validated"

    draft = probe.get("prompt_twi_draft")
    if draft:
        return draft, "draft_unreviewed"

    # Backward compatibility with the pre-clarifications flat schema
    legacy = probe.get("twi_prompt")
    if legacy:
        return legacy, "legacy_flat"

    raise KeyError(
        f"Probe {probe.get('probe_id', '?')} has no usable Twi text — "
        f"expected one of: prompt_twi_validated, prompt_twi_draft, twi_prompt"
    )


def translation_correction_rate(bilingual_probes: list[dict]) -> dict:
    """
    Per §1: measure how often human validators needed to correct the
    machine-translated draft. This is itself a methodology data point for
    the submission ("X% of GhanaNLP/Khaya drafts required human correction").

    Only meaningful for probes that HAVE both a draft and a validated
    field recorded (new schema) — legacy flat-field probes are excluded
    from the denominator since no draft/correction distinction exists for them.

    Returns:
        {
            "total_with_draft_and_validation": int,
            "corrected_count":                  int,   # validated != draft
            "unchanged_count":                  int,   # validated == draft
            "correction_rate_pct":               float,
            "still_unreviewed_count":            int,   # has draft, no validated yet
        }
    """
    total_dual   = 0
    corrected    = 0
    unchanged    = 0
    unreviewed   = 0

    for p in bilingual_probes:
        draft     = p.get("prompt_twi_draft")
        validated = p.get("prompt_twi_validated")

        if draft is None:
            continue  # legacy schema, no draft recorded — not comparable

        if validated is None:
            unreviewed += 1
            continue

        total_dual += 1
        if validated.strip() != draft.strip():
            corrected += 1
        else:
            unchanged += 1

    rate = round(corrected / total_dual * 100, 2) if total_dual else 0.0

    summary = {
        "total_with_draft_and_validation": total_dual,
        "corrected_count":                  corrected,
        "unchanged_count":                  unchanged,
        "correction_rate_pct":              rate,
        "still_unreviewed_count":           unreviewed,
    }
    logger.info(
        f"Translation correction rate: {rate}% "
        f"({corrected}/{total_dual} drafts corrected, "
        f"{unreviewed} still unreviewed)"
    )
    return summary


def filter_by_domain(probes: list[dict], domain: str) -> list[dict]:
    """Return only probes matching the given disease domain."""
    return [p for p in probes if p.get("disease_domain") == domain]


def filter_by_category(probes: list[dict], category: str) -> list[dict]:
    """Return only probes matching the given failure category."""
    return [p for p in probes if p.get("failure_category") == category]


def filter_approved(probes: list[dict]) -> list[dict]:
    """Return only probes with validation_status == 'Approved'."""
    return [p for p in probes if p.get("validation_status") == "Approved"]


def build_pilot_set(probes: list[dict], per_domain: int = 5) -> list[dict]:
    """
    Build a balanced pilot set with `per_domain` probes from each disease domain.

    Args:
        probes     : full probe list
        per_domain : how many probes to take from each domain

    Returns:
        Balanced pilot list.
    """
    domain_counts: dict[str, int] = {}
    pilot: list[dict] = []
    for probe in probes:
        domain = probe.get("disease_domain", "")
        if domain_counts.get(domain, 0) < per_domain:
            pilot.append(probe)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    logger.info(
        f"Built pilot set: {len(pilot)} probes "
        f"({per_domain}/domain across {len(domain_counts)} domains)"
    )
    return pilot
