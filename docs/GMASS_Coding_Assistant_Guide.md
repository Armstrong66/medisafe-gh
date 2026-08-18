# G-MASS Coding Assistant — Versioning & Commit Guide
**For automated assistants (Claude, Cursor, Copilot, etc.) working on the medisafe-gh repo**

---

## Your Role and Identity Rules

You are assisting with code, not authoring it. Follow these identity rules exactly:

```bash
# NEVER set yourself as the commit author.
# The human developer's git config is already set:
#   git config user.name  "Derrick [Surname]"
#   git config user.email "[email]"
# All commits go out under that identity automatically.
# You write the commit message content. The human presses enter.

# If asked to run git commands directly, use:
git commit --author="$(git config user.name) <$(git config user.email)>"
# This ensures the commit records the human, not you.
```

When writing commit messages, append a trailer line to acknowledge AI assistance without making yourself a contributor:

```
feat(scoring): add embedding-based referral detector

Replace keyword dictionary with cosine similarity on sentence-BERT anchors.
Threshold calibrated at 0.45 on pilot transcripts (recall=0.87, precision=0.82).

Results-comparability: RAR values not directly comparable to v1.0 keyword baseline.

Co-authored-by: AI-assisted <noreply@anthropic.com>
```

The `Co-authored-by` trailer is a GitHub convention that appears in the commit but does not add a contributor to the repository contributor graph unless an actual GitHub account is linked. Use `noreply@anthropic.com` (or `noreply@cursor.sh` for Cursor) — these resolve to nothing and keep the contributor list clean.

---

## Branch Structure

The repo currently has three branches. Understand each one's purpose before touching anything:

```
main          ← stable, production-ready code only
              ← all version tags (v1.0.0, v1.1.0, etc.) live here
              ← what pip install git+... pulls
              ← what HuggingFace Spaces deploys from
              ← NEVER commit unreviewed work directly here

dev           ← integration branch
              ← all feature branches merge here first
              ← tested and reviewed before merging to main
              ← no version tags here

feature/*     ← individual work branches
  feature/A-evaluation-pipeline
  feature/B-probe-design
  feature/C-twi-translation
  feature/D-scoring-pipeline
  (or whatever names the team is using)
              ← short-lived, one task per branch
              ← merge to dev when complete
```

**Rule for your commit placement:**
- Write code changes on `feature/*` or `dev`
- Only suggest merging to `main` when explicitly asked AND changes are complete
- Never push directly to `main`
- Never create tags on `feature/*` or `dev` — tags only go on `main`

---

## Versioning Rules You Must Know

Read `GMASS_Versioning_Roadmap.md` fully before making any version-related decisions. Summary:

```
PATCH  v1.0.x  — bug fix only, no scoring logic change, results fully comparable
MINOR  v1.x.0  — new feature or new scorer logic, results may not be comparable
MAJOR  v2.0.0  — protocol/interface change, backwards incompatibility
```

**Critical G-MASS-specific rule:** If you change ANY of the following, it is AT MINIMUM a `MINOR` bump, and you must add a results-comparability warning in the commit and CHANGELOG:

- `ReferralDetector` logic
- `HallucinationDetector` logic
- Any scorer (LlamaGuard3, AfroLM, Gemma3) version or configuration
- Scoring thresholds (`sds_deploy_ready_pp`, `rar_target_pct`, `scorer_confidence_threshold`)
- Probe set content (any probe added, removed, or modified)
- fastText routing logic

**Do not bump the version yourself.** Write the commit message with the recommended bump type noted, and let the human apply the tag. Your job is to label it correctly, not to execute it.

---

## Step 1: Audit Existing Uncommitted or Untagged Work

Before writing any new code, run this audit to understand what has already happened in the repo since v1.0.0:

```bash
# See all commits since v1.0.0 that have no tag
git log v1.0.0..HEAD --oneline --all

# See commits with their full messages
git log v1.0.0..HEAD --format="%H %s" --all

# See what files changed in each commit
git log v1.0.0..HEAD --stat --all

# See all existing tags
git tag -l --sort=version:refname

# See which commits are on which branch
git log --oneline --graph --all --decorate
```

After running these, produce a table in this format for the human to review:

```markdown
## Commit Audit since v1.0.0

| Commit SHA | Branch | Summary | Recommended Version Bump | Reason |
|---|---|---|---|---|
| abc1234 | dev | Add embedding-based referral detector | MINOR (v1.1.0) | Changes ReferralDetector scoring logic — RAR not comparable |
| def5678 | feature/D | Fix broken import in scorer.py | PATCH (v1.0.1) | Bug fix only, no logic change |
| ghi9012 | main | Add Gradio Tab 3 with pilot results | MINOR (v1.1.0) | New feature, backwards-compatible |
```

Do not guess. If you cannot determine whether a change is a PATCH or MINOR from the diff alone, flag it explicitly:

```markdown
| xyz3456 | dev | Update scorer thresholds | ⚠ UNCLEAR — needs human review | Threshold change may affect result comparability |
```

---

## Step 2: Retroactive Tagging Plan

Once the audit table is reviewed and confirmed by the human, produce the tagging commands in the correct order. Tags must be applied chronologically — earlier commits get lower version numbers.

**Template output for the human to run:**

```bash
# ── Retroactive tags — run in this order ──────────────────────

# v1.0.1 — Bug fix: broken import in scorer.py
git tag -a v1.0.1 def5678 -m "fix: broken import in scorer.py

Patch release — bug fix only.
No scoring logic changes. Results fully comparable to v1.0.0.

Changes:
- Fix ModuleNotFoundError in medisafe_gh.scoring.scorer on Python 3.11
"

# v1.1.0 — Embedding-based referral detector + Gradio upgrades
git tag -a v1.1.0 abc1234 -m "feat: embedding-based referral/hallucination detection + Gradio demo upgrade

Minor release — new features, backwards-compatible CLI.
⚠ Results comparability: RAR values computed with v1.1 semantic detection
are NOT directly comparable to v1.0 keyword-based RAR. Document version
when reporting results.

Changes:
- Replace ReferralDetector keyword dict with sentence-BERT cosine similarity
- Add HallucinationDetector semantic scoring (AfroLM for Twi, MiniLM for EN)
- Gradio app: add Tab 3 benchmark results with pilot data
- Gradio app: add Tab 4 About section with citation block
- Add sentence-transformers dependency
"

# Push all tags at once
git push origin --tags
```

**Never run `git push origin --tags` yourself.** Output the commands and let the human execute them. Tag operations on remote repos can be destructive if wrong.

---

## Step 3: Commit Message Standards Going Forward

Every commit you write must follow this format exactly:

```
<type>(<scope>): <short description in imperative mood, max 72 chars>

<body — what changed and why, not how>
<blank line if body present>
<results-comparability note if applicable>
<blank line>
<trailers>
```

### Types
```
feat      new feature (MINOR bump candidate)
fix       bug fix (PATCH bump candidate)
docs      documentation only
refactor  code restructure, no behaviour change (PATCH)
test      adding or fixing tests
chore     build, CI, dependency updates
perf      performance improvement, no API change
break     breaking change (MAJOR bump candidate — use rarely, flag loudly)
```

### Scopes
```
scoring   changes to scorer.py, ensemble logic, detectors
metrics   changes to metrics.py (CSR/SDS/RAR)
probes    changes to probe set, builder, loader
cli       changes to cli.py or command behaviour
config    changes to gmass_config.yaml or models.yaml
gradio    changes to app.py (Gradio demo)
pipeline  changes to evaluate.py or pipeline.py
registry  changes to registry.py (v2.x only)
docs      README, CHANGELOG, deployment plan
ci        GitHub Actions, setup.sh, Dockerfile
```

### Good commit message examples

```
fix(scoring): correct AfroLM language code from 'tw' to 'twi'

fastText LID returns '__label__twi' not '__label__tw' for Twi.
Routing was silently falling back to LlamaGuard3 for all Twi responses.
No change to scoring logic — PATCH release.

Co-authored-by: AI-assisted <noreply@anthropic.com>
```

```
feat(gradio): add single probe tester with colour-coded verdict card

Tab 1 now accepts a query, language, model, and failure category.
Returns safety label, referral flag, hallucination flag, and scorer
agreement as an HTML verdict card.

Backwards-compatible — no CLI or scoring changes.
Recommended bump: MINOR (v1.1.0)

Co-authored-by: AI-assisted <noreply@anthropic.com>
```

```
feat(scoring): replace keyword referral detector with sentence-BERT cosine similarity

⚠ RESULTS COMPARABILITY: RAR values computed with this change are NOT
directly comparable to v1.0 keyword-based RAR. When reporting results,
specify scorer version (v1.1.0+).

Threshold: 0.45 (calibrated on 60-probe pilot sample, recall=0.87)
New dependency: sentence-transformers>=2.6.0
Embedding model: all-MiniLM-L6-v2 (EN/GH-EN), AfroLM CLS (Twi)

Recommended bump: MINOR (v1.1.0)

Co-authored-by: AI-assisted <noreply@anthropic.com>
```

```
break(pipeline): introduce ModelCaller and SafetyScorer abstract protocols

Pipeline no longer imports model callers directly — all models must be
registered via GMassRegistry before use. Existing models (GPT-4o mini,
Gemini Flash, LLaMA, Phi-3, BioMistral) are pre-registered in
medisafe_gh/__init__.py for backwards compatibility with existing CLI usage.

Migration: custom code calling call_model() directly must update to
GMassRegistry.register_model() + GMassPipeline.run() pattern.

See MIGRATION_v2.md for full upgrade guide.

Recommended bump: MAJOR (v2.0.0)

Co-authored-by: AI-assisted <noreply@anthropic.com>
```

---

## Step 4: CHANGELOG.md Maintenance

Keep `CHANGELOG.md` in the repo root. Update it with every version change. You write the content — the human commits it.

Format:
```markdown
# Changelog

All notable changes to G-MASS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com).
Versioning follows [Semantic Versioning](https://semver.org).

---

## [Unreleased]
### Added
- (list features merged to dev but not yet tagged on main)

---

## [v1.1.0] — YYYY-MM-DD
### Added
- Embedding-based referral detection using sentence-BERT cosine similarity
- Embedding-based hallucination detection using AfroLM (Twi) and MiniLM (EN)
- Gradio Tab 3: Benchmark Results with pilot CSR/SDS/RAR charts
- Gradio Tab 4: About G-MASS with citation block

### Changed
- ReferralDetector: keyword dict replaced with semantic anchor matching
- HallucinationDetector: LlamaGuard3 M1 category replaced with cosine similarity

### ⚠ Results Comparability
RAR values computed with v1.1 are NOT directly comparable to v1.0.
Always specify `gmass_version: v1.1.0` in published results.

---

## [v1.0.1] — YYYY-MM-DD
### Fixed
- AfroLM language routing: corrected ISO code from 'tw' to 'twi'
- setup.sh: fixed Ollama binary download 404 (tarball URL)
- pyproject.toml: removed duplicate license field causing TOML parse error

### Notes
Full results comparability with v1.0.0 maintained.

---

## [v1.0.0] — 2026-06-30
### Initial Release
- Africa AI Safety Prize Competition 2026 submission (3rd Place, Track II)
- 300 probe set: 6 disease domains × 3 failure categories × EN + GH-EN + Twi
- Scorer ensemble: LlamaGuard3-1B · AfroLM · Gemma3-1B · fastText LID
- Metrics: CSR · SDS · RAR
- CLI: gmass evaluate, score, profile, report, probe, combine
- HuggingFace dataset: BioinstLab/GMASS-probe-set-v1.0
- Gradio demo: huggingface.co/spaces/BioinstLab/gmass-demo
```

---

## Step 5: Branch Merge Protocol

When work on a feature branch is complete and ready to merge:

### Feature → Dev (your normal workflow)
```bash
# You write the merge commit message:
git checkout dev
git merge --no-ff feature/D-scoring-pipeline -m "merge(feature/D): embedding-based detectors ready for integration

Merges feature/D-scoring-pipeline into dev.
Changes peer-reviewed and tested locally.
No version tag yet — awaiting integration test on dev.

Includes:
- sentence-BERT referral detector (v1.1.0 candidate)
- AfroLM hallucination detector (v1.1.0 candidate)
- Updated tests: test_scorer.py coverage 87%
"
```

### Dev → Main (suggest only, never execute)
When dev is stable and ready for release, produce this for the human:

```bash
# Human runs these — do not execute yourself:
git checkout main
git merge --no-ff dev -m "release: merge dev into main for v1.1.0

Integration-tested on dev branch.
All tests passing. CHANGELOG updated. pyproject.toml version bumped.
"
git tag -a v1.1.0 HEAD -m "v1.1.0 — embedding-based semantic detectors

See CHANGELOG.md for full details.
⚠ RAR values not comparable to v1.0 — see results-comparability note.
"
git push origin main
git push origin v1.1.0
```

---

## Quick Reference — What Version Bump to Suggest

When you make a change, determine the correct bump before writing the commit:

```
Did you change ReferralDetector or HallucinationDetector logic?   → MINOR ⚠ comparability note
Did you change a scorer model version or configuration?            → MINOR ⚠ comparability note
Did you change a metric threshold (SDS threshold, RAR target)?     → MINOR ⚠ comparability note
Did you add a new probe or modify an existing one?                 → MINOR + probe set version bump
Did you add a new language or disease domain?                      → MINOR + probe set version bump
Did you add a new CLI command or Gradio tab?                       → MINOR
Did you fix a bug with no logic change?                            → PATCH
Did you update documentation or comments only?                     → PATCH
Did you rename a function, class, or CLI argument?                 → MAJOR if public API
Did you introduce abstract protocols or registry pattern?          → MAJOR
Did you change the JSONL schema (field names, required fields)?    → MAJOR
Not sure?                                                          → flag for human review
```

---

## Things You Must Never Do

```
✗ Never push to main directly
✗ Never create or push tags without human confirmation
✗ Never set yourself as git author or change git user config
✗ Never modify the probe set JSONL files without flagging it as a probe set version bump
✗ Never bump pyproject.toml version without also updating CHANGELOG.md
✗ Never merge dev → main without confirming all tests pass
✗ Never silently change scoring logic — always add comparability warning to commit
✗ Never delete a git tag (tags are permanent scientific records for this project)
```

---

*G-MASS Coding Assistant Guide v1.0 · MediSafe-GH · For AI coding assistants working on Armstrong66/medisafe-gh*
