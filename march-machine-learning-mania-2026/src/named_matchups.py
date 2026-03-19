"""
Resolve men's team display names to Kaggle TeamIDs (via ``MTeamSpellings.csv``),
build matchup feature rows, and run logreg + XGB predictions for ad-hoc games.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src import config
from src.matchup_builder import build_submission_matchups
from src.train_logreg import FEATURES as XGB_FEATURE_ORDER


def _clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(p, config.SUBMISSION_PROB_MIN, config.SUBMISSION_PROB_MAX)


def _predict_from_matchup_dataframe(
    matchup_df: pd.DataFrame,
    *,
    logreg_model_path: Optional[Path] = None,
    logreg_features_path: Optional[Path] = None,
    xgb_model_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    write_csv: bool = False,
    id_column: str = "ID",
) -> pd.DataFrame:
    """
    Same logic as ``predict_2026.predict_from_matchup_dataframe`` but defined
    here so this module loads even if an older ``predict_2026`` lacks that helper.
    """
    logreg_model_path = Path(logreg_model_path or config.MODELS_DIR / "logreg_men.pkl")
    logreg_features_path = Path(logreg_features_path or config.LOGREG_FEATURE_NAMES_JSON)
    xgb_model_path = Path(xgb_model_path or config.MODELS_DIR / "xgb_men.pkl")

    logreg_features: list[str] = json.loads(
        logreg_features_path.read_text(encoding="utf-8")
    )
    logreg_model = joblib.load(logreg_model_path)
    xgb_model = joblib.load(xgb_model_path)

    X_lr = matchup_df[logreg_features]
    X_xgb = matchup_df[list(XGB_FEATURE_ORDER)]

    if X_lr.isna().any().any() or X_xgb.isna().any().any():
        n_lr = int(X_lr.isna().any(axis=1).sum())
        n_xgb = int(X_xgb.isna().any(axis=1).sum())
        raise ValueError(
            f"NaNs in feature matrix (logreg rows with any NaN: {n_lr}, "
            f"xgb rows with any NaN: {n_xgb}). Check team_stats coverage."
        )

    pred_logreg = _clip_probs(logreg_model.predict_proba(X_lr)[:, 1].astype(np.float64))
    pred_xgb = _clip_probs(xgb_model.predict_proba(X_xgb)[:, 1].astype(np.float64))

    out = pd.DataFrame(
        {
            id_column: matchup_df[id_column],
            "pred_logreg": pred_logreg,
            "pred_xgb": pred_xgb,
        }
    )
    if write_csv:
        out_path = Path(output_path or config.PREDICTIONS_2026_LOGREG_XGB)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
    return out

# User-provided 2026 Round of 64 lines (first team listed = Team A in ``ID`` / model).
FIRST_ROUND_2026_LINES = r"""
(8) Ohio State vs. (9) TCU | 12:15 p.m. | CBS
(4) Nebraska vs. (13) Troy | 12:40 p.m. | truTV
(6) Louisville vs. (11) South Florida | 1:30 p.m. | TNT
(5) Wisconsin vs. (12) High Point | 1:50 p.m. | TBS
(1) Duke vs. (16) Siena | 2:50 p.m. | CBS
(5) Vanderbilt vs. (12) McNeese | 3:15 p.m. | truTV
(3) Michigan State vs. (14) North Dakota State | 4:05 p.m. | TNT
(4) Arkansas vs. (13) Hawai'i | 4:25 p.m. | TBS
(6) North Carolina vs. (11) VCU | 6:50 p.m. | TNT
(1) Michigan vs. Howard | 7:10 p.m. | CBS
(6) BYU vs. Texas | 7:25 p.m. | TBS
(7) Saint Mary's vs. (10) Texas A&M | 7:35 p.m. | truTV
(3) Illinois vs. (14) Penn | 9:25 p.m. | TNT
(8) Georgia vs. (9) Saint Louis | 9:45 p.m. | CBS
(3) Gonzaga vs. (14) Kennesaw State | 10 p.m. | TNT
(2) Houston vs. (15) Idaho | 10:10 p.m. | truTV
Friday, March 20 (First Round/Round of 64)
(7) Kentucky vs. (10) Santa Clara | 12:15 p.m. | CBS
(5) Texas Tech vs. (12) Akron | 12:40 p.m. | truTV
(1) Arizona vs. (16) Long Island University | 1:35 p.m. | TNT
(3) Virginia vs. (14) Wright State | 1:50 p.m. | TBS
(2) Iowa State vs. (15) Tennessee State | 2:50 p.m. | CBS
(4) Alabama vs. (13) Hofstra | 3:15 p.m. | truTV
(8) Villanova vs. (9) Utah State | 4:10 p.m. | TNT
(6) Tennessee vs. (11) Miami (Ohio) | 4:25 p.m. | TBS
(8) Clemson vs. (9) Iowa | 6:50 p.m. | TNT
(5) St. John's vs. (12) UNI | 7:10 p.m. | CBS
(7) UCLA vs. (10) UCF | 7:25 p.m. | TBS
(2) Purdue vs. (15) Queens | 7:35 p.m. | truTV
(1) Florida vs. (16) Prairie View A&M | 9:25 p.m. | TNT
(4) Kansas vs. (13) Cal Baptist | 9:45 p.m. | CBS
(2) UConn vs. (15) Furman | 10 p.m. | TBS
(7) Miami (Fla.) vs. (10) Missouri | 10:10 p.m. | truTV
""".strip()

_VS_SPLIT = re.compile(r"\s+vs\.?\s+", re.IGNORECASE)
_SEED_PREFIX = re.compile(r"^\(\d+\)\s+")
_SEED_EXTRACT = re.compile(r"^\((\d+)\)\s+")

# Normalized alias → normalized spelling-table key (must exist in ``MTeamSpellings``).
_EXTRA_ALIASES: dict[str, str] = {
    "uni": "northern iowa",
    "st johns": "st john's",
    "st. johns": "st john's",
    "st. john's": "st john's",
    "queens": "queens nc",
    "ucf": "central florida",
    # Disambiguate between Miami (FL) and Miami (OH).
    "miami": "miami (fl)",
}


def normalize_spelling_key(name: str) -> str:
    """Lowercase, unify apostrophes, collapse spaces for spelling CSV lookup."""
    s = str(name).strip().lower()
    for old, new in (
        ("`", "'"),
        ("\u2019", "'"),
        ("\u2018", "'"),
        ("´", "'"),
    ):
        s = s.replace(old, new)
    s = re.sub(r"\s+", " ", s)
    return s


def strip_seed(team_fragment: str) -> str:
    """Remove a leading ``(seed) `` prefix from a team label."""
    t = team_fragment.strip()
    return _SEED_PREFIX.sub("", t).strip()


def extract_seed(team_fragment: str) -> Optional[int]:
    """Extract a leading `(seed)` integer if present."""
    m = _SEED_EXTRACT.match(str(team_fragment).strip())
    return int(m.group(1)) if m else None


def build_spelling_lookup(spellings_df: pd.DataFrame) -> dict[str, int]:
    """Map normalized ``TeamNameSpelling`` → ``TeamID`` (last row wins on duplicates)."""
    col = "TeamNameSpelling" if "TeamNameSpelling" in spellings_df.columns else "TeamName"
    lookup: dict[str, int] = {}
    for _, row in spellings_df.iterrows():
        k = normalize_spelling_key(row[col])
        lookup[k] = int(row["TeamID"])
    return lookup


def resolve_team_id(
    raw_name: str,
    lookup: dict[str, int],
) -> int:
    """
    Resolve a display name to ``TeamID`` using ``MTeamSpellings`` (+ small aliases).
    """
    key = normalize_spelling_key(strip_seed(raw_name))
    if key in _EXTRA_ALIASES:
        key = _EXTRA_ALIASES[key]
    if key not in lookup:
        raise KeyError(
            f"No TeamID for {raw_name!r} (normalized {key!r}). "
            "Add a row to MTeamSpellings or extend _EXTRA_ALIASES in named_matchups.py."
        )
    return lookup[key]


def parse_matchup_line(line: str) -> Optional[tuple[str, str]]:
    """
    Parse one schedule line into (left_team_fragment, right_team_fragment).

    Text after ``|`` (time / network) is ignored. Returns ``None`` if the line
    is not a ``vs`` matchup (e.g. section headers).
    """
    line = line.strip()
    if not line or " vs" not in line.lower():
        return None
    main = line.split("|", 1)[0].strip()
    parts = _VS_SPLIT.split(main, maxsplit=1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def parse_matchup_lines(text: str) -> list[tuple[str, str, str]]:
    """
    Parse multiple lines; each entry is (team_a_raw, team_b_raw, full_line).
    """
    rows: list[tuple[str, str, str]] = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        parsed = parse_matchup_line(line)
        if parsed is None:
            continue
        a, b = parsed
        rows.append((a, b, line))
    return rows


def parse_seed_block_lines_to_vs_text(text: str) -> str:
    """
    Parse the bracket-style block you pasted where each team is listed as:

        <seed>
        <team_name>

    Repeated for both teams in a matchup. Returns newline-joined `A vs. B`
    lines suitable for `predict_named_matchups`.

    Non-team lines like `Preview`, `TBD` are ignored.
    """
    cleaned: list[str] = []
    for raw in str(text).splitlines():
        s = raw.strip()
        if not s:
            continue
        low = s.lower()
        if low in {"preview", "tbd"}:
            continue
        cleaned.append(s)

    # Keep non-numeric as team names; seeds are numeric lines.
    tokens: list[str] = []
    for s in cleaned:
        if s.isdigit():
            continue
        tokens.append(s)

    if len(tokens) % 2 != 0:
        raise ValueError(
            f"Expected an even count of team names, got {len(tokens)}. "
            "Ensure the text is pairs of seed/team blocks."
        )

    vs_lines: list[str] = []
    for i in range(0, len(tokens), 2):
        vs_lines.append(f"{tokens[i]} vs. {tokens[i+1]}")
    return "\n".join(vs_lines)


def predict_named_matchups(
    lines_text: str,
    *,
    season: int = 2026,
    team_stats_path: Optional[Path] = None,
    spellings_path: Optional[Path] = None,
    teams_path: Optional[Path] = None,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Predict P(Team A wins) for each ``Team A vs. Team B`` line (Team A = first name).

    Adds ``team_a_name``, ``team_b_name`` from ``MTeams.csv``.
    """
    if hasattr(config, "ensure_project_dirs"):
        config.ensure_project_dirs()

    data_dir = config.DATA_RAW
    spellings_path = Path(spellings_path or data_dir / "MTeamSpellings.csv")
    teams_path = Path(teams_path or data_dir / "MTeams.csv")
    team_stats_path = Path(team_stats_path or config.DATA_INTERIM / "men_team_features.csv")

    spellings_df = pd.read_csv(spellings_path)
    teams_df = pd.read_csv(teams_path)
    team_stats = pd.read_csv(team_stats_path)
    lookup = build_spelling_lookup(spellings_df)
    id_to_name = teams_df.set_index("TeamID")["TeamName"].to_dict()

    parsed = parse_matchup_lines(lines_text)
    if not parsed:
        raise ValueError("No matchup lines parsed; expected lines like '(1) X vs. (16) Y | ...'.")

    ids_list: list[str] = []
    records: list[dict] = []

    for team_a_raw, team_b_raw, full_line in parsed:
        id_a = resolve_team_id(team_a_raw, lookup)
        id_b = resolve_team_id(team_b_raw, lookup)
        # First team in the listing = Team A in the model (same as submission ID order).
        cid = f"{season}_{id_a}_{id_b}"
        ids_list.append(cid)
        records.append(
            {
                "matchup_line": full_line,
                "team_a_id": id_a,
                "team_b_id": id_b,
                "team_a_name": id_to_name.get(id_a, f"TeamID {id_a}"),
                "team_b_name": id_to_name.get(id_b, f"TeamID {id_b}"),
                "ID": cid,
            }
        )

    sample_df = pd.DataFrame({"ID": ids_list})
    matchup_df = build_submission_matchups(
        sample_df,
        team_stats,
        feature_columns=None,
        id_column="ID",
        season=season,
    )

    preds = _predict_from_matchup_dataframe(matchup_df, write_csv=False)
    base = pd.DataFrame.from_records(records)
    out = base.merge(preds, on="ID", how="left")

    if output_csv is not None:
        outp = Path(output_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(outp, index=False)

    return out


def predict_named_matchups_with_seeds(
    lines_text: str,
    *,
    season: int = 2026,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Like `predict_named_matchups`, but also includes `seed_a` / `seed_b` columns
    when the input lines use `(seed) Team vs. (seed) Team` format.
    """
    parsed = parse_matchup_lines(lines_text)
    if not parsed:
        raise ValueError("No matchup lines parsed; expected lines like '(1) X vs. (16) Y'.")

    # Build a canonical vs text (so we reuse the existing resolver + predictor).
    vs_lines: list[str] = []
    seeds: list[tuple[Optional[int], Optional[int]]] = []
    for a_raw, b_raw, _ in parsed:
        seeds.append((extract_seed(a_raw), extract_seed(b_raw)))
        vs_lines.append(f"{a_raw} vs. {b_raw}")

    out = predict_named_matchups("\n".join(vs_lines), season=season, output_csv=output_csv)
    out.insert(0, "seed_a", [s[0] for s in seeds])
    out.insert(1, "seed_b", [s[1] for s in seeds])
    return out


def predict_first_round_2026_from_default_list(
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Convenience wrapper using :data:`FIRST_ROUND_2026_LINES`."""
    path = Path(output_csv or config.OUTPUTS_DIR / "first_round_2026_named_predictions.csv")
    return predict_named_matchups(FIRST_ROUND_2026_LINES, output_csv=path)


def predict_seed_block_matchups_2026(
    seed_block_text: str,
    *,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper for the pasted bracket-style block (seed + team per line).
    """
    vs_text = parse_seed_block_lines_to_vs_text(seed_block_text)
    return predict_named_matchups(
        vs_text,
        season=2026,
        output_csv=output_csv,
    )
