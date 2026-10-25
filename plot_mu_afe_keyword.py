#!/usr/bin/env python3
"""Plot MU, AFE, keyword leakage, garbage score, and refusal from mu_afe_results_qa*.json.

Main grid: two diseases per row (Cancer | Infectious, then Cardio | Patient);
QA types are columns within each disease; one line per unlearning method (x = forget percent).

Default QA panels are factual and reasoning only (no pooled ``all``); pass
``--qa-types factual,reasoning,all`` to include the combined QA bucket.

Also ``mu_vs_afe_scatter_factual_reasoning.png``: train retain MU vs train forget AFE (2×3:
Factual / Reasoning rows × disease columns).
Also ``mu_vs_afe_scatter_factual_reasoning_holdout.png`` when ``mu_afe_holdout_gr.json`` is present
(or ``--holdout-input``): holdout retain-group MU vs holdout forget-group AFE from
``compute_mu_afe_holdout.py`` (same 2×3 grid). Rotated Factual / Reasoning labels on the left;
disease names above columns.

Also ``mu_afe_barchart_patient_factual_reasoning.png``: 4 rows (Factual/Reasoning × MU/AFE),
5 cols (forget %). Each panel shows train + holdout bars per method; dashed baselines are
separate for train (matched finetuned on train QA ids) vs holdout (``baseline/metrics_by_id_holdout.json``,
needs ``--metrics-root``). Requires ``mu_afe_holdout_gr.json`` for holdout bars (``--holdout-input``).
"""

from __future__ import annotations
  
import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from statistics import harmonic_mean, mean
from typing import Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams["hatch.linewidth"] = 0.5

TASK_RE = re.compile(r"^(?P<disease>[^_]+)_(?P<percent>\d+)_(?P<method>.+)$")

DISEASE_LABELS = {
    "patient": "Random Patient",
    "cancer": "Cancer",
    "cardio": "Cardiovascular",
    "infectious": "Infectious or Immune",
}

# Multi-panel grid: two diseases per row (left → right); two rows (top → bottom).
PLOT_DISEASE_GRID = [
    ["cancer", "infectious"],
    ["cardio", "patient"],
]
# Flat order for line plots / legends that iterate one disease at a time.
PLOT_DISEASE_ORDER = [d for row in PLOT_DISEASE_GRID for d in row]

# Figure size in inches (per subplot column / row); keep panels usable when scaled to column width.
FIG_WIDTH_PER_QA_COL = 13.5
FIG_HEIGHT_PER_DISEASE_ROW = 8.0

ALL_QA_TYPES = ("factual", "reasoning", "all")

plt.rcParams.update(
    {
        "font.size": 22,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
        "legend.fontsize": 21,
        "lines.linewidth": 3.0,
        "lines.markersize": 9.5,
    }
)

def _legend_below(
    fig,
    handles,
    labels,
    *,
    layout_bottom: float | None = None,
    gap_below_axes: float = 0.07,
) -> None:
    """Single horizontal legend row under all axes (frees width for larger plots).

    ``layout_bottom`` must match the bottom of ``tight_layout(rect=[..., bottom, ...])``
    so the legend sits just under the subplot area instead of at the figure edge.
    """
    if not handles:
        return
    if layout_bottom is not None:
        anchor_y = max(0.02, layout_bottom - gap_below_axes)
    else:
        anchor_y = 0.025
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=len(handles),
        frameon=False,
    )


def _clean_method_label(method: str) -> str:
    m = str(method)
    m = m.replace("_rmuL7v2", "")
    m = m.replace("rmuL7v2", "")
    return m


def _collect_unique_legend_items(axes) -> tuple[list, list]:
    """Collect unique (handle,label) pairs across a list/array of axes."""
    seen = set()
    handles_out = []
    labels_out = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if not ll or ll in seen:
                continue
            seen.add(ll)
            handles_out.append(hh)
            labels_out.append(ll)
    return handles_out, labels_out


def _parse_rows(
    data: Dict[str, object], qa_types: List[str]
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for task, task_metrics in data.items():
        m = TASK_RE.match(task)
        if not m:
            continue
        disease = m.group("disease")
        percent_raw = m.group("percent")
        percent = int(percent_raw)
        method = m.group("method")

        if disease not in DISEASE_LABELS:
            continue
        if not isinstance(task_metrics, dict):
            continue

        for qa_type in qa_types:
            sec = task_metrics.get(qa_type, {})
            if not isinstance(sec, dict):
                continue

            retain = sec.get("retain", {}) or {}
            forget = sec.get("forget", {}) or {}
            bsec = sec.get("baseline_matched") if isinstance(sec.get("baseline_matched"), dict) else None
            bret = (bsec or {}).get("retain", {}) if bsec else {}
            bfor = (bsec or {}).get("forget", {}) if bsec else {}

            rows.append(
                {
                    "task": task,
                    "disease": disease,
                    "disease_label": DISEASE_LABELS[disease],
                    "percent": percent,
                    "percent_raw": percent_raw,
                    "method": method,
                    "qa_type": qa_type,
                    "mu": sec.get("mu"),
                    "afe": sec.get("afe"),
                    "mu_afe_avg": (
                        (
                            2.0 / (1.0/float(sec.get("mu")) + 1.0/float(sec.get("afe")))
                            if float(sec.get("mu")) != 0 and float(sec.get("afe")) != 0
                            else 0
                        )
                        if sec.get("mu") is not None and sec.get("afe") is not None
                        else None
                    ),
                    "baseline_mu": (bsec or {}).get("mu"),
                    "baseline_afe": (bsec or {}).get("afe"),
                    "baseline_mu_afe_avg": (
                        (
                            2.0 / (1.0/(bsec or {}).get("mu") + 1.0/(bsec or {}).get("afe"))
                            if (bsec or {}).get("mu") != 0 and (bsec or {}).get("afe") != 0
                            else 0
                        )
                        if (bsec and bsec.get("mu") is not None and bsec.get("afe") is not None)
                        else None
                    ),
                    "keyword_leakage_em_retain": retain.get("keyword_leakage_em"),
                    "keyword_leakage_em_forget": forget.get("keyword_leakage_em"),
                    "garbage_score_retain": retain.get("garbage_score"),
                    "garbage_score_forget": forget.get("garbage_score"),
                    "refusal_retain": retain.get("refusal"),
                    "refusal_forget": forget.get("refusal"),
                    "rougeL_recall_retain": retain.get("rougeL"),
                    "rougeL_recall_forget": forget.get("rougeL"),
                    "baseline_keyword_leakage_em_retain": bret.get("keyword_leakage_em"),
                    "baseline_keyword_leakage_em_forget": bfor.get("keyword_leakage_em"),
                    "baseline_rougeL_recall_retain": bret.get("rougeL"),
                    "baseline_rougeL_recall_forget": bfor.get("rougeL"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No matching rows found in input JSON.")
    return df


def _plot_metric(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title_prefix: str,
    method: str,
    qa_types: List[str],
):
    fig, axes = plt.subplots(
        1,
        len(qa_types),
        figsize=(FIG_WIDTH_PER_QA_COL * len(qa_types), FIG_HEIGHT_PER_DISEASE_ROW),
        sharey=False,
    )
    if len(qa_types) == 1:
        axes = [axes]
    disease_order = list(PLOT_DISEASE_ORDER)

    for ax, qa_type in zip(axes, qa_types):
        part = (
            df[(df["qa_type"] == qa_type) & (df["method"] == method)]
            .groupby(["disease", "disease_label", "percent"], as_index=False)[metric]
            .mean()
            .sort_values(["disease", "percent"])
        )
        for disease in disease_order:
            ddf = part[part["disease"] == disease]
            if ddf.empty:
                continue
            ax.plot(
                ddf["percent"],
                ddf[metric],
                marker="o",
                linewidth=2.6,
                label=ddf["disease_label"].iloc[0],
            )
        ax.set_title(f"{title_prefix} ({qa_type}, {_clean_method_label(method)})")
        ax.set_xlabel("Forget percentage")
        ax.set_ylabel(_ylabel_from_metric(metric, None))
        ax.grid(alpha=0.25)
        _set_forget_pct_xticks(ax)
    handles, labels = _collect_unique_legend_items(axes)
    bottom_pad = 0.14 if handles else 0.0
    fig.tight_layout(rect=[0.0, bottom_pad, 1.0, 1.0])
    _legend_below(fig, handles, labels, layout_bottom=bottom_pad if handles else None)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _qa_column_header_label(qa_type: str) -> str:
    if qa_type == "all":
        return "All (factual + reasoning)"
    return qa_type.capitalize()


def _ylabel_from_metric(metric: str, ylabel: str | None) -> str:
    """Left-axis label; explicit ``ylabel`` wins; default map for MU/AFE."""
    if ylabel is not None:
        return ylabel
    if metric == "mu":
        return "MU"
    if metric == "afe":
        return "AFE"
    return metric


_FORGET_PCT_X_NUM = [5, 10, 15, 20, 25]
_FORGET_PCT_X_LABELS = ["5%", "10%", "15%", "20%", "25%"]


def _set_forget_pct_xticks(ax) -> None:
    """Same x ticks on every panel: 5…25 with ``%`` labels."""
    ax.set_xticks(_FORGET_PCT_X_NUM)
    ax.set_xticklabels(_FORGET_PCT_X_LABELS)


def _plot_metric_methods_grid(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title_prefix: str,
    *,
    methods: List[str],
    qa_types: List[str],
    ylabel: str | None = None,
    figure_title: str | None = None,
    qa_column_headers: bool = True,
):
    """Two diseases side-by-side per row; QA types are columns within each disease block.

    Row 1: Cancer | Infectious; row 2: Cardio | Patient.
    When ``qa_column_headers`` is True, QA type titles repeat above each disease block;
    each disease name is centered above its QA pair; y-axis shows only the metric name.
    """
    disease_rows = list(PLOT_DISEASE_GRID)
    n_rows = len(disease_rows)
    n_qa = len(qa_types)
    n_cols = n_qa * 2  # two diseases per row
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            FIG_WIDTH_PER_QA_COL * n_cols,
            FIG_HEIGHT_PER_DISEASE_ROW * n_rows,
        ),
        sharex=True,
        sharey=False,
        squeeze=False,
        gridspec_kw={"hspace": 0.22, "wspace": 0.11},
    )

    ylab_base = _ylabel_from_metric(metric, ylabel)

    for r, disease_pair in enumerate(disease_rows):
        for d_idx, disease in enumerate(disease_pair):
            disease_label = DISEASE_LABELS.get(disease, disease)
            base_col = d_idx * n_qa
            for c, qa_type in enumerate(qa_types):
                ax = axes[r][base_col + c]
                sub = df[(df["qa_type"] == qa_type) & (df["disease"] == disease)]
                if not sub.empty:
                    part = (
                        sub.groupby(["method", "percent"], as_index=False)[metric]
                        .mean()
                        .sort_values(["method", "percent"])
                    )
                    for method in methods:
                        mdf = part[part["method"] == method]
                        if mdf.empty:
                            continue
                        ax.plot(
                            mdf["percent"],
                            mdf[metric],
                            marker="o",
                            linewidth=2.6,
                            label=_clean_method_label(method),
                        )
                if qa_column_headers:
                    ax.set_title("")
                elif figure_title is not None:
                    ax.set_title(f"{disease_label} ({qa_type})")
                else:
                    ax.set_title(f"{title_prefix} — {disease_label} ({qa_type})")
                ax.grid(alpha=0.25)
                _set_forget_pct_xticks(ax)
                if c == 0:
                    ax.set_ylabel(ylab_base)
                ax.set_xlabel("Forget percentage")

    # sharex hides x tick labels on upper rows by default; force tick labels on every row.
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r][c].tick_params(axis="x", which="major", labelbottom=True)

    if qa_column_headers:
        for d_idx in range(len(disease_rows[0])):
            base_col = d_idx * n_qa
            for c, qa_type in enumerate(qa_types):
                axes[0][base_col + c].set_title(_qa_column_header_label(qa_type))

    flat_axes = [axes[r][c] for r in range(n_rows) for c in range(n_cols)]
    handles, labels = _collect_unique_legend_items(flat_axes)
    bottom_for_legend = 0.14 if handles else 0.04

    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=int(plt.rcParams["axes.titlesize"]) + 2)

    if figure_title and qa_column_headers:
        layout_rect = [0.03, bottom_for_legend, 1.0, 0.945]
    elif figure_title:
        layout_rect = [0.0, bottom_for_legend, 1.0, 0.965]
    elif qa_column_headers:
        layout_rect = [0.03, bottom_for_legend, 1.0, 0.995]
    else:
        layout_rect = [0.0, bottom_for_legend, 1.0, 1.0]
    fig.tight_layout(rect=layout_rect)
    _legend_below(
        fig,
        handles,
        labels,
        layout_bottom=bottom_for_legend if handles else None,
    )

    if qa_column_headers:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title_fs = int(plt.rcParams["axes.titlesize"])
        for r, disease_pair in enumerate(disease_rows):
            for d_idx, disease in enumerate(disease_pair):
                disease_label = DISEASE_LABELS.get(disease, disease)
                base_col = d_idx * n_qa
                ax_l = axes[r][base_col]
                ax_r = axes[r][base_col + n_qa - 1]
                bb_l = ax_l.get_tightbbox(renderer).transformed(
                    fig.transFigure.inverted()
                )
                bb_r = ax_r.get_tightbbox(renderer).transformed(
                    fig.transFigure.inverted()
                )
                x_mid = (bb_l.x0 + bb_r.x1) / 2
                y = max(bb_l.y1, bb_r.y1) + 0.008
                fig.text(
                    x_mid,
                    y,
                    disease_label,
                    ha="center",
                    va="bottom",
                    fontsize=title_fs,
                    transform=fig.transFigure,
                )

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _task_ret_mu_forget_afe(task_metrics: object) -> tuple[float | None, float | None]:
    """MU (retain / model utility) and AFE (forget) for pooled ``all`` QA bucket.

    Supports (1) ``mu_afe_results*.json`` shape: ``task[qa_type]['mu'|'afe']`` under ``all``,
    and (2) nested ``task['retain']['all']['mu']`` / ``task['forget']['all']['afe']`` if present.
    """
    if not isinstance(task_metrics, dict):
        return None, None
    ret = task_metrics.get("retain")
    forg = task_metrics.get("forget")
    if isinstance(ret, dict) and isinstance(forg, dict):
        r_all = ret.get("all") if isinstance(ret.get("all"), dict) else None
        f_all = forg.get("all") if isinstance(forg.get("all"), dict) else None
        if isinstance(r_all, dict) and isinstance(f_all, dict):
            if r_all.get("mu") is not None and f_all.get("afe") is not None:
                return float(r_all["mu"]), float(f_all["afe"])
    all_sec = task_metrics.get("all")
    if isinstance(all_sec, dict):
        mu, afe = all_sec.get("mu"), all_sec.get("afe")
        if mu is not None and afe is not None:
            return float(mu), float(afe)
    return None, None


def _ret_mu_forget_afe_for_qa(
    entry: object, qa: str
) -> tuple[float | None, float | None]:
    """Retain MU and forget AFE for QA bucket ``qa`` (``factual`` | ``reasoning`` | ``all``).

    Tries holdout-style ``entry['retain'][qa]['mu']`` / ``entry['forget'][qa]['afe']``, then
    ``entry[qa]['mu']`` / ``entry[qa]['afe']`` (``mu_afe_results``). For ``all`` only, falls
    back to :func:`_task_ret_mu_forget_afe`.
    """
    if not isinstance(entry, dict):
        return None, None
    try:
        rm = float(entry["retain"][qa]["mu"])  # type: ignore[index]
        fa = float(entry["forget"][qa]["afe"])  # type: ignore[index]
        return rm, fa
    except (KeyError, TypeError, ValueError):
        pass
    sec = entry.get(qa)
    if isinstance(sec, dict):
        mu, afe = sec.get("mu"), sec.get("afe")
        if mu is not None and afe is not None:
            return float(mu), float(afe)
    if qa == "all":
        return _task_ret_mu_forget_afe(entry)
    return None, None


def _retain_mu_forget_afe_from_task(entry: object) -> tuple[float | None, float | None]:
    """Pooled ``all`` QA (delegates to :func:`_ret_mu_forget_afe_for_qa`)."""
    return _ret_mu_forget_afe_for_qa(entry, "all")


def _holdout_ret_mu_forget_afe_for_qa(
    entry: object, qa: str
) -> tuple[float | None, float | None]:
    """Holdout metrics JSON (``compute_mu_afe_holdout.py``): retain-group MU, forget-group AFE.

    Structure per task: ``entry['retain'][qa]['mu']``, ``entry['forget'][qa]['afe']``.
    Retain MU can be ``0.0`` while forget AFE is high; bar charts plot MU and AFE separately,
    so both values are kept whenever present (degenerate-pair filtering applies only to MU vs AFE scatters).
    """
    if not isinstance(entry, dict):
        return None, None
    ret = entry.get("retain")
    forg = entry.get("forget")
    if not isinstance(ret, dict) or not isinstance(forg, dict):
        return None, None
    rq = ret.get(qa)
    fq = forg.get(qa)
    if not isinstance(rq, dict) or not isinstance(fq, dict):
        return None, None
    rm, fa = rq.get("mu"), fq.get("afe")
    if rm is None or fa is None:
        return None, None
    try:
        return float(rm), float(fa)
    except (TypeError, ValueError):
        return None, None


# Folder suffixes per logical method.
_METHOD_SUFFIX_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "graddiff": ("graddiff", "graddiff_rmuL7v2"),
    "klmin": ("klmin", "klmin_rmuL7v2"),
    "rmu": ("rmu",),
    "simnpo": ("simnpo", "simnpo_rmuL7v2"),
}
# Preferred suffix when writing keys (first candidate); kept for callers that
# still expect a single-string map.
_METHOD_MAP = {m: cands[0] for m, cands in _METHOD_SUFFIX_CANDIDATES.items()}


def _resolve_task_key(
    data_dict: Dict[str, object],
    disease: str,
    pct: str,
    method: str,
) -> str | None:
    for suffix in _METHOD_SUFFIX_CANDIDATES.get(method, (method,)):
        key = f"{disease}_{pct}_{suffix}"
        if key in data_dict:
            return key
    return None


def _resolve_task_dir(metrics_root: Path, disease: str, pct: str) -> Path | None:
    for cands in _METHOD_SUFFIX_CANDIDATES.values():
        for suffix in cands:
            td = metrics_root / f"{disease}_{pct}_{suffix}"
            if (td / "metrics_by_id_retain.json").is_file() and (
                td / "metrics_by_id_forget.json"
            ).is_file():
                return td
    return None


def _resolve_baseline_metrics_dir(metrics_root: Path) -> Path | None:
    """``baseline/`` may live next to task folders or one level up (e.g. ``no_note_qa``)."""
    for candidate in (metrics_root, metrics_root.parent):
        bd = candidate / "baseline"
        if (bd / "metrics_by_id_rest.json").is_file() or (
            bd / "metrics_by_id_holdout.json"
        ).is_file():
            return bd
    return None


_COLORS = {
    "graddiff": "#4477AA",
    "klmin":    "#CCBB44",
    "rmu":      "#228833",
    "simnpo":   "#EE6677",
}
_MARKERS = {"graddiff": "o", "klmin": "s", "rmu": "^", "simnpo": "D"}
_LABELS = {
    "graddiff": "GradDiff",
    "klmin":    "KL-Min",
    "rmu":      "RMU",
    "simnpo":   "SimNPO",
}
_DISEASES = ["cancer", "infectious", "cardio"]
_DLABELS  = ["Cancer", "Infectious or Immune", "Cardiovascular"]
_QA_SPECS: List[Tuple[str, str]] = [("factual", "Factual"), ("reasoning", "Reasoning")]
_ZOOM_METHODS = ("graddiff", "klmin")
 
 
# Inset geometry: kept small. Offset from the lower-left corner so its tick
# labels don't overlap the main-panel ticks/axis labels.
_INSET_SIZE = 0.38
_INSET_X0   = 0.16  # axes-coords offset from left edge
_INSET_Y0   = 0.16  # axes-coords offset from bottom edge
 
 
# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
 
def _plot_retain_mu_forget_afe_scatter_fac_reasoning(
    data: Dict[str, object],
    out_path: Path,
    *,
    holdout_data: Dict[str, object] | None = None,
    metrics_root: Path | None = None,
    baseline_dir: Path | None = None,
    zoom_methods: Tuple[str, ...] | None = None,
) -> None:
    """Train + (optionally) holdout MU-vs-AFE scatter, horizontal layout.

    Rows: diseases (Cancer, Infectious or Immune, Cardiovascular).
    Cols: (Train-Factual, Train-Reasoning, Holdout-Factual, Holdout-Reasoning)
          -- holdout columns only when holdout_data is provided.

    Train panels: white-edged stars = matched finetuned baseline per forget %
    (MU from train retain QA ids, AFE from train forget QA ids).

    Holdout panels: stars = baseline finetuned on ``baseline/metrics_by_id_holdout.json``,
    same retain/forget *patient* grouping as ``compute_mu_afe_holdout.py`` (needs
    ``metrics_root`` and per-task ``metrics_by_id_retain.json`` / ``forget.json``).
    If ``metrics_root`` is None, holdout baseline stars are skipped.

    Baseline stars use the same **area** and **alpha** per forget %
    as unlearned method points (``split_sizes`` / ``split_alphas``); edge color
    follows a blue ramp by split so multiple baselines are distinguishable.

    **Note:** Holdout baseline MU/AFE use the same rounding as
    ``compute_mu_afe_holdout.aggregate`` (per-metric means to 4 decimals, then
    harmonic mean rounded to 4 decimals). Harmonic mean uses ``statistics.harmonic_mean``
    (equivalent to SciPy ``hmean`` for these clamped inputs).
    """
    _apply_acl_rc()
    zoom_methods = tuple(zoom_methods) if zoom_methods else _ZOOM_METHODS

    # ---- Resolve helpers and per-column settings ----
    train_pair_fn = _ret_mu_forget_afe_for_qa
    holdout_pair_fn = _holdout_ret_mu_forget_afe_for_qa
 
    # col_groups[ci] = (qa_short, ddict, pair_fn, filter_flag, qa_key, block_label)
    col_groups: List[Tuple[str, Dict[str, object],
                           Callable[[object, str], tuple[float | None, float | None]],
                           bool, str, str]] = []
    for qa_key, qa_short in _QA_SPECS:
        col_groups.append((qa_short, data, train_pair_fn, True, qa_key, "Train"))
    if holdout_data is not None:
        for qa_key, qa_short in _QA_SPECS:
            col_groups.append((qa_short, holdout_data, holdout_pair_fn, False, qa_key, "Holdout"))
 
    n_rows = len(_DISEASES)
    n_cols = len(col_groups)
 
    pcts = ["5", "10", "15", "20", "25"]
    split_sizes  = {"5": 18, "10": 28, "15": 40, "20": 55, "25": 72}
    split_alphas = {"5": 0.35, "10": 0.50, "15": 0.65, "20": 0.80, "25": 1.0}

    def _baseline_star_style(pct_key: str) -> tuple[float, float, tuple]:
        """(area, alpha, edge_rgba) — size/alpha match method scatters; edge color encodes split."""
        s = float(split_sizes.get(pct_key, 72))
        a = float(split_alphas.get(pct_key, 1.0))
        try:
            pi = pcts.index(pct_key)
        except ValueError:
            pi = 0
        denom = max(len(pcts) - 1, 1)
        blues_t = 0.38 + 0.55 * (pi / denom)
        edge_rgba = plt.cm.Blues(blues_t)
        return s, a, edge_rgba

    # ---- Font sizes (calibrated for a wide horizontal figure) ----
    fs_tick       = 8.0
    fs_inset_tick = 6.5
    fs_collabel   = 10.5
    fs_blockhdr   = 12.0
    fs_rowtag     = 11.0
    fs_axislabel  = 10.0
    fs_legend     = 9.5
 
    # Wide figure: width grows with #columns.
    per_col_w = 2.1
    fig_w = per_col_w * n_cols + 1.0  # +margin for row labels on the left
    per_row_h = 2.1
    fig_h = per_row_h * n_rows + 1.30  # +space for headers and legend
 
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
 
    # ---- Draw all panels ----
    for ri, disease in enumerate(_DISEASES):
        for ci, (_qa_short, ddict, pair_fn, filter_flag, qa_key, _block) in enumerate(col_groups):
            ax = axes[ri, ci]
 
            # Pass 1: collect points for this panel.
            panel_pts: List[Tuple[str, str, float, float]] = []
            zoom_xs: List[float] = []
            zoom_ys: List[float] = []
            for method in _METHOD_SUFFIX_CANDIDATES:
                for pct in pcts:
                    key = _resolve_task_key(ddict, disease, pct, method)
                    if key is None:
                        continue
                    rm, fa = pair_fn(ddict[key], qa_key)
                    if rm is None or fa is None:
                        continue
                    if filter_flag and rm == 0.0 and fa > 0.95:
                        continue
                    panel_pts.append((method, pct, rm, fa))
                    if method in zoom_methods:
                        zoom_xs.append(rm)
                        zoom_ys.append(fa)

            # Baseline stars: train from ``baseline_matched``; holdout from baseline holdout cache.
            baseline_pts: List[Tuple[str, float, float]] = []
            if filter_flag:
                for pct in pcts:
                    bm, ba = _baseline_mu_afe_for_task_column(ddict, disease, pct, qa_key)
                    if bm is not None and ba is not None:
                        baseline_pts.append((pct, bm, ba))
            elif metrics_root is not None and baseline_dir is not None:
                for pct in pcts:
                    bm, ba = _holdout_baseline_mu_afe(
                        metrics_root, baseline_dir, disease, pct, qa_key
                    )
                    if bm is not None and ba is not None:
                        baseline_pts.append((pct, bm, ba))

            # Draw main panel.
            for method in _METHOD_MAP:
                xs = [p[2] for p in panel_pts if p[0] == method]
                ys = [p[3] for p in panel_pts if p[0] == method]
                ps = [p[1] for p in panel_pts if p[0] == method]
                if not xs:
                    continue
                ax.plot(
                    xs, ys,
                    color=_COLORS[method], alpha=0.20,
                    linewidth=1.2, zorder=1,
                )
                for x, y, p in zip(xs, ys, ps):
                    ax.scatter(
                        x, y,
                        color=_COLORS[method], marker=_MARKERS[method],
                        s=split_sizes[p], alpha=split_alphas[p],
                        linewidths=0, zorder=2,
                    )

            for _bpct, bx, by in baseline_pts:
                bs, ba, bec = _baseline_star_style(_bpct)
                ax.scatter(
                    bx,
                    by,
                    marker="*",
                    s=bs,
                    facecolors="white",
                    edgecolors=bec,
                    linewidths=0.9,
                    zorder=6,
                    alpha=ba,
                )

            # ---- Per-panel axis limits sized to data + padding for inset ----
            pxs = [p[2] for p in panel_pts] + [p[1] for p in baseline_pts]
            pys = [p[3] for p in panel_pts] + [p[2] for p in baseline_pts]
            if pxs:
                dx_lo, dx_hi = min(pxs), max(pxs)
                dy_lo, dy_hi = min(pys), max(pys)
                rx = max(dx_hi - dx_lo, 0.03)
                ry = max(dy_hi - dy_lo, 0.03)
 
                upper_pad_x = 0.08 * rx
                upper_pad_y = 0.08 * ry
                lower_frac_x = _INSET_X0 + _INSET_SIZE
                lower_frac_y = _INSET_Y0 + _INSET_SIZE
                lower_ext_x = lower_frac_x / max(1e-3, 1.0 - lower_frac_x) + 0.10
                lower_ext_y = lower_frac_y / max(1e-3, 1.0 - lower_frac_y) + 0.10
                x_lo = max(-0.05, dx_lo - lower_ext_x * rx)
                x_hi = min(1.05,  dx_hi + upper_pad_x)
                y_lo = max(-0.05, dy_lo - lower_ext_y * ry)
                y_hi = min(1.05,  dy_hi + upper_pad_y)
            else:
                x_lo, x_hi = -0.02, 0.52
                y_lo, y_hi = 0.2, 1.05

            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)

            ax.grid(True, alpha=0.15, linestyle=":", linewidth=0.3)
            ax.tick_params(axis="both", labelsize=fs_tick)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for axis_attr in (ax.xaxis, ax.yaxis):
                axis_attr.set_major_locator(plt.MaxNLocator(3))
 
            # ---- Y-axis label "AFE" on the leftmost column ----
            if ci == 0:
                ax.set_ylabel("AFE", fontsize=fs_axislabel)
            # ---- X-axis label "MU" on the bottom row, every column ----
            if ri == n_rows - 1:
                ax.set_xlabel("MU", fontsize=fs_axislabel)
 
            # ---- Inset in the lower-left (shifted up and right) ----
            axins = ax.inset_axes([_INSET_X0, _INSET_Y0, _INSET_SIZE, _INSET_SIZE])
            axins.set_facecolor("white")
            axins.set_zorder(10)
            axins.patch.set_alpha(1.0)
 
            for method in zoom_methods:
                xs = [p[2] for p in panel_pts if p[0] == method]
                ys = [p[3] for p in panel_pts if p[0] == method]
                ps = [p[1] for p in panel_pts if p[0] == method]
                if not xs:
                    continue
                axins.plot(
                    xs, ys,
                    color=_COLORS[method], alpha=0.85,
                    linewidth=1.4, zorder=3,
                )
                for x, y, p in zip(xs, ys, ps):
                    axins.scatter(
                        x, y,
                        color=_COLORS[method], marker=_MARKERS[method],
                        s=split_sizes[p] * 0.55, alpha=split_alphas[p],
                        linewidths=0.2, edgecolors="white", zorder=4,
                    )

            for _bpct, bx, by in baseline_pts:
                bs, ba, bec = _baseline_star_style(_bpct)
                axins.scatter(
                    bx,
                    by,
                    marker="*",
                    s=bs * 0.55,
                    facecolors="white",
                    edgecolors=bec,
                    linewidths=0.65,
                    zorder=5,
                    alpha=ba,
                )

            if zoom_xs and zoom_ys:
                if baseline_pts:
                    zoom_xs = zoom_xs + [p[1] for p in baseline_pts]
                    zoom_ys = zoom_ys + [p[2] for p in baseline_pts]
                ix_lo, ix_hi = min(zoom_xs), max(zoom_xs)
                iy_lo, iy_hi = min(zoom_ys), max(zoom_ys)
                rx = max(ix_hi - ix_lo, 0.03)
                ry = max(iy_hi - iy_lo, 0.03)
                axins.set_xlim(ix_lo - 0.15 * rx, ix_hi + 0.15 * rx)
                axins.set_ylim(iy_lo - 0.15 * ry, iy_hi + 0.15 * ry)
 
            axins.tick_params(labelsize=fs_inset_tick, pad=0.8)
            axins.grid(True, alpha=0.25, linestyle=":", linewidth=0.25)
            for sp in axins.spines.values():
                sp.set_linewidth(0.4)
            for axis_attr in (axins.xaxis, axins.yaxis):
                axis_attr.set_major_locator(plt.MaxNLocator(2))
 
    # ---- Per-column inner header on the top row (QA: Factual / Reasoning) ----
    for ci, (qa_short, _, _, _, _, _) in enumerate(col_groups):
        axes[0, ci].set_title(qa_short, fontsize=fs_collabel, pad=2.0, fontweight="bold")
 
    # ---- Layout ----
    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        top=0.90,
        bottom=0.14,
        wspace=0.24,
        hspace=0.30,
    )
 
    fig.canvas.draw()
 
    # ---- Outer column headers spanning each Train / Holdout block ----
    block_ranges: List[Tuple[str, range]] = []
    if col_groups:
        cur_label = col_groups[0][5]
        start = 0
        for i in range(1, n_cols + 1):
            if i == n_cols or col_groups[i][5] != cur_label:
                block_ranges.append((cur_label, range(start, i)))
                if i < n_cols:
                    cur_label = col_groups[i][5]
                    start = i
 
    for block_label, col_range in block_ranges:
        left_bbox  = axes[0, col_range.start].get_position()
        right_bbox = axes[0, col_range.stop - 1].get_position()
        x_center = 0.5 * (left_bbox.x0 + right_bbox.x1)
        y_top = max(left_bbox.y1, right_bbox.y1)
        fig.text(
            x_center, y_top + 0.055, block_label,
            ha="center", va="center",
            fontsize=fs_blockhdr, fontweight="bold",
        )
 
    # ---- Row labels on the left (disease names) ----
    for ri, dl in enumerate(_DLABELS):
        bbox = axes[ri, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(
            0.018, y_center, dl,
            rotation=90, va="center", ha="center",
            fontsize=fs_rowtag, fontweight="bold",
        )
 
    # ---- Two-row legend at the bottom ----
    method_handles = [
        Line2D([0], [0],
               marker=_MARKERS[m], color=_COLORS[m], linestyle="-",
               markersize=5.0, lw=1.2, label=_LABELS[m])
        for m in _METHOD_MAP
    ]
    method_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="#111111",
            linestyle="",
            markersize=8.0,
            markerfacecolor="white",
            markeredgecolor="#111111",
            markeredgewidth=0.8,
            label="Baseline (finetuned)",
        )
    )
    pct_handles = [
        Line2D([0], [0],
               marker="o", color="#444444", linestyle="",
               markersize=max(2.5, (split_sizes[p] ** 0.5) * 0.9),
               label=f"{p}%")
        for p in pcts
    ]
 
    leg1 = fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.060),
        ncol=len(method_handles),
        frameon=False,
        fontsize=fs_legend,
        handlelength=1.4,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.add_artist(leg1)
    fig.legend(
        handles=pct_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=len(pct_handles),
        frameon=False,
        fontsize=fs_legend,
        handlelength=1.0,
        columnspacing=0.9,
        handletextpad=0.3,
    )
 
    fig.savefig(out_path, dpi=600)
    plt.close(fig)

def _apply_acl_rc() -> None:
    """Global rc tweaks for an ACL single-column figure."""
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
            "axes.labelpad": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
 
 
def _lighten(hex_color: str, amount: float = 0.55) -> tuple[float, float, float]:
    """Blend `hex_color` toward white. amount=0 -> original, amount=1 -> white."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _safe_hmean_mu_afe(values: List[float]) -> float:
    """Same clamp + harmonic mean as ``compute_mu_afe_type.compute_mu`` / ``compute_afe``."""
    clamped = [max(1e-9, min(1.0, float(v))) for v in values]
    return float(harmonic_mean(clamped))


def _mu_from_retain_metrics_block(m: object) -> float | None:
    """MU from a retain-style aggregate dict (``compute_mu_afe_type`` retain bucket)."""
    if not isinstance(m, dict):
        return None
    try:
        return _safe_hmean_mu_afe(
            [
                float(m["rougeL"]),
                float(m["em"]),
                float(m["cosine_similarity"]),
                float(m["token_entropy"]),
                float(m["entailment_proportion"]),
            ]
        )
    except (KeyError, TypeError, ValueError):
        return None


def _afe_from_forget_metrics_block(m: object) -> float | None:
    """AFE from a forget-style aggregate dict (``compute_mu_afe_type`` forget bucket)."""
    if not isinstance(m, dict):
        return None
    try:
        return _safe_hmean_mu_afe(
            [
                1.0 - float(m["rougeL"]),
                1.0 - float(m["em"]),
                1.0 - float(m["cosine_similarity"]),
                1.0 - float(m["entailment_proportion"]),
            ]
        )
    except (KeyError, TypeError, ValueError):
        return None


def _qa_type_from_id(qa_id: str) -> str | None:
    lower = str(qa_id).lower()
    if "_factual_" in lower:
        return "factual"
    if "_reasoning_" in lower:
        return "reasoning"
    return None


def _aggregate_metrics_by_qa_type(
    data: Dict[str, object],
    *,
    allowed_ids: set[str] | None = None,
) -> Dict[str, Dict[str, float] | None]:
    """Mirror ``compute_mu_afe_type.extract_metrics_by_type`` (factual/reasoning only)."""
    buckets: Dict[str, Dict[str, List[float]]] = {
        "factual": {
            "rougeL": [],
            "em": [],
            "cosine_similarity": [],
            "token_entropy": [],
            "entailment_labels": [],
        },
        "reasoning": {
            "rougeL": [],
            "em": [],
            "cosine_similarity": [],
            "token_entropy": [],
            "entailment_labels": [],
        },
    }
    for qa_id, entry in data.items():
        if allowed_ids is not None and str(qa_id) not in allowed_ids:
            continue
        if not isinstance(entry, dict):
            continue
        qt = _qa_type_from_id(str(qa_id))
        if qt not in buckets:
            continue
        m = entry.get("metrics", {})
        if not isinstance(m, dict):
            continue
        gq = m.get("generate_qa", {})
        if not isinstance(gq, dict):
            continue
        b = buckets[qt]
        b["rougeL"].append(float(gq.get("rougeL_recall", 0.0)))
        b["em"].append(float(gq.get("em", 0.0)))
        b["cosine_similarity"].append(float(gq.get("cosine_similarity", 0.0)))
        b["token_entropy"].append(float(gq.get("token_entropy", 0.0)))
        b["entailment_labels"].append(str(gq.get("entailment_label", "")).lower())

    out: Dict[str, Dict[str, float] | None] = {}
    for qt, b in buckets.items():
        if not b["rougeL"]:
            out[qt] = None
            continue
        n = len(b["rougeL"])
        ep = sum(1 for lbl in b["entailment_labels"] if lbl == "entailment") / n
        out[qt] = {
            "rougeL": round(mean(b["rougeL"]), 4),
            "em": round(mean(b["em"]), 4),
            "cosine_similarity": round(mean(b["cosine_similarity"]), 4),
            "token_entropy": round(mean(b["token_entropy"]), 4),
            "entailment_proportion": round(ep, 4),
        }
    return out


@lru_cache(maxsize=256)
def _train_baseline_mu_afe_from_metrics_cached(
    metrics_root_s: str,
    baseline_dir_s: str,
    disease: str,
    pct: str,
    qa_key: str,
    metric: str,
) -> float | None:
    """Matched finetuned baseline from ``baseline/metrics_by_id_rest.json`` + task QA ids."""
    metrics_root = Path(metrics_root_s)
    baseline_dir = Path(baseline_dir_s)
    rest_path = baseline_dir / "metrics_by_id_rest.json"
    if not rest_path.is_file():
        return None
    task_dir = _resolve_task_dir(metrics_root, disease, pct)
    if task_dir is None:
        return None
    try:
        retain_data: Dict[str, object] = json.loads(
            (task_dir / "metrics_by_id_retain.json").read_text(encoding="utf-8")
        )
        forget_data: Dict[str, object] = json.loads(
            (task_dir / "metrics_by_id_forget.json").read_text(encoding="utf-8")
        )
        baseline_rest: Dict[str, object] = json.loads(
            rest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None

    retain_ids = {str(k) for k in retain_data}
    forget_ids = {str(k) for k in forget_data}
    base_retain = _aggregate_metrics_by_qa_type(baseline_rest, allowed_ids=retain_ids)
    base_forget = _aggregate_metrics_by_qa_type(baseline_rest, allowed_ids=forget_ids)
    if metric == "mu":
        return _mu_from_retain_metrics_block(base_retain.get(qa_key))
    if metric == "afe":
        return _afe_from_forget_metrics_block(base_forget.get(qa_key))
    return None


def _baseline_matched_metric_for_qa(
    entry: object, qa: str, metric: str
) -> float | None:
    """Baseline MU from ``baseline_matched['retain']``; baseline AFE from ``['forget']``.

    Matches ``compute_mu_afe_type.analyze_folder``: baseline model on the same QA ids
    as the unlearned task's retain split (MU) and forget split (AFE), factual/reasoning.
    Falls back to pre-aggregated ``baseline_matched['mu'|'afe']`` if nested dicts are absent.
    """
    if not isinstance(entry, dict):
        return None
    sec = entry.get(qa)
    if not isinstance(sec, dict):
        return None
    bsec = sec.get("baseline_matched")
    if not isinstance(bsec, dict):
        return None
    if metric == "mu":
        v = _mu_from_retain_metrics_block(bsec.get("retain"))
    elif metric == "afe":
        v = _afe_from_forget_metrics_block(bsec.get("forget"))
    else:
        return None
    if v is not None:
        return v
    raw = bsec.get(metric)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _baseline_series_for_patient(
    data_dict: Dict[str, object],
    *,
    disease: str,
    pcts: List[str],
    method_map: Dict[str, str],
    meth_list: List[str],
    qa_key: str,
    metric: str,
    metrics_root: Path | None = None,
    baseline_dir: Path | None = None,
) -> List[float | None]:
    """One baseline value per forget-% column (same across methods at a given pct)."""
    del method_map  # suffixes resolved via _METHOD_SUFFIX_CANDIDATES
    out: List[float | None] = []
    for pct in pcts:
        val: float | None = None
        for method in meth_list:
            key = _resolve_task_key(data_dict, disease, pct, method)
            if key is None:
                continue
            val = _baseline_matched_metric_for_qa(data_dict[key], qa_key, metric)
            if val is not None:
                break
        if (
            val is None
            and metrics_root is not None
            and baseline_dir is not None
        ):
            val = _train_baseline_mu_afe_from_metrics_cached(
                str(metrics_root.resolve()),
                str(baseline_dir.resolve()),
                disease,
                pct,
                qa_key,
                metric,
            )
        out.append(val)
    return out


def _holdout_baseline_series_for_patient(
    metrics_root: Path,
    baseline_dir: Path,
    *,
    disease: str,
    pcts: List[str],
    qa_key: str,
    metric: str,
) -> List[float | None]:
    """Per forget-% column: holdout baseline MU (retain patients) or AFE (forget patients)."""
    out: List[float | None] = []
    for pct in pcts:
        bm, ba = _holdout_baseline_mu_afe(
            metrics_root, baseline_dir, disease, pct, qa_key
        )
        if metric == "mu":
            out.append(bm)
        elif metric == "afe":
            out.append(ba)
        else:
            out.append(None)
    return out


def _baseline_mu_afe_for_task_column(
    data_dict: Dict[str, object],
    disease: str,
    pct: str,
    qa_key: str,
) -> tuple[float | None, float | None]:
    """Matched finetuned baseline (MU from retain ids, AFE from forget ids) for one split."""
    for method in _METHOD_SUFFIX_CANDIDATES:
        key = _resolve_task_key(data_dict, disease, pct, method)
        if key is None:
            continue
        bm = _baseline_matched_metric_for_qa(data_dict[key], qa_key, "mu")
        ba = _baseline_matched_metric_for_qa(data_dict[key], qa_key, "afe")
        if bm is not None and ba is not None:
            return bm, ba
    return None, None


def _qa_type_holdout(qa_id: str) -> str | None:
    lower = str(qa_id).lower()
    if "_factual_" in lower:
        return "factual"
    if "_reasoning_" in lower:
        return "reasoning"
    return None


def _empty_holdout_metric_bucket() -> Dict[str, List[float]]:
    return {
        "rougeL": [],
        "em": [],
        "cosine_similarity": [],
        "token_entropy": [],
        "entailment_labels": [],
    }


def _holdout_bucket_mu_afe(bucket: Dict[str, List[float]]) -> tuple[float | None, float | None]:
    """Return (MU, AFE) for one holdout-style bucket; mirrors ``compute_mu_afe_holdout.aggregate``."""
    if not bucket["rougeL"]:
        return None, None
    ep = sum(1 for l in bucket["entailment_labels"] if l == "entailment") / len(
        bucket["entailment_labels"]
    )
    m = {
        "rougeL": round(mean(bucket["rougeL"]), 4),
        "em": round(mean(bucket["em"]), 4),
        "cosine_similarity": round(mean(bucket["cosine_similarity"]), 4),
        "token_entropy": round(mean(bucket["token_entropy"]), 4),
        "entailment_proportion": round(ep, 4),
    }
    mu = round(
        _safe_hmean_mu_afe(
            [
                m["rougeL"],
                m["em"],
                m["cosine_similarity"],
                m["token_entropy"],
                m["entailment_proportion"],
            ]
        ),
        4,
    )
    afe = round(
        _safe_hmean_mu_afe(
            [
                1.0 - m["rougeL"],
                1.0 - m["em"],
                1.0 - m["cosine_similarity"],
                1.0 - m["entailment_proportion"],
            ]
        ),
        4,
    )
    return mu, afe


@lru_cache(maxsize=512)
def _holdout_baseline_mu_afe_cached(
    metrics_root_s: str,
    baseline_dir_s: str,
    disease: str,
    pct: str,
    qa_key: str,
) -> tuple[float | None, float | None]:
    """Baseline finetuned model on holdout QAs: MU = retain-patient group, AFE = forget-patient group."""
    metrics_root = Path(metrics_root_s)
    baseline_dir = Path(baseline_dir_s)
    ho_path = baseline_dir / "metrics_by_id_holdout.json"
    if not ho_path.is_file():
        return None, None
    task_dir = _resolve_task_dir(metrics_root, disease, pct)
    if task_dir is None:
        return None, None
    rf = task_dir / "metrics_by_id_retain.json"
    ff = task_dir / "metrics_by_id_forget.json"
    try:
        retain_data: Dict[str, object] = json.loads(rf.read_text(encoding="utf-8"))
        forget_data: Dict[str, object] = json.loads(ff.read_text(encoding="utf-8"))
        baseline_ho: Dict[str, object] = json.loads(
            ho_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None, None

    forget_patients = {
        str(v["patient_id"])
        for v in forget_data.values()
        if isinstance(v, dict) and "patient_id" in v
    }
    retain_patients = {
        str(v["patient_id"])
        for v in retain_data.values()
        if isinstance(v, dict) and "patient_id" in v
    }
    buckets: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        "retain": {
            "factual": _empty_holdout_metric_bucket(),
            "reasoning": _empty_holdout_metric_bucket(),
        },
        "forget": {
            "factual": _empty_holdout_metric_bucket(),
            "reasoning": _empty_holdout_metric_bucket(),
        },
    }
    for qa_id, entry in baseline_ho.items():
        if not isinstance(entry, dict):
            continue
        patient_id = entry.get("patient_id")
        if patient_id is None:
            continue
        sp = str(patient_id)
        if sp in forget_patients:
            group = "forget"
        elif sp in retain_patients:
            group = "retain"
        else:
            continue
        qt = _qa_type_holdout(str(qa_id))
        if qt not in ("factual", "reasoning"):
            continue
        m = entry.get("metrics", {})
        if not isinstance(m, dict):
            continue
        gq = m.get("generate_qa", {})
        if not isinstance(gq, dict):
            continue
        b = buckets[group][qt]
        b["rougeL"].append(float(gq.get("rougeL_recall", 0.0)))
        b["em"].append(float(gq.get("em", 0.0)))
        b["cosine_similarity"].append(float(gq.get("cosine_similarity", 0.0)))
        b["token_entropy"].append(float(gq.get("token_entropy", 0.0)))
        b["entailment_labels"].append(str(gq.get("entailment_label", "")).lower())

    r_mu, _ = _holdout_bucket_mu_afe(buckets["retain"][qa_key])
    _, f_afe = _holdout_bucket_mu_afe(buckets["forget"][qa_key])
    if r_mu is None or f_afe is None:
        return None, None
    return r_mu, f_afe


def _holdout_baseline_mu_afe(
    metrics_root: Path,
    baseline_dir: Path,
    disease: str,
    pct: str,
    qa_key: str,
) -> tuple[float | None, float | None]:
    return _holdout_baseline_mu_afe_cached(
        str(metrics_root.resolve()),
        str(baseline_dir.resolve()),
        disease,
        pct,
        qa_key,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _plot_patient_mu_afe_barchart_fac_reasoning(
    data: Dict[str, object],
    out_path: Path,
    *,
    holdout_data: Dict[str, object] | None = None,
    metrics_root: Path | None = None,
    baseline_dir: Path | None = None,
) -> None:
    """Random Patient: 4 rows × 5 cols (forget %), train + holdout bars per method.

    Train bars/ baseline from ``data`` (``baseline_matched`` on train QA ids).
    Holdout bars from ``holdout_data``; holdout baseline from
    ``metrics_root/baseline/metrics_by_id_holdout.json`` (same logic as scatter stars).
    If ``holdout_data`` is None, only train bars are drawn.
    """
    _apply_acl_rc()

    # Make hatch strokes render crisply at 600 dpi.
    mpl.rcParams["hatch.linewidth"] = 0.5

    disease = "patient"

    method_map = dict(_METHOD_MAP)
    # Paul Tol "bright" -- colorblind-safe.
    colors = {
        "graddiff": "#4477AA",
        "klmin":    "#CCBB44",
        "rmu":      "#228833",
        "simnpo":   "#EE6677",
    }
    full_labels = {
        "graddiff": "GradDiff",
        "klmin":    "KL-Min",
        "rmu":      "RMU",
        "simnpo":   "SimNPO",
    }
    pale_colors = {m: _lighten(c, 0.60) for m, c in colors.items()}

    pcts = ["5", "10", "15", "20", "25"]
    meth_list = list(method_map.keys())

    def _series_for_qa(
        data_dict: Dict[str, object],
        qa_key: str,
        pair_fn: Callable[[object, str], tuple[float | None, float | None]],
    ) -> Tuple[Dict[str, List[float | None]], Dict[str, List[float | None]]]:
        mu_vals: Dict[str, List[float | None]] = {m: [] for m in meth_list}
        afe_vals: Dict[str, List[float | None]] = {m: [] for m in meth_list}
        for method in meth_list:
            for pct in pcts:
                key = _resolve_task_key(data_dict, disease, pct, method)
                if key is None:
                    mu_vals[method].append(None)
                    afe_vals[method].append(None)
                    continue
                rm, fa = pair_fn(data_dict[key], qa_key)
                mu_vals[method].append(float(rm) if rm is not None else None)
                afe_vals[method].append(float(fa) if fa is not None else None)
        return mu_vals, afe_vals

    # Train rows
    mu_f_tr,  afe_f_tr  = _series_for_qa(data, "factual",   _ret_mu_forget_afe_for_qa)
    mu_r_tr,  afe_r_tr  = _series_for_qa(data, "reasoning", _ret_mu_forget_afe_for_qa)

    # Matched baseline (finetuned model on same QA ids); one line per panel column.
    bl_mu_f = _baseline_series_for_patient(
        data, disease=disease, pcts=pcts, method_map=method_map,
        meth_list=meth_list, qa_key="factual", metric="mu",
        metrics_root=metrics_root, baseline_dir=baseline_dir,
    )
    bl_afe_f = _baseline_series_for_patient(
        data, disease=disease, pcts=pcts, method_map=method_map,
        meth_list=meth_list, qa_key="factual", metric="afe",
        metrics_root=metrics_root, baseline_dir=baseline_dir,
    )
    bl_mu_r = _baseline_series_for_patient(
        data, disease=disease, pcts=pcts, method_map=method_map,
        meth_list=meth_list, qa_key="reasoning", metric="mu",
        metrics_root=metrics_root, baseline_dir=baseline_dir,
    )
    bl_afe_r = _baseline_series_for_patient(
        data, disease=disease, pcts=pcts, method_map=method_map,
        meth_list=meth_list, qa_key="reasoning", metric="afe",
        metrics_root=metrics_root, baseline_dir=baseline_dir,
    )

    # Holdout baseline (finetuned on holdout patient grouping; needs metrics_root).
    bl_mu_f_ho = bl_afe_f_ho = bl_mu_r_ho = bl_afe_r_ho = [None] * len(pcts)
    if (
        holdout_data is not None
        and metrics_root is not None
        and baseline_dir is not None
    ):
        bl_mu_f_ho = _holdout_baseline_series_for_patient(
            metrics_root, baseline_dir,
            disease=disease, pcts=pcts, qa_key="factual", metric="mu",
        )
        bl_afe_f_ho = _holdout_baseline_series_for_patient(
            metrics_root, baseline_dir,
            disease=disease, pcts=pcts, qa_key="factual", metric="afe",
        )
        bl_mu_r_ho = _holdout_baseline_series_for_patient(
            metrics_root, baseline_dir,
            disease=disease, pcts=pcts, qa_key="reasoning", metric="mu",
        )
        bl_afe_r_ho = _holdout_baseline_series_for_patient(
            metrics_root, baseline_dir,
            disease=disease, pcts=pcts, qa_key="reasoning", metric="afe",
        )

    # Holdout rows (may be empty if not provided)
    if holdout_data is not None:
        mu_f_ho, afe_f_ho = _series_for_qa(
            holdout_data, "factual",   _holdout_ret_mu_forget_afe_for_qa
        )
        mu_r_ho, afe_r_ho = _series_for_qa(
            holdout_data, "reasoning", _holdout_ret_mu_forget_afe_for_qa
        )
    else:
        empty = {m: [None] * len(pcts) for m in meth_list}
        mu_f_ho, afe_f_ho, mu_r_ho, afe_r_ho = empty, empty, empty, empty

    # Rows: (label, train_series, holdout_series, train baseline, holdout baseline)
    rows: List[
        Tuple[
            str,
            Dict[str, List[float | None]],
            Dict[str, List[float | None]],
            List[float | None],
            List[float | None],
        ]
    ] = [
        ("Factual MU",  mu_f_tr,  mu_f_ho,  bl_mu_f,   bl_mu_f_ho),
        ("Factual AFE", afe_f_tr, afe_f_ho, bl_afe_f,  bl_afe_f_ho),
        ("Reason. MU",  mu_r_tr,  mu_r_ho,  bl_mu_r,   bl_mu_r_ho),
        ("Reason. AFE", afe_r_tr, afe_r_ho, bl_afe_r,  bl_afe_r_ho),
    ]

    # ---- Font sizes (calibrated for final 3.3in PDF) ----
    fs_tick     = 6.0
    fs_collabel = 9.0
    fs_rowtag   = 9.0
    fs_legend       = 8.0
    fs_legend_split = 7.0   # train / holdout / baseline row (aligned with method legend)

    # ---- Geometry ----
    n_rows = len(rows)
    n_cols = len(pcts)
    n_meth = len(meth_list)

    fig_w = 3.3
    per_row_h = 1.00
    # Extra: column headers on top + two-line legend on bottom.
    fig_h = per_row_h * n_rows + 0.95

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey="row",
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # ---- Bar geometry within a panel ----
    # Per method: a pair of bars (train, holdout). Group center sits at index mi.
    # Pair width = 2 * bar_w; small gap between method-groups.
    bar_w = 0.36     # each bar
    pair_gap = 0.02  # gap between the two bars in a pair
    x_centers = list(range(n_meth))

    def _draw_panel(
        ax: plt.Axes,
        train_h: Dict[str, List[float | None]],
        ho_h: Dict[str, List[float | None]],
        col_j: int,
        baseline_train: float | None,
        baseline_ho: float | None,
    ) -> List[float]:
        plotted: List[float] = []
        for mi, m in enumerate(meth_list):
            t = train_h[m][col_j]
            h = ho_h[m][col_j]
            # Train (left of center), Holdout (right of center)
            x_train = mi - (bar_w / 2 + pair_gap / 2)
            x_ho    = mi + (bar_w / 2 + pair_gap / 2)
            if t is not None:
                ax.bar(x_train, t, width=bar_w, color=colors[m], linewidth=0, zorder=2)
                plotted.append(t)
            if h is not None:
                ax.bar(
                    x_ho, h, width=bar_w,
                    color=pale_colors[m],
                    edgecolor=colors[m],
                    linewidth=0.4,
                    hatch="////",
                    zorder=2,
                )
                plotted.append(h)
        if baseline_train is not None:
            ax.axhline(
                baseline_train,
                color="#222222",
                linestyle=(0, (4, 2)),
                linewidth=0.7,
                zorder=4,
                alpha=0.9,
            )
            plotted.append(float(baseline_train))
        if baseline_ho is not None:
            ax.axhline(
                baseline_ho,
                color="#555555",
                linestyle=(0, (1.5, 1.5)),
                linewidth=0.7,
                zorder=4,
                alpha=0.85,
            )
            plotted.append(float(baseline_ho))
        ax.set_xticks(x_centers)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=fs_tick)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":", linewidth=0.4)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # X-limits: pad slightly so outermost bars aren't clipped.
        ax.set_xlim(-0.6, n_meth - 0.4)
        return plotted

    # Draw all panels, sharing y-limits per row.
    for ri, (_label, tr, ho, bl_tr, bl_ho) in enumerate(rows):
        row_vals: List[float] = []
        for cj in range(n_cols):
            row_vals.extend(
                _draw_panel(axes[ri, cj], tr, ho, cj, bl_tr[cj], bl_ho[cj])
            )
        if row_vals:
            # Fixed y-range across all rows: 0 to slightly above 1 so the
            # 1.0 tick label has a bit of breathing room from the top spine.
            y_bot = -0.02
            y_top = 1.05
            # Ticks at 0, 0.15, 0.30, ..., 0.90, then 1.0 as the explicit top.
            step = 0.15
            grid = [round(k * step, 4) for k in range(int(0.90 / step) + 1)]
            yticks = grid + [1.0]
            for cj in range(n_cols):
                axes[ri, cj].set_ylim(y_bot, y_top)
                axes[ri, cj].set_yticks(yticks)

    # Column headers (top row only)
    for cj, pct in enumerate(pcts):
        axes[0, cj].set_title(f"{pct}%", fontsize=fs_collabel, pad=2.5)

    for ri in range(n_rows):
        axes[ri, 0].set_ylabel("")

    fig.subplots_adjust(
        left=0.16,
        right=0.99,
        top=0.93,
        bottom=0.13,
        wspace=0.20,
        hspace=0.30,
    )

    fig.canvas.draw()

    # Row tags
    for ri, (row_label, _, _, _, _) in enumerate(rows):
        bbox = axes[ri, 0].get_position()
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        fig.text(
            0.020, y_center, row_label,
            rotation=90, va="center", ha="left",
            fontsize=fs_rowtag, fontweight="bold",
        )

    # ---- Two-part legend ----
    # Row 1: method colors (full saturation).
    method_handles = [
        Patch(facecolor=colors[m], edgecolor="none", label=full_labels[m])
        for m in meth_list
    ]
    # Row 2: train vs holdout key, using a neutral gray to show the
    # saturation + hatch convention without tying it to a specific method.
    train_demo = "#555555"
    ho_demo    = _lighten(train_demo, 0.60)
    split_handles = [
        Patch(facecolor=train_demo, edgecolor="none", label="Train"),
        Patch(
            facecolor=ho_demo,
            edgecolor=train_demo,
            linewidth=0.4,
            hatch="////",
            label="Holdout",
        ),
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle=(0, (4, 2)),
            linewidth=0.75,
            label="Train baseline",
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=(0, (1.5, 1.5)),
            linewidth=0.75,
            label="Holdout baseline",
        ),
    ]

    _leg_kw = dict(
        frameon=False,
        handlelength=1.3,
        handleheight=1.1,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    leg1 = fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.060),
        ncol=n_meth,
        fontsize=fs_legend,
        **_leg_kw,
    )
    fig.add_artist(leg1)  # keep leg1 visible after adding leg2
    fig.legend(
        handles=split_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        fontsize=fs_legend_split,
        **_leg_kw,
    )

    fig.savefig(out_path, dpi=600)
    plt.close(fig)
 
 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="/saves/eval_recomputed_new/mu_afe_results_qa_gr.json",
    )
    ap.add_argument(
        "--output-dir",
        default="/saves/eval_recomputed_new/plots_gr",
    )
    ap.add_argument(
        "--qa-types",
        default="factual,reasoning",
        help=(
            "Comma-separated QA types to plot. Default: factual,reasoning (no pooled all). "
            "Use factual,reasoning,all to add the combined bucket."
        ),
    )
    ap.add_argument(
        "--holdout-input",
        default=None,
        help=(
            "Optional ``mu_afe_holdout_gr.json`` from scripts/eval_ou/compute_mu_afe_holdout.py. "
            "If omitted, uses ``<directory of --input>/mu_afe_holdout_gr.json`` when that file exists."
        ),
    )
    ap.add_argument(
        "--metrics-root",
        type=Path,
        default=None,
        help=(
            "Root with ``baseline/metrics_by_id_holdout.json`` and per-task folders "
            "(for holdout baseline stars). Default: parent directory of --input."
        ),
    )
    ap.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help=(
            "Optional read-only directory containing metrics_by_id_rest.json and "
            "metrics_by_id_holdout.json. Defaults to baseline resolved from --metrics-root."
        ),
    )
    ap.add_argument(
        "--zoom-methods",
        default="graddiff,klmin",
        help=(
            "Comma-separated method keys to show in scatter insets. "
            "Default: graddiff,klmin. Use rmu,simnpo to zoom the baseline cluster."
        ),
    )
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    metrics_root = Path(args.metrics_root).resolve() if args.metrics_root else inp.parent.resolve()
    baseline_dir = (
        Path(args.baseline_dir).resolve()
        if args.baseline_dir
        else _resolve_baseline_metrics_dir(metrics_root)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    with inp.open("r") as f:
        data = json.load(f)

    qa_types = [x.strip() for x in args.qa_types.split(",") if x.strip()]
    zoom_methods = tuple(x.strip().lower() for x in args.zoom_methods.split(",") if x.strip())
    bad_zoom = [x for x in zoom_methods if x not in _METHOD_MAP]
    if bad_zoom:
        raise SystemExit(f"Unsupported zoom methods: {bad_zoom}. Allowed: {list(_METHOD_MAP)}")
    if not zoom_methods:
        raise SystemExit("No zoom methods selected.")
    bad = [x for x in qa_types if x not in ALL_QA_TYPES]
    if bad:
        raise SystemExit(f"Unsupported qa types: {bad}. Allowed: {list(ALL_QA_TYPES)}")
    if not qa_types:
        raise SystemExit("No QA types selected.")

    df = _parse_rows(data, qa_types=qa_types)
    df.to_csv(out_dir / "mu_afe_keyword_long.csv", index=False)

    methods = sorted(df["method"].dropna().unique().tolist())

    _plot_metric_methods_grid(
        df,
        "mu",
        out_dir / "mu_trend_methods_grid.png",
        "",
        methods=methods,
        qa_types=qa_types,
        figure_title="Model Utility (MU)",
    )
    _plot_metric_methods_grid(
        df,
        "afe",
        out_dir / "afe_trend_methods_grid.png",
        "",
        methods=methods,
        qa_types=qa_types,
        figure_title="Answer Forget Efficacy (AFE)",
    )
    
    holdout_path = (
        Path(args.holdout_input)
        if args.holdout_input
        else inp.parent / "mu_afe_holdout_gr.json"
    )
    holdout_data: Dict[str, object] | None = None
    if holdout_path.is_file():
        with holdout_path.open("r", encoding="utf-8") as hf:
            holdout_data = json.load(hf)
        _plot_retain_mu_forget_afe_scatter_fac_reasoning(
            data=data,
            out_path=out_dir / "mu_vs_afe_scatter_factual_reasoning.png",
            holdout_data=holdout_data,
            metrics_root=metrics_root,
            baseline_dir=baseline_dir,
            zoom_methods=zoom_methods,
        )
    
    _plot_patient_mu_afe_barchart_fac_reasoning(
        data,
        out_dir / "mu_afe_barchart_patient_factual_reasoning.png",
        holdout_data=holdout_data,
        metrics_root=metrics_root,
        baseline_dir=baseline_dir,
    )
    _plot_metric_methods_grid(
        df,
        "mu_afe_avg",
        out_dir / "mu_afe_avg_trend_methods_grid.png",
        "",
        methods=methods,
        qa_types=qa_types,
        figure_title="Average of MU and AFE",
        ylabel="Average",
    )
    _plot_metric_methods_grid(
        df,
        "keyword_leakage_em_forget",
        out_dir / "keyword_leakage_forget_trend_methods_grid.png",
        "Keyword Leakage EM (Forget)",
        methods=methods,
        qa_types=qa_types,
    )
    _plot_metric_methods_grid(
        df,
        "keyword_leakage_em_retain",
        out_dir / "keyword_leakage_retain_trend_methods_grid.png",
        "Keyword Leakage EM (Retain)",
        methods=methods,
        qa_types=qa_types,
    )
    _plot_metric_methods_grid(
        df,
        "rougeL_recall_retain",
        out_dir / "rougeL_recall_retain_trend_methods_grid.png",
        "ROUGE-L Recall (Retain)",
        methods=methods,
        qa_types=qa_types,
    )
    _plot_metric_methods_grid(
        df,
        "rougeL_recall_forget",
        out_dir / "rougeL_recall_forget_trend_methods_grid.png",
        "ROUGE-L Recall (Forget)",
        methods=methods,
        qa_types=qa_types,
    )
    _plot_metric_methods_grid(
        df,
        "garbage_score_forget",
        out_dir / "garbage_score_forget_trend_methods_grid.png",
        "Garbage score (Forget)",
        methods=methods,
        qa_types=qa_types,
        ylabel="Garbage score",
    )
    _plot_metric_methods_grid(
        df,
        "garbage_score_retain",
        out_dir / "garbage_score_retain_trend_methods_grid.png",
        "Garbage score (Retain)",
        methods=methods,
        qa_types=qa_types,
        ylabel="Garbage score",
    )
    _plot_metric_methods_grid(
        df,
        "refusal_forget",
        out_dir / "refusal_forget_trend_methods_grid.png",
        "Refusal rate (Forget)",
        methods=methods,
        qa_types=qa_types,
        ylabel="Refusal rate",
    )
    _plot_metric_methods_grid(
        df,
        "refusal_retain",
        out_dir / "refusal_retain_trend_methods_grid.png",
        "Refusal rate (Retain)",
        methods=methods,
        qa_types=qa_types,
        ylabel="Refusal rate",
    )

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()

