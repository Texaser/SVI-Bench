"""
Analysis script for SVI-Bench T3 (Compositional Video Retrieval) results.

Reads the result JSONs produced by
``svi_bench.tasks.t3_compositional_video_retrieval.evaluate.run(..., save_results=True)``
and produces a structured insights report (composition-wise, hard negatives,
attribute retrievability, tier difficulty).

Invocation:
    python -m svi_bench.tasks.t3_compositional_video_retrieval.analyze
        [--results-dir DIR] [--caption full|partial] [--split val|test]

Defaults to ``$T3_OUTPUT_DIR`` (set by evaluate.run) or
``$T3_ROOT/results`` (when running against a local fixture).
"""

import argparse
import json
import os
import pathlib
from collections import defaultdict


def _default_results_dir() -> pathlib.Path:
    """Resolve the default results dir, honoring env-var overrides."""
    if v := os.environ.get("T3_OUTPUT_DIR"):
        return pathlib.Path(v)
    for env in ("T3_ROOT", "SVI_BENCH_T3_FIXTURES_DIR"):
        if v := os.environ.get(env):
            return pathlib.Path(v) / "results"
    return pathlib.Path.cwd() / "results"


DEFAULT_RESULTS_DIR = _default_results_dir()

SPORTS = ["basketball", "soccer", "hockey"]
HN_BIN_ORDER = ["0-100", "100-200", "200-300", "300-400", "400-500"]

CATEGORY_DISPLAY = {
    'entity':                              'Entity',
    'dynamics':                            'Dynamics',
    'dynamcis':                            'Dynamics',
    'context':                             'Context',
    'st':                                  'ST',
    'entity_st':                           'Entity + ST',
    'entity_context':                      'Entity + Context',
    'entity_dynamics':                     'Entity + Dynamics',
    'context_st':                          'Context + ST',
    'dynamics_st':                         'Dynamics + ST',
    'dynamics_context':                    'Dynamics + Context',
    'entity_context_st':                   'Entity + Context + ST',
    'entity_dynamics_st':                  'Entity + Dynamics + ST',
    'dynamics_context_st':                 'Dynamics + Context + ST',
    'entity_dynamics_context':             'Entity + Dynamics + Context',
    'entity_dynamics_dynamics_context_st': 'Entity + Dynamics + Context + ST',
}

CATEGORY_ORDER = [
    'entity', 'entity_dynamics', 'entity_context', 'entity_st',
    'entity_dynamics_context', 'entity_dynamics_st', 'entity_context_st',
    'entity_dynamics_dynamics_context_st',
    'dynamics', 'dynamcis', 'dynamics_context', 'dynamics_st', 'dynamics_context_st',
    'context', 'context_st', 'st',
]


def fmt(val, width, align='r'):
    s = str(val)
    return ' ' + (s.rjust(width - 2) if align == 'r' else s.ljust(width - 2)) + ' '


def format_table(headers, rows, title=None, col_align=None):
    n_cols = len(headers)
    if col_align is None:
        col_align = ['l'] + ['r'] * (n_cols - 1)
    str_headers = [str(h) for h in headers]
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(h) + 2 for h in str_headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell) + 2)

    def hline():
        return '+' + '+'.join('-' * w for w in widths) + '+'

    lines = []
    if title:
        lines.append(f"\n  {title}")
    lines.append(hline())
    lines.append('|' + '|'.join(fmt(str_headers[i], widths[i], col_align[i])
                                 for i in range(n_cols)) + '|')
    lines.append(hline())
    for row in str_rows:
        if row[0] == '__SEP__':
            lines.append(hline())
        else:
            lines.append('|' + '|'.join(fmt(row[i], widths[i], col_align[i])
                                         for i in range(n_cols)) + '|')
    lines.append(hline())
    return '\n'.join(lines)


SEPARATOR_ROW = lambda n: ['__SEP__'] + [''] * (n - 1)


def readable_cat(cat):
    return CATEGORY_DISPLAY.get(cat, cat)


def sort_categories(cats):
    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    return sorted(cats, key=lambda c: order.get(c, len(CATEGORY_ORDER)))


def load_results(results_dir: pathlib.Path, caption_type: str, split: str) -> dict:
    """Load all result JSONs for the configured caption type and split."""
    out = {}
    for sport in SPORTS:
        path = results_dir / sport / f"{caption_type}_{split}.json"
        if path.exists():
            with open(path) as f:
                out[sport] = json.load(f)
        else:
            print(f"  [WARN] Missing: {path}")
    return out


def section_overall(results, caption_type, split):
    headers = ['Metric'] + [s.capitalize() for s in SPORTS]
    rows = []
    for metric in ['n', 'R@1', 'R@5', 'R@10', 'R@50', 'R@100', 'MedR']:
        row = [metric]
        for sport in SPORTS:
            val = results[sport]['overall'].get(metric, '-')
            row.append(f"{val:.2f}" if isinstance(val, float) else str(val))
        rows.append(row)
    return format_table(headers, rows,
                        title=f"1. OVERALL COMPARISON ({caption_type}, {split})")


def section_per_tier(results):
    headers = ['Tier', 'Metric'] + [s.capitalize() for s in SPORTS]
    rows = []
    for tier in ['tier1', 'tier2', 'tier3']:
        for metric in ['n', 'R@1', 'R@100', 'MedR']:
            row = [tier if metric == 'n' else '', metric]
            for sport in SPORTS:
                val = results[sport]['per_tier'].get(tier, {}).get(metric, '-')
                row.append(f"{val:.2f}" if isinstance(val, float) else str(val))
            rows.append(row)
        if tier != 'tier3':
            rows.append(SEPARATOR_ROW(len(headers)))
    return format_table(headers, rows, title="2. PER-TIER COMPARISON")


def section_composition_analysis(results):
    all_cats = {cat for sport in SPORTS
                for tier_name in results[sport].get('per_category', {})
                for cat in results[sport]['per_category'][tier_name]}
    sorted_cats = sort_categories(all_cats)

    lines = ["\n  3. COMPOSITION-WISE ANALYSIS (per category)",
             "  " + "=" * 76]
    for tier in ['tier1', 'tier2', 'tier3']:
        headers = (['Category', 'n']
                   + [f'{s[:5].capitalize()} R@1' for s in SPORTS]
                   + [f'{s[:5].capitalize()} R@100' for s in SPORTS]
                   + [f'{s[:5].capitalize()} MedR' for s in SPORTS])
        rows = []
        for cat in sorted_cats:
            if not any(cat in results[s].get('per_category', {}).get(tier, {})
                       for s in SPORTS):
                continue
            row = [readable_cat(cat)]
            n_vals = [results[s].get('per_category', {}).get(tier, {}).get(cat, {}).get('n', 0)
                      for s in SPORTS]
            row.append('/'.join(str(n) for n in n_vals))
            for metric in ['R@1', 'R@100']:
                for sport in SPORTS:
                    m = results[sport].get('per_category', {}).get(tier, {}).get(cat, {})
                    val = m.get(metric, '-')
                    row.append(f"{val:.1f}" if isinstance(val, float) else str(val))
            for sport in SPORTS:
                m = results[sport].get('per_category', {}).get(tier, {}).get(cat, {})
                row.append(str(m.get('MedR', '-')))
            rows.append(row)
        if rows:
            lines.append(format_table(headers, rows, title=f"  {tier.upper()}"))

    lines += ["", "  KEY INSIGHTS - Composition Analysis:", "  " + "-" * 76]
    for tier in ['tier1', 'tier2', 'tier3']:
        cat_avg = []
        for cat in sorted_cats:
            vals = [results[s].get('per_category', {}).get(tier, {}).get(cat, {}).get('R@100', 0)
                    for s in SPORTS
                    if results[s].get('per_category', {}).get(tier, {}).get(cat, {}).get('n', 0) >= 10]
            if vals:
                cat_avg.append((cat, sum(vals) / len(vals), vals))
        if cat_avg:
            cat_avg.sort(key=lambda x: x[1])
            hardest, easiest = cat_avg[0], cat_avg[-1]
            lines.append(f"  {tier}: Hardest = {readable_cat(hardest[0])} "
                         f"(avg R@100={hardest[1]:.1f}%, per-sport={[f'{v:.1f}' for v in hardest[2]]})")
            lines.append(f"  {tier}: Easiest = {readable_cat(easiest[0])} "
                         f"(avg R@100={easiest[1]:.1f}%, per-sport={[f'{v:.1f}' for v in easiest[2]]})")
    return '\n'.join(lines)


def section_hn_bin_analysis(results):
    lines = ["\n  4. HARD NEGATIVE ANALYSIS (per hn_bin)",
             "  " + "=" * 76]
    if not any('per_hn_bin' in results[s] for s in SPORTS):
        lines.append("  [NO DATA] per_hn_bin not found. Re-run evaluate_retrieval.py first.")
        return '\n'.join(lines)

    headers = (['HN Bin', 'n']
               + [f'{s[:5].capitalize()} R@1' for s in SPORTS]
               + [f'{s[:5].capitalize()} R@100' for s in SPORTS]
               + [f'{s[:5].capitalize()} MedR' for s in SPORTS])
    rows = []
    for hb in HN_BIN_ORDER:
        row = [hb]
        n_vals = [results[s].get('per_hn_bin', {}).get(hb, {}).get('n', 0) for s in SPORTS]
        row.append('/'.join(str(n) for n in n_vals))
        for metric in ['R@1', 'R@100']:
            for sport in SPORTS:
                val = results[sport].get('per_hn_bin', {}).get(hb, {}).get(metric, '-')
                row.append(f"{val:.2f}" if isinstance(val, float) else str(val))
        for sport in SPORTS:
            row.append(str(results[sport].get('per_hn_bin', {}).get(hb, {}).get('MedR', '-')))
        rows.append(row)
    lines.append(format_table(headers, rows, title="  All Tiers Combined (tier2 + tier3)"))

    for tier in ['tier2', 'tier3']:
        rows = []
        for hb in HN_BIN_ORDER:
            row = [hb]
            n_vals = [results[s].get('per_hn_bin_per_tier', {}).get(tier, {}).get(hb, {}).get('n', 0)
                      for s in SPORTS]
            row.append('/'.join(str(n) for n in n_vals))
            for metric in ['R@1', 'R@100']:
                for sport in SPORTS:
                    val = results[sport].get('per_hn_bin_per_tier', {}).get(tier, {}).get(hb, {}).get(metric, '-')
                    row.append(f"{val:.2f}" if isinstance(val, float) else str(val))
            for sport in SPORTS:
                row.append(str(results[sport].get('per_hn_bin_per_tier', {}).get(tier, {}).get(hb, {}).get('MedR', '-')))
            rows.append(row)
        lines.append(format_table(headers, rows, title=f"  {tier.upper()} Only"))

    lines += ["", "  KEY INSIGHTS - Hard Negative Analysis:", "  " + "-" * 76]
    for sport in SPORTS:
        hn = results[sport].get('per_hn_bin', {})
        if hn.get('0-100') and hn.get('400-500'):
            r0, r1 = hn['0-100']['R@100'], hn['400-500']['R@100']
            m0, m1 = hn['0-100']['MedR'], hn['400-500']['MedR']
            lines.append(f"  {sport.capitalize()}: R@100 drops from {r0:.1f}% (0-100) "
                         f"to {r1:.1f}% (400-500), delta={r0 - r1:+.1f}pp; "
                         f"MedR rises from {m0} to {m1}")
    lines.append("")
    for sport in SPORTS:
        for tier in ['tier2', 'tier3']:
            d = results[sport].get('per_hn_bin_per_tier', {}).get(tier, {})
            if d.get('0-100') and d.get('400-500'):
                r0, r1 = d['0-100']['R@100'], d['400-500']['R@100']
                lines.append(f"  {sport.capitalize()} {tier}: R@100 {r0:.1f}% -> "
                             f"{r1:.1f}% (delta={r0 - r1:+.1f}pp)")
    return '\n'.join(lines)


def section_difficult_compositions(results):
    lines = ["\n  5. MOST DIFFICULT COMPOSITIONS",
             "  " + "=" * 76]

    for sport in SPORTS:
        comps = []
        per_comp = results[sport].get('per_composition', {})
        for tier_name in per_comp:
            for cat_name in per_comp[tier_name]:
                for comp_key, m in per_comp[tier_name][cat_name].items():
                    comps.append({
                        'tier': tier_name, 'category': readable_cat(cat_name),
                        'composition': comp_key,
                        'n': m.get('n', 0), 'R@1': m.get('R@1', 0),
                        'R@100': m.get('R@100', 0), 'MedR': m.get('MedR', 0),
                    })
        reliable = [c for c in comps if c['n'] >= 5]
        headers = ['Rank', 'Tier', 'Category', 'Composition', 'n', 'R@1', 'R@100', 'MedR']

        reliable.sort(key=lambda x: (x['R@100'], -x['MedR']))
        rows = [[str(i + 1), c['tier'], c['category'], c['composition'],
                 str(c['n']), f"{c['R@1']:.1f}", f"{c['R@100']:.1f}", str(c['MedR'])]
                for i, c in enumerate(reliable[:15])]
        lines.append(f"\n  --- {sport.upper()} - Top 15 Hardest (n >= 5) ---")
        lines.append(format_table(headers, rows))

        reliable.sort(key=lambda x: (-x['R@100'], x['MedR']))
        rows = [[str(i + 1), c['tier'], c['category'], c['composition'],
                 str(c['n']), f"{c['R@1']:.1f}", f"{c['R@100']:.1f}", str(c['MedR'])]
                for i, c in enumerate(reliable[:15])]
        lines.append(f"\n  --- {sport.upper()} - Top 15 Easiest (n >= 5) ---")
        lines.append(format_table(headers, rows))

    lines.append("\n  --- ATTRIBUTE DIFFICULTY RANKING (cross-sport) ---")
    lines.append("  Which individual attributes appear most in hard compositions?\n")
    attr_difficulty = defaultdict(list)
    for sport in SPORTS:
        per_comp = results[sport].get('per_composition', {})
        for tier_name in per_comp:
            for cat_name in per_comp[tier_name]:
                for comp_key, m in per_comp[tier_name][cat_name].items():
                    if m.get('n', 0) < 5:
                        continue
                    for attr in (a.strip() for a in comp_key.split(' + ')):
                        attr_difficulty[attr].append((m.get('R@100', 0), m.get('n', 0)))
    attr_avg = []
    for attr, entries in attr_difficulty.items():
        if len(entries) >= 3:
            avg_r100 = sum(e[0] for e in entries) / len(entries)
            attr_avg.append((attr, avg_r100, len(entries), sum(e[1] for e in entries)))
    attr_avg.sort(key=lambda x: x[1])
    rows = [[a, f"{r:.1f}", str(nc), str(tn)] for a, r, nc, tn in attr_avg]
    lines.append(format_table(['Attribute', 'Avg R@100', '#Compositions', 'Total n'], rows,
                              title="  Attributes Ranked by Avg R@100 (lower = harder)"))

    lines.append("\n  --- TIER1 ATTRIBUTE DIFFICULTY (per sport) ---")
    lines.append("  Tier1 compositions have exactly one attribute, so each row = one attribute.\n")
    for sport in SPORTS:
        per_comp = results[sport].get('per_composition', {})
        tier1_comps = []
        for cat_name, items in per_comp.get('tier1', {}).items():
            for comp_key, m in items.items():
                tier1_comps.append({'attribute': comp_key, 'category': readable_cat(cat_name),
                                    'n': m.get('n', 0), 'R@1': m.get('R@1', 0),
                                    'R@100': m.get('R@100', 0), 'MedR': m.get('MedR', 0)})
        if not tier1_comps:
            continue
        tier1_comps.sort(key=lambda x: (x['R@100'], -x['MedR']))
        headers_t1 = ['Rank', 'Attribute', 'Category', 'n', 'R@1', 'R@100', 'MedR']
        rows_t1 = [[str(i + 1), c['attribute'], c['category'], str(c['n']),
                    f"{c['R@1']:.1f}", f"{c['R@100']:.1f}", str(c['MedR'])]
                   for i, c in enumerate(tier1_comps)]
        lines.append(format_table(headers_t1, rows_t1,
                                  title=f"  {sport.upper()} - Tier1 Attributes (sorted by R@100)"))
    return '\n'.join(lines)


def section_attribute_retrievability(results):
    lines = ["\n  6. PER-ATTRIBUTE RETRIEVABILITY ANALYSIS",
             "  " + "=" * 76,
             "  Weighted-average R@100 and MedR per atomic attribute "
             "across compositions (n>=5).\n"]

    for sport in SPORTS:
        attr_data = defaultdict(list)
        per_comp = results[sport].get('per_composition', {})
        for tier_name in per_comp:
            for cat_name in per_comp[tier_name]:
                for comp_key, m in per_comp[tier_name][cat_name].items():
                    n = m.get('n', 0)
                    if n < 5:
                        continue
                    for attr in (a.strip() for a in comp_key.split(' + ')):
                        attr_data[attr].append((m.get('R@100', 0), m.get('MedR', 0), n))
        if not attr_data:
            continue
        attr_stats = []
        for attr, entries in attr_data.items():
            total_n = sum(e[2] for e in entries)
            if total_n < 10:
                continue
            wr100 = sum(e[0] * e[2] for e in entries) / total_n
            wmedr = sum(e[1] * e[2] for e in entries) / total_n
            attr_stats.append((attr, wr100, wmedr, total_n, len(entries)))
        attr_stats.sort(key=lambda x: x[1], reverse=True)
        rows = []
        for attr, r, medr, tn, nc in attr_stats:
            verdict = 'EASY' if r >= 40 else ('MODERATE' if r >= 25 else 'HARD')
            rows.append([attr, f"{r:.1f}", f"{medr:.0f}", str(tn), str(nc), verdict])
        lines.append(format_table(
            ['Attribute', 'Wt-Avg R@100', 'Wt-Avg MedR', 'Total n', '#Comps', 'Verdict'],
            rows, title=f"  {sport.upper()} - Attribute Retrievability"))

    cross_sport = defaultdict(dict)
    for sport in SPORTS:
        attr_data = defaultdict(list)
        per_comp = results[sport].get('per_composition', {})
        for tier_name in per_comp:
            for cat_name in per_comp[tier_name]:
                for comp_key, m in per_comp[tier_name][cat_name].items():
                    n = m.get('n', 0)
                    if n < 5:
                        continue
                    for attr in (a.strip() for a in comp_key.split(' + ')):
                        attr_data[attr].append((m.get('R@100', 0), n))
        for attr, entries in attr_data.items():
            total_n = sum(e[1] for e in entries)
            if total_n >= 10:
                cross_sport[attr][sport] = sum(e[0] * e[1] for e in entries) / total_n

    shared = [(a, v) for a, v in cross_sport.items() if len(v) >= 2]
    easy = sorted([(a, sum(v.values()) / len(v), v) for a, v in shared
                   if sum(v.values()) / len(v) >= 35], key=lambda x: -x[1])
    hard = sorted([(a, sum(v.values()) / len(v), v) for a, v in shared
                   if sum(v.values()) / len(v) < 25], key=lambda x: x[1])

    lines += ["", "  CROSS-SPORT ATTRIBUTE PATTERNS:", "  " + "-" * 76]
    if easy:
        lines.append("  RETRIEVABLE attributes (avg R@100 >= 35%):")
        for attr, avg, vals in easy:
            sd = ', '.join(f"{s[:3]}={vals[s]:.1f}%" for s in SPORTS if s in vals)
            lines.append(f"    {attr}: avg R@100={avg:.1f}% ({sd})")
    if hard:
        lines.append("  NON-RETRIEVABLE attributes (avg R@100 < 25%):")
        for attr, avg, vals in hard:
            sd = ', '.join(f"{s[:3]}={vals[s]:.1f}%" for s in SPORTS if s in vals)
            lines.append(f"    {attr}: avg R@100={avg:.1f}% ({sd})")
    return '\n'.join(lines)


def section_tier_difficulty(results):
    lines = ["\n  7. TIER DIFFICULTY ANALYSIS",
             "  " + "=" * 76]
    headers = (['Tier', '#Attrs']
               + [f'{s.capitalize()} R@1' for s in SPORTS]
               + [f'{s.capitalize()} R@100' for s in SPORTS]
               + [f'{s.capitalize()} MedR' for s in SPORTS])
    rows = []
    tier_attrs = {'tier1': '1', 'tier2': '2', 'tier3': '3+'}
    for tier in ['tier1', 'tier2', 'tier3']:
        row = [tier, tier_attrs[tier]]
        for metric in ['R@1', 'R@100']:
            for sport in SPORTS:
                val = results[sport].get('per_tier', {}).get(tier, {}).get(metric, '-')
                row.append(f"{val:.2f}" if isinstance(val, float) else str(val))
        for sport in SPORTS:
            row.append(str(results[sport].get('per_tier', {}).get(tier, {}).get('MedR', '-')))
        rows.append(row)
    lines.append(format_table(headers, rows))

    lines += ["", "  KEY INSIGHTS - Tier Difficulty:", "  " + "-" * 76]
    for sport in SPORTS:
        per_t = results[sport].get('per_tier', {})
        medrs = [(t, per_t.get(t, {}).get('MedR', 0)) for t in ['tier1', 'tier2', 'tier3']]
        medrs.sort(key=lambda x: x[1], reverse=True)
        ordering = ' > '.join(f"{t}(MedR={m})" for t, m in medrs)
        r100 = {t: per_t.get(t, {}).get('R@100', 0) for t in ['tier1', 'tier2', 'tier3']}
        lines.append(f"  {sport.capitalize()}: Difficulty: {ordering}")
        lines.append(f"    R@100: tier1={r100['tier1']:.1f}%, "
                     f"tier2={r100['tier2']:.1f}%, tier3={r100['tier3']:.1f}%")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=pathlib.Path, default=DEFAULT_RESULTS_DIR,
                        help=f"Where evaluate_retrieval.py wrote its outputs. Default: {DEFAULT_RESULTS_DIR}")
    parser.add_argument("--caption", default="full",
                        choices=["full", "partial"],
                        help="Caption type to analyze. Default: full.")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="Split to analyze. Default: test.")
    parser.add_argument("--report", type=pathlib.Path, default=None,
                        help="Path to write the report. Default: <results-dir>/analysis_report.txt")
    args = parser.parse_args()

    caption_type = f"{args.caption}_caption"
    print(f"Loading results from {args.results_dir} (caption={caption_type}, split={args.split})...")
    results = load_results(args.results_dir, caption_type, args.split)
    if not results:
        print("No results found!")
        return

    sections = [
        "=" * 80,
        "  COMPOSITIONAL VIDEO RETRIEVAL - ANALYSIS REPORT",
        f"  Caption type: {caption_type} | Split: {args.split}",
        "=" * 80,
        section_overall(results, caption_type, args.split),
        section_per_tier(results),
        section_composition_analysis(results),
        section_hn_bin_analysis(results),
        section_difficult_compositions(results),
        section_attribute_retrievability(results),
        section_tier_difficulty(results),
    ]
    full_report = '\n'.join(sections)
    print(full_report)

    out_path = args.report or (args.results_dir / "analysis_report.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(full_report)
    print(f"\n  Report saved to: {out_path}")


if __name__ == "__main__":
    main()
