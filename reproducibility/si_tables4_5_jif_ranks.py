"""SI Tables S4 and S5: field-level preference ranks versus Journal Impact Factor ranks.

Within each field, the venues chosen by more than 10% of the field's respondents and carrying a
2-year JIF are ranked twice (public_data/field_jif_rankings.csv): by the field consensus score
(subset_ordinal_porc, "Rank (Pref.)") and by the JIF (subset_ordinal_jif, "Rank (JIF)"); 1 is
the best rank and tied JIFs receive the average of their ranks. Diff. = Rank (JIF) - Rank
(Pref.): a positive value means the JIF ranks the venue lower than the field's respondents do
(JIF undervalues the venue), a negative value that the JIF overvalues it.

  Table S4 (label stab:venue_rankings_multidisc)   Nature, Science and PNAS in every field.
      Rows are shaded where the venue was selected by 25% or fewer of the field's respondents.
  Table S5 (label stab:venue_rankings_flagships)   the nine field flagship venues (the most
      common top aspiration of each field) in every field where more than 10% of respondents
      selected them. The flagship's own field is listed first and left white; the other fields
      are shaded.

Ranks are printed as integers (a tie-averaged rank of 10.5 prints as 10), and Diff. is the
integer part of the difference of the unrounded ranks, so it can differ by one from the
difference of the two printed ranks when the JIF rank is a tie average.

Inputs
  public_data/venues.csv              venue names
  public_data/field_jif_rankings.csv  subset_ordinal_porc, subset_ordinal_jif, frac_selected

Outputs
  output/tables/si_table4.tex, output/tables/si_table5.tex
  output/results/jif_rank_tables.csv   one row per table row: table, venue, field, the unrounded
                                       ranks and difference, frac_selected, flagship_field,
                                       shaded, and the printed values

Run from the project root:  .venv/bin/python reproducibility/si_tables4_5_jif_ranks.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

GENERAL_INTEREST = ["Nature", "Science", "PNAS"]
FLAGSHIPS = {  # venue -> field for which it is the flagship
    "The American Economic Review": "Economics",
    "The Accounting Review": "Business",
    "American Sociological Review": "Sociology",
    "Journal of Personality and Social Psychology": "Psychology",
    "Journal of the American Chemical Society": "Chemistry",
    "Physical Review Letters": "Physics",
    "Annals of Mathematics": "Mathematics",
    "The American Historical Review": "History",
    "The Philosophical Review": "Philosophy",
}
# Order in which fields are listed within a venue block (the paper's order: the fields in which
# the general-interest venues are commonly chosen come first).
FIELD_ORDER = ["Biology", "Chemistry", "Physics", "Medicine", "Psychology", "Engineering", "Sociology",
               "Computer science", "Mathematics", "Philosophy", "History", "Business", "Economics"]
SHADE_BELOW = 0.25  # Table S4: shade rows where frac_selected <= 25%

venues = pd.read_csv(common.PUBLIC / "venues.csv")
name_to_id = dict(zip(venues["name"], venues["venue_db_id"]))
ranks = pd.read_csv(common.PUBLIC / "field_jif_rankings.csv")
ranks = ranks.dropna(subset=["subset_ordinal_porc", "subset_ordinal_jif"]).set_index(["field", "venue_db_id"])
ranks["diff"] = ranks["subset_ordinal_jif"] - ranks["subset_ordinal_porc"]


def table_rows(table, venue, fields, flagship_field=None):
    rows = []
    for field in fields:
        key = (field, name_to_id[venue])
        if key not in ranks.index:  # venue not in this field's ranking (< 10% selected or no JIF)
            continue
        r = ranks.loc[key]
        shaded = (r["frac_selected"] <= SHADE_BELOW) if flagship_field is None else (field != flagship_field)
        rows.append({"table": table, "venue": venue, "field": field,
                     "rank_pref": r["subset_ordinal_porc"], "rank_jif": r["subset_ordinal_jif"], "diff": r["diff"],
                     "frac_selected": r["frac_selected"], "flagship_field": field == flagship_field, "shaded": shaded,
                     "printed_rank_pref": int(r["subset_ordinal_porc"]), "printed_rank_jif": int(r["subset_ordinal_jif"]),
                     "printed_diff": int(r["diff"])})
    return rows


rows = []
for venue in GENERAL_INTEREST:
    rows += table_rows("S4", venue, FIELD_ORDER)
for venue, home in FLAGSHIPS.items():
    rows += table_rows("S5", venue, [home] + [f for f in FIELD_ORDER if f != home], flagship_field=home)
table = pd.DataFrame(rows)
table.to_csv(common.RESULTS / "jif_rank_tables.csv", index=False)


def latex(t, caption, label):
    L = ["% Requires \\usepackage{booktabs,colortbl,xcolor}",
         "\\begin{table}[htbp]", "\\centering", "\\small", "\\begin{tabular}{@{}lllrr@{}}", "\\toprule",
         "Venue & Field & Rank (Pref.) & Rank (JIF) & Diff. \\\\", "\\midrule"]
    for i, (venue, block) in enumerate(t.groupby("venue", sort=False)):
        if i:
            L.append("\\midrule")
        for _, r in block.iterrows():
            diff = "0" if r["printed_diff"] == 0 else f"${r['printed_diff']:+d}$"
            if r["shaded"]:
                L.append("\\rowcolor{gray!25}")
            L.append(f"{venue} & {r['field']} & {r['printed_rank_pref']} & {r['printed_rank_jif']} & {diff} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(L) + "\n"


CAPTION_S4 = ("\\textbf{Field-Level Preference Rankings vs. Journal Impact Factor Rankings for Multidisciplinary "
              "Venues.} {\\normalfont Gray-shaded rows indicate fields where the venue is not commonly chosen "
              "($\\leq25\\%$ of respondents in that field). Difference = Rank (JIF) $-$ Rank (Pref.); positive values "
              "indicate that JIF undervalues the venue relative to the field's preferences, while negative values "
              "indicate that JIF overvalues the venue. Only venues chosen by at least 10\\% of a field's respondents "
              "are included in the underlying rankings.}")
CAPTION_S5 = ("\\textbf{Field-Level Preference Rankings vs. Journal Impact Factor Rankings for Field Flagship "
              "Venues.} {\\normalfont White rows indicate fields where the venue is a flagship, while gray-shaded rows "
              "indicate fields where the venue is in at least 10\\% of respondents' consideration sets, but not a "
              "flagship. Diff. = Rank (JIF) $-$ Rank (Pref.); positive values indicate that JIF undervalues the venue "
              "relative to the field's preferences, while negative values indicate that JIF overvalues the venue. "
              "Only venues chosen by at least 10\\% of a field's respondents are included in both the table and the "
              "underlying rankings.}")

for name, tab, caption, label in [("si_table4.tex", "S4", CAPTION_S4, "stab:venue_rankings_multidisc"),
                                  ("si_table5.tex", "S5", CAPTION_S5, "stab:venue_rankings_flagships")]:
    out = common.TABLES / name
    out.write_text(latex(table[table["table"] == tab], caption, label))
    print(f"Saved: {out}")

cols = ["venue", "field", "printed_rank_pref", "printed_rank_jif", "printed_diff", "frac_selected", "shaded"]
print(table[cols].round(3).to_string(index=False))
home = table[table["flagship_field"]]
print(f"\nFlagships in their own field: Diff. mean {home['diff'].mean():.1f}, median {home['diff'].median():.1f}, "
      f"range {home['diff'].min():.1f} to {home['diff'].max():.1f}; positive for {(home['diff'] > 0).sum()} of {len(home)}")
