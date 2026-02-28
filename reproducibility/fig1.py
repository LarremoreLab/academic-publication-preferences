"""
Reproducibility script for Figure 1 (fig1_rev.pdf).

Inputs (all from public_data/):
  - venue_selections.csv  — one row per venue shown during initial survey pass
  - venues.csv            — venue metadata including 'name'

Output:
  - fig1_rev.pdf  (written to the same directory as this script)

Usage:
  cd <repo_root>
  .venv/bin/python reproducibility/fig1.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
PUBLIC = HERE.parent / "public_data"

# ---------------------------------------------------------------------------
# Inlined utils/field_colors.py
# ---------------------------------------------------------------------------
overlap_field_order = [
    "Economics",
    "Business",
    "Sociology",
    "Psychology",
    "Medicine",
    "Biology",
    "Chemistry",
    "Physics",
    "Engineering",
    "Mathematics",
    "Computer science",
    "History",
    "Philosophy",
]

fields_to_use = [
    "Biology", "Business", "Sociology", "Computer science",
    "Economics", "Engineering", "History", "Mathematics",
    "Medicine", "Philosophy", "Physics", "Psychology", "Chemistry",
]

n_fields = len(fields_to_use)
field_color_map = {
    field: plt.get_cmap("tab20")(i / n_fields)
    for i, field in enumerate(fields_to_use)
}

# ---------------------------------------------------------------------------
# Inlined utils/short_venue_names.py
# ---------------------------------------------------------------------------
short_venue_names = {
    "The American Economic Review": "Am. Econ. Rev.",
    "Journal of Political Economy": "J. Polit. Econ.",
    "Quarterly Journal of Economics": "Q. J. Econ.",
    "Management Science": "Manage. Sci.",
    "Academy of Management Journal": "Acad. Manage. J.",
    "Administrative Science Quarterly": "Adm. Sci. Q.",
    "American Sociological Review": "Am. Sociol. Rev.",
    "American Journal of Sociology": "Am. J. Sociol.",
    "Social Forces": "Social Forces",
    "Science": "Science",
    "Nature": "Nature",
    "PNAS": "PNAS",
    "The New England Journal of Medicine": "N. Engl. J. Med.",
    "Journal of the American Chemical Society": "J. Am. Chem. Soc.",
    "Physical Review Letters": "Phys. Rev. Lett.",
    "Transactions of the American Mathematical Society": "Trans. Am. Math. Soc.",
    "Annals of Mathematics": "Ann. Math.",
    "Inventiones Mathematicae": "Invent. Math.",
    "Communications of The ACM": "Commun. ACM",
    "ACM Conference on Human Factors in Computing Systems (CHI)": "ACM CHI",
    "The American Historical Review": "Am. Hist. Rev.",
    "Comparative Studies in Society and History": "Comp. Stud. Soc. Hist.",
    "The Journal of American History": "J. Am. Hist.",
    "Mind": "Mind",
    "Philosophical Studies": "Philos. Stud.",
    "The Philosophical Review": "Philos. Rev.",
    "USENIX Security Symposium (UsenixSec)": "UsenixSec",
    "The Accounting Review": "Account. Rev.",
    "Journal of Personality and Social Psychology": "J. Pers. Soc. Psychol.",
    "Cell": "Cell",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
shown_df = pd.read_csv(PUBLIC / "venue_selections.csv")
would_publish_df = shown_df.loc[shown_df["would_publish"] == True].copy()

venues_df = pd.read_csv(PUBLIC / "venues.csv")
venue_names = dict(zip(venues_df["venue_db_id"], venues_df["name"]))

field_counts = would_publish_df.groupby("field")["user_db_id"].nunique().to_dict()
venue_field_counts = (
    would_publish_df.groupby(["venue_db_id", "field"])["user_db_id"]
    .nunique()
    .to_dict()
)

# ---------------------------------------------------------------------------
# Precompute for heatmap (Panel C / B)
# ---------------------------------------------------------------------------
venue_counts = {field: {} for field in overlap_field_order}

for u_id, shown in shown_df.groupby("user_db_id"):
    f = shown["field"].values[0]
    for _, row in shown.iterrows():
        v = row["venue_db_id"]
        if v not in venue_counts[f]:
            venue_counts[f][v] = [0, 0, 0]
        if row["would_publish"] == True:
            venue_counts[f][v][1] += 1
        else:
            venue_counts[f][v][0] += 1
        venue_counts[f][v][2] += 1

pop_venues = []
f_tops = {f: [] for f in overlap_field_order}
mx_venues = 3
for f, v_cnts in venue_counts.items():
    v_df = (
        pd.DataFrame(v_cnts).T.sort_values(by=1, ascending=False).head(mx_venues)
    )
    for v, y in (v_df[1] / field_counts[f]).items():
        pop_venues.append(v)
        f_tops[f].append(v)
pop_venues = list(set(pop_venues))

pop_cnts = {venue_names[v]: {f: 0 for f in overlap_field_order} for v in pop_venues}
pop_accept = {venue_names[v]: {f: 0 for f in overlap_field_order} for v in pop_venues}
for v in pop_venues:
    for f in overlap_field_order:
        cnts = venue_counts[f].get(v, [0, 0, 0])
        pop_cnts[venue_names[v]][f] = cnts[1] / field_counts[f]
        pop_accept[venue_names[v]][f] = (
            0
            if (cnts[2] / field_counts[f]) < 0.1
            else cnts[1] / cnts[2]
        )


def form_df_from_pop(pop):
    pop_df = pd.DataFrame(pop).T
    pop_df = (100 * pop_df).round().astype(int)
    pop_df = pop_df[overlap_field_order]
    new_venue_order = []
    for field in pop_df.columns:
        new_venue_order.extend(
            [
                venue_names[v]
                for v in f_tops[field]
                if venue_names[v] not in new_venue_order
            ]
        )
    pop_df = pop_df.loc[new_venue_order]
    return pop_df


pop_cnts_df = form_df_from_pop(pop_cnts)
pop_accepts_df = form_df_from_pop(pop_accept)

pop_cnts_df.index = [short_venue_names[name] for name in pop_cnts_df.index]
pop_accepts_df.index = [short_venue_names[name] for name in pop_accepts_df.index]

# ---------------------------------------------------------------------------
# Precompute venue accumulation curves (Panel A)
# ---------------------------------------------------------------------------
np.random.seed(42)  # for reproducibility


def cumulative_unique_venues(venue_lists):
    unique_venues = set()
    n_unique = [0]
    for venues in venue_lists:
        unique_venues.update(set(venues))
        n_unique.append(len(unique_venues))
    return n_unique


accumulation_curves = {}
for field, field_df in would_publish_df.groupby("field"):
    accumulation_curves[field] = []
    venue_lists = field_df.groupby("user_db_id")["venue_db_id"].apply(list).values
    for _ in range(100):
        sampled_venue_lists = np.random.choice(
            venue_lists, size=len(venue_lists), replace=False
        )
        accumulation_curves[field].append(cumulative_unique_venues(sampled_venue_lists))

# ---------------------------------------------------------------------------
# Precompute overlap matrix (Panel B diagonal)
# ---------------------------------------------------------------------------


def compute_overlap_df(would_df, count_df):
    field_overlaps = {
        f: {field: [] for field in overlap_field_order} for f in overlap_field_order
    }
    for u_id, would_publish in would_df.groupby("user_db_id"):
        f = would_publish["field"].values[0]
        u_f_selects = {field: [] for field in overlap_field_order}
        for _, row in would_publish.iterrows():
            venue = row["venue_db_id"]
            for field in overlap_field_order:
                cnt = count_df.get((venue, field), 0)
                f_cnt = field_counts[field]
                if f == field:
                    cnt -= 1
                    f_cnt -= 1
                u_f_selects[field].append(cnt / f_cnt)
        for field, f_selects in u_f_selects.items():
            field_overlaps[f][field].extend(f_selects)
    return 100 * pd.DataFrame(
        {
            field: {
                f: np.mean(field_overlaps[f][field]) for f in overlap_field_order
            }
            for field in overlap_field_order
        }
    )


overlap_df = compute_overlap_df(would_publish_df, venue_field_counts)
symmetric_overlap_df = (overlap_df + overlap_df.T) / 2
diagonal_lower_values = {
    field_name: round(symmetric_overlap_df.iloc[i, i], 2)
    for i, field_name in enumerate(symmetric_overlap_df.index)
}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
abbrev = {
    "Biology": "Biology",
    "Business": "Business",
    "Chemistry": "Chemistry",
    "Computer science": "Comp. Sci.",
    "Economics": "Economics",
    "Engineering": "Engineering",
    "History": "History",
    "Mathematics": "Math",
    "Medicine": "Medicine",
    "Philosophy": "Philosophy",
    "Physics": "Physics",
    "Psychology": "Psychology",
    "Sociology": "Sociology",
}

show_borders = True
overall_figure_height = 4
overall_figure_width = 14

panel_a_width = 4
panel_b_width = 0.2
panel_c_width = 5.5

fig = plt.figure(figsize=(overall_figure_width, overall_figure_height))

main_gs = gridspec.GridSpec(
    1, 2, figure=fig,
    width_ratios=[panel_a_width, panel_b_width + panel_c_width],
    hspace=0.0, wspace=0.18,
)
bc_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, main_gs[0, 1],
    width_ratios=[panel_b_width, panel_c_width],
    hspace=0.0, wspace=0.05,
)

ax_a = fig.add_subplot(main_gs[0, 0])
ax_b = fig.add_subplot(bc_gs[0, 0])
ax_c = fig.add_subplot(bc_gs[0, 1])

# --- Panel A: Accumulation curves ---
final_values = {}
for field, curves in accumulation_curves.items():
    curves = np.array(curves)
    mean_curve = np.mean(curves, axis=0)
    ax_a.plot(mean_curve[:], c=field_color_map[field], label=field)
    final_values[field] = (len(mean_curve), mean_curve[-1])

ax_a.set_xlim(0, 610)
ax_a.set_ylim(0, 1150)

label_distance = 4

# Rotation angles hardcoded for reproducibility (computed from data slope +
# aspect-ratio correction + manual tweaks, then fixed to avoid sensitivity to
# the rendering environment).
rotation_angles = {
    "Biology":          16.43,
    "Business":         16.14,
    "Chemistry":        20.89,
    "Computer science": 44.62,
    "Economics":        24.78,
    "Engineering":      41.30,
    "History":          34.69,
    "Mathematics":      35.90,
    "Medicine":         33.94,  # 38.94 - 5 (manual tweak folded in)
    "Philosophy":       37.29,  # 44.29 - 7 (manual tweak folded in)
    "Physics":          15.14,
    "Psychology":       24.37,
    "Sociology":        34.78,
}

# Label position offsets: small nudge along the curve direction (data coords).
slope_offsets = {}
for field, curves in accumulation_curves.items():
    curves = np.array(curves)
    mean_curve = np.mean(curves, axis=0)
    last_5_x = np.arange(len(mean_curve) - 5, len(mean_curve))
    last_5_y = mean_curve[-5:]
    slope = np.polyfit(last_5_x, last_5_y, 1)[0]
    visual_length = np.sqrt(1 + slope ** 2)
    slope_offsets[field] = (
        label_distance / visual_length,
        label_distance * slope / visual_length,
    )

sorted_fields = sorted(final_values.items(), key=lambda x: x[1][1])
min_spacing = 25
adjusted_positions = {}
for i, (field, (final_x, final_y)) in enumerate(sorted_fields):
    offset_x, offset_y = slope_offsets[field]
    label_x = final_x + offset_x
    label_y = final_y + offset_y
    if i == 0:
        adjusted_positions[field] = (label_x, label_y)
    else:
        prev_field = sorted_fields[i - 1][0]
        prev_x, prev_y = adjusted_positions[prev_field]
        if abs(label_x - prev_x) < 60:
            label_y = max(label_y, prev_y + min_spacing)
        adjusted_positions[field] = (label_x, label_y)

tweaks_x = {f: 0 for f in adjusted_positions}
tweaks_x["Philosophy"] = 7

tweaks_y = {f: 0 for f in adjusted_positions}
tweaks_y["Philosophy"] = -15
tweaks_y["Chemistry"] = -30
tweaks_y["Medicine"] = 15
tweaks_y["Physics"] = -30

for field, (label_x, label_y) in adjusted_positions.items():
    ax_a.text(
        label_x + tweaks_x[field],
        label_y + tweaks_y[field],
        abbrev[field],
        color=field_color_map[field],
        fontsize=10,
        ha="left",
        va="baseline",
        fontweight="bold",
        rotation=rotation_angles[field],
    )

for field, (final_x, final_y) in final_values.items():
    ax_a.scatter(
        final_x, final_y,
        color=field_color_map[field],
        s=40, zorder=10,
        edgecolors="black", linewidth=1,
    )

ax_a.set_xlabel("Number of Participants Considered")
ax_a.set_ylabel("Cumulative Number of Unique Venues")
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.set_xticks([0, 100, 200, 300, 400, 500, 600])

# --- Panel B: Diagonal overlap column ---
field_names = pop_cnts_df.T.index.tolist()
col1_data = [diagonal_lower_values.get(field, 0) for field in field_names]

panel_c_cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "#387780"])
panel_d_cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "#0866D9"])

ax_b.imshow(
    [[val] for val in col1_data],
    cmap=panel_c_cmap, aspect="auto",
    vmin=min(col1_data), vmax=max(col1_data),
)
for i, val in enumerate(col1_data):
    ax_b.text(0, i, f"{int(round(val))}", ha="center", va="center", fontsize=10)
ax_b.set_xticks([])
ax_b.set_yticks(range(len(field_names)))
ax_b.set_yticklabels([abbrev[f] for f in field_names])
ax_b.tick_params(left=False, bottom=False)
ax_b.set_title("Mean Resp.\nOverlap (%)", fontsize=10, x=-1)
for spine in ax_b.spines.values():
    spine.set_visible(show_borders)

# --- Panel C: Heatmap ---
annot = [
    ["" if v < 10 else "{}".format(int(round(v))) for v in row]
    for row in pop_cnts_df.T.values
]
sns.heatmap(
    pop_cnts_df.T, annot=annot, fmt="", cmap=panel_d_cmap,
    cbar=False, vmin=0, vmax=100,
    annot_kws={"color": "black"}, ax=ax_c,
)
ax_c.set_yticks([])
plt.setp(ax_c.xaxis.get_majorticklabels(), rotation=50, ha="right", rotation_mode="anchor")
ax_c.set_title("Selected by Respondents (%)", fontsize=10)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
for spine in ax_c.spines.values():
    spine.set_visible(show_borders)

# --- Subplot labels ---
labs = ["A", "B", "C"]
y_label_pos = 1.02
x_label_positions = [-0.05, -4.5, 0.01]

for idx, ax in enumerate([ax_a, ax_b, ax_c]):
    ax.text(
        x_label_positions[idx], y_label_pos, labs[idx],
        transform=ax.transAxes,
        fontweight="bold", fontsize=14,
    )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = HERE / "fig1.pdf"
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
