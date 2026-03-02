# Network Datasets — Academic Publication Preferences

This directory contains network representations of the survey data described in
**"Consensus and Fragmentation in Academic Publication Preferences"** (Van Buskirk et al.).
Each network is provided in two formats: **GraphML** (single-file, attributes embedded,
gzip-compressed as `.graphml.gz`) and **CSV** (edge list + node attribute table, two
files per network). Both NetworkX and graph-tool read `.graphml.gz` files transparently —
no manual decompression needed.

---

## Directory Structure

```
networks/
├── README.md
├── graphml/
│   ├── comparison/          ← directed venue comparison multigraphs
│   │   ├── all.graphml.gz
│   │   ├── Biology.graphml.gz
│   │   └── ...              (one per field, 13 total)
│   └── bipartite/           ← undirected respondent × venue bipartite graphs
│       ├── all.graphml.gz
│       ├── Biology.graphml.gz
│       └── ...              (one per field, 13 total)
└── csv/
    ├── comparison/
    │   ├── all_nodes.csv
    │   ├── all_edges.csv
    │   ├── Biology_nodes.csv
    │   ├── Biology_edges.csv
    │   └── ...              (one pair per field, 13 total)
    └── bipartite/
        ├── all_nodes.csv
        ├── all_edges.csv
        ├── Biology_nodes.csv
        ├── Biology_edges.csv
        └── ...              (one pair per field, 13 total)
```

Fields: Biology, Business, Chemistry, Computer_science, Economics, Engineering,
History, Mathematics, Medicine, Philosophy, Physics, Psychology, Sociology.

---

## Network Types

### 1 & 2 — Venue Comparison Multigraph (directed)

**Scope:** `all` (global, all 13 fields pooled) and one per field.

Each node is an academic venue. Each directed edge represents one pairwise
comparison made by a survey respondent. Because a respondent may prefer venue A
over venue B in more than one comparison session, multiple parallel edges between
the same pair of nodes are possible — hence *multigraph*.

**Tie handling:** when a respondent indicated no preference between two venues,
two directed edges are added (A→B and B→A), each with `weight=0.5`. A non-tie
comparison adds one directed edge (preferred→other) with `weight=1.0`.

#### Node attributes

| Attribute   | Type  | Description                                         |
|-------------|-------|-----------------------------------------------------|
| `name`      | str   | Full venue name                                     |
| `jcr_jif`   | float | Journal Citation Reports Impact Factor (2023); absent if unavailable |

#### Edge attributes

| Attribute               | Type  | Description                                                  |
|-------------------------|-------|--------------------------------------------------------------|
| `weight`                | float | 1.0 for a non-tie; 0.5 for each direction of a tie          |
| `is_tie`                | bool  | True if the respondent indicated no preference               |
| `user_db_id`            | int   | Anonymous respondent ID (links to `public_data/respondents.csv`) |
| `respondent_field`      | str   | Respondent's academic field (13 values)                      |
| `career_stage`          | str   | Assistant / Associate / Full Professor                       |
| `gender`                | str   | `gm` (men) / `gf` (women); absent if not reported           |
| `institution_bin_3`     | int   | Institution prestige tercile (1 = most prestigious); absent if unavailable |
| `academia_prestige_bin_3` | int | Academia-wide prestige tercile (1 = most prestigious); absent if unavailable |

#### CSV schema

`{scope}_nodes.csv` — one row per venue node
`{scope}_edges.csv` — one row per directed edge (non-ties: 1 row; ties: 2 rows)

---

### 3 & 4 — Respondent × Venue Bipartite Graph (undirected)

**Scope:** `all` (global) and one per field (filtered by respondent's own field).

Nodes are either respondents or venues. An edge between a respondent and a venue
means the venue was in that respondent's *consideration set* — i.e., it was an
aspiration venue, shown and selected during the survey, or self-nominated.

#### Node attributes — respondents (`node_type = "respondent"`)

| Attribute                | Type | Description                                                   |
|--------------------------|------|---------------------------------------------------------------|
| `field`                  | str  | Academic field (13 values)                                    |
| `career_stage`           | str  | Assistant / Associate / Full Professor                        |
| `gender`                 | str  | `gm` / `gf`; absent if not reported                          |
| `institution_bin_3`      | int  | Institution prestige tercile; absent if unavailable           |
| `institution_bin_10`     | int  | Institution prestige decile; absent if unavailable            |
| `academia_prestige_bin_3`  | int | Academia-wide prestige tercile; absent if unavailable         |
| `academia_prestige_bin_10` | int | Academia-wide prestige decile; absent if unavailable          |

In GraphML, respondent nodes additionally carry `bipartite=0` and `node_type="respondent"`.

#### Node attributes — venues (`node_type = "venue"`)

| Attribute  | Type  | Description                                           |
|------------|-------|-------------------------------------------------------|
| `name`     | str   | Full venue name                                       |
| `jcr_jif`  | float | Journal Impact Factor (2023); absent if unavailable   |

In GraphML, venue nodes additionally carry `bipartite=1` and `node_type="venue"`.

#### CSV schema

`{scope}_nodes.csv` — one row per node; `node_id` is the respondent or venue integer ID
`{scope}_edges.csv` — columns `respondent_id`, `venue_db_id`

---

## Notes

**Missing attributes.** Prestige bins, gender, and JIF are absent for some nodes/edges
because they were not available in the source data. In GraphML, absent attributes are
simply omitted (the attribute key will not be present on that element). In CSV, absent
values appear as empty cells (pandas NaN).

**Node ID namespaces.** Respondent IDs (range ~147,000–221,000) and venue IDs
(range ~1,647,000–1,713,000) do not overlap, so the integer ID alone uniquely identifies
a node across both types in the bipartite graphs.

**Field assignment.** Per-field networks are filtered by the respondent's own field
(from `public_data/respondents.csv`). This is consistent with the comparison data,
where each respondent's comparisons are always within their own field.

**Generating script.** All network files are produced by
`private_reproducibility/generate_networks.py`. Re-running the script will overwrite
all files in this directory.

---

## Python Examples

### Loading a comparison multigraph

#### NetworkX (from GraphML)

```python
import networkx as nx

# Global network
G = nx.read_graphml("networks/graphml/comparison/all.graphml.gz", node_type=int)
# G is a MultiDiGraph: 5686 nodes, 166136 edges

# Per-field
G_bio = nx.read_graphml("networks/graphml/comparison/Biology.graphml.gz", node_type=int)

# Access node attributes
print(G.nodes[1647094])        # {'name': 'Science', 'jcr_jif': 44.7}

# Access edge attributes (multigraph: indexed by key)
edges_from_science = list(G.edges(1647094, data=True, keys=True))

# Aggregate edge weights into a simple weighted digraph
import pandas as pd
edges = pd.read_csv("networks/csv/comparison/all_edges.csv")
agg = edges.groupby(["source", "target"])["weight"].sum().reset_index()
```

#### graph-tool (from GraphML)

```python
import graph_tool.all as gt

g = gt.load_graph("networks/graphml/comparison/all.graphml.gz")

# Vertex properties
name    = g.vp["name"]        # StringPropertyMap
jcr_jif = g.vp["jcr_jif"]    # DoublePropertyMap (absent for some vertices)

# Edge properties
weight       = g.ep["weight"]
is_tie       = g.ep["is_tie"]
user_db_id   = g.ep["user_db_id"]
resp_field   = g.ep["respondent_field"]

# Example: filter to non-tie edges
non_tie_view = gt.GraphView(g, efilt=lambda e: not is_tie[e])

# Per-field
g_econ = gt.load_graph("networks/graphml/comparison/Economics.graphml.gz")
```

#### pandas (from CSV)

```python
import pandas as pd

nodes = pd.read_csv("networks/csv/comparison/all_nodes.csv")
edges = pd.read_csv("networks/csv/comparison/all_edges.csv")

# Filter to a single respondent's comparisons
user_edges = edges[edges["user_db_id"] == 147627]

# Aggregate to simple weighted digraph (sum of weights)
agg = (edges.groupby(["source", "target"])["weight"]
            .sum()
            .reset_index()
            .rename(columns={"weight": "total_weight"}))
```

---

### Loading a bipartite graph

#### NetworkX (from GraphML)

```python
import networkx as nx
from networkx.algorithms import bipartite

G = nx.read_graphml("networks/graphml/bipartite/all.graphml.gz", node_type=int)

# Separate node sets
respondents = {n for n, d in G.nodes(data=True) if d.get("node_type") == "respondent"}
venues      = {n for n, d in G.nodes(data=True) if d.get("node_type") == "venue"}

# Degree (number of venues considered per respondent)
resp_degree = {n: G.degree(n) for n in respondents}

# Venue co-consideration projection (venues connected if shared by ≥1 respondent)
venue_proj = bipartite.weighted_projected_graph(G, venues)

# Per-field
G_cs = nx.read_graphml("networks/graphml/bipartite/Computer_science.graphml.gz", node_type=int)
```

#### graph-tool (from GraphML)

```python
import graph_tool.all as gt

g = gt.load_graph("networks/graphml/bipartite/all.graphml.gz")

node_type = g.vp["node_type"]   # "respondent" or "venue"
bipartite  = g.vp["bipartite"]  # 0 = respondent, 1 = venue
field      = g.vp["field"]
name       = g.vp["name"]
jcr_jif    = g.vp["jcr_jif"]

# Filter to respondent nodes only
resp_view = gt.GraphView(g, vfilt=lambda v: bipartite[v] == 0)
```

#### pandas (from CSV)

```python
import pandas as pd

nodes = pd.read_csv("networks/csv/bipartite/all_nodes.csv")
edges = pd.read_csv("networks/csv/bipartite/all_edges.csv")

respondents = nodes[nodes["node_type"] == "respondent"].copy()
venues      = nodes[nodes["node_type"] == "venue"].copy()

# Venues considered per respondent
venue_count = edges.groupby("respondent_id")["venue_db_id"].count()

# Respondents per venue (popularity in consideration sets)
resp_count = edges.groupby("venue_db_id")["respondent_id"].count()

# Join respondent demographics onto edges
edges_with_demo = edges.merge(
    respondents[["node_id", "field", "career_stage", "gender",
                 "institution_bin_3", "academia_prestige_bin_3"]],
    left_on="respondent_id", right_on="node_id",
)
```
