"""Generate per-field venue preference network visualizations (Fig 3).

For each of 13 fields, produces an interactive D3.js force-directed graph where:
  - Nodes = venues, sized by how many field members considered them
  - Vertical position = consensus field rank (SpringRank, rescaled)
  - Edges = consecutive venue pairs in 50 randomly sampled users' rankings
  - Color = white-to-field-color gradient by consideration frequency

Inputs:
  - public_data/respondents.csv
  - public_data/comparisons.csv
  - public_data/venues.csv
  - reproducibility/derived/individual_rankings.csv
  - reproducibility/derived/field_consensus_rankings.csv

Outputs:
  - reproducibility/fig3/<Field>.html  (one per field, 13 total)

Usage:
  .venv/bin/python reproducibility/fig3.py

Run compute_rankings.py first if derived/ files are missing.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
PUBLIC = HERE.parent / "public_data"
DERIVED = HERE / "derived"
OUT_DIR = HERE / "fig3"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Check required derived files
# ---------------------------------------------------------------------------
_required = [
    DERIVED / "individual_rankings.csv",
    DERIVED / "field_consensus_rankings.csv",
]
_missing = [p for p in _required if not p.exists()]
if _missing:
    print("Missing required derived files:")
    for p in _missing:
        print(f"  {p}")
    print("\nPlease run first:\n  .venv/bin/python reproducibility/compute_rankings.py")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
respondents = pd.read_csv(PUBLIC / "respondents.csv")
comparisons = pd.read_csv(PUBLIC / "comparisons.csv")
venues = pd.read_csv(PUBLIC / "venues.csv")
indiv = pd.read_csv(DERIVED / "individual_rankings.csv")
field_consensus = pd.read_csv(DERIVED / "field_consensus_rankings.csv")

id_to_name = dict(zip(venues["venue_db_id"], venues["name"]))
field_to_users = {
    f: grp["user_db_id"].tolist()
    for f, grp in respondents.groupby("field")
}

# ---------------------------------------------------------------------------
# Fields and per-field colors (tab20, same order as original notebook)
# ---------------------------------------------------------------------------
FIELDS = [
    "Biology", "Business", "Sociology", "Computer science",
    "Economics", "Engineering", "History", "Mathematics",
    "Medicine", "Philosophy", "Physics", "Psychology", "Chemistry",
]
field_color_map = {
    field: mcolors.to_hex(plt.get_cmap("tab20")(i / len(FIELDS)))
    for i, field in enumerate(FIELDS)
}

# ---------------------------------------------------------------------------
# Consideration counts: per field, how many users included each venue
# ---------------------------------------------------------------------------
consideration_counts = {}
for field in FIELDS:
    users = set(field_to_users.get(field, []))
    field_comps = comparisons[comparisons["user_db_id"].isin(users)]
    venues_seen = []
    for _, grp in field_comps.groupby("user_db_id"):
        venues_seen.extend(
            set(grp["pref_venue_db_id"]).union(set(grp["other_venue_db_id"]))
        )
    consideration_counts[field] = Counter(venues_seen)

# ---------------------------------------------------------------------------
# Chain graph: directed edges between consecutive venues in each user's ranking
# ---------------------------------------------------------------------------
K = 50    # users sampled per field
SEED = 42


def create_field_user_chain_graph(field, n_to_plot, seed=None):
    if seed is not None:
        np.random.seed(seed)
    field_users = field_to_users.get(field, [])
    selected = np.random.choice(
        field_users, size=min(n_to_plot, len(field_users)), replace=False
    )
    edge_counts = {}
    all_venues = set()
    for user in selected:
        chain = (
            indiv[indiv["user_db_id"] == user]
            .sort_values("ordinal_rank")["venue_db_id"]
            .tolist()
        )
        all_venues.update(chain)
        for a, b in zip(chain, chain[1:]):
            edge_counts[(a, b)] = edge_counts.get((a, b), 0) + 1
    node_ids = sorted(all_venues)
    idx = {v: i for i, v in enumerate(node_ids)}
    adj = np.zeros((len(node_ids), len(node_ids)))
    for (src, tgt), cnt in edge_counts.items():
        adj[idx[src], idx[tgt]] = cnt
    return adj, node_ids


# ---------------------------------------------------------------------------
# HTML generation — D3 v7 force-directed graph, vertical rank layout.
# Adapted from networks_visualizations_v4.ipynb (inline function, cell 27886cf3).
# ---------------------------------------------------------------------------
def create_vertical_force_directed_graph(
    adjacency_matrix, ranks, node_names=None, node_sizes=None, node_colors=None,
    top_margin=50, bottom_margin=50,
    min_node_radius=5, max_node_radius=15,
    min_stroke=0.5, max_stroke=3.0, edge_opacity=0.6,
    node_stroke_width=2, custom_color="#007bff",
    output_file="graph.html", vertical_force=0.99,
    color_range_min=0.0, color_range_max=1.0,
):
    if len(ranks) > 0:
        min_rank, max_rank = min(ranks), max(ranks)
        if max_rank > min_rank:
            normalized_ranks = [(r - min_rank) / (max_rank - min_rank) for r in ranks]
        else:
            normalized_ranks = [0.5] * len(ranks)
    else:
        normalized_ranks = []

    if node_sizes is not None and len(node_sizes) > 0:
        min_s, max_s = min(node_sizes), max(node_sizes)
        if max_s > min_s:
            normalized_sizes = [(s - min_s) / (max_s - min_s) for s in node_sizes]
        else:
            normalized_sizes = [0.5] * len(node_sizes)
    else:
        normalized_sizes = [0.5] * len(normalized_ranks)

    if node_colors is not None and len(node_colors) > 0:
        min_c, max_c = min(node_colors), max(node_colors)
        if max_c > min_c:
            normalized_colors = [(c - min_c) / (max_c - min_c) for c in node_colors]
        else:
            normalized_colors = [0.5] * len(node_colors)
    else:
        normalized_colors = normalized_ranks

    if node_names is None:
        node_names = [str(i) for i in range(len(normalized_ranks))]

    min_area = min_node_radius ** 2
    max_area = max_node_radius ** 2
    nodes = []
    for i, (rank, size, color) in enumerate(
        zip(normalized_ranks, normalized_sizes, normalized_colors)
    ):
        node_area = min_area + size * (max_area - min_area)
        nodes.append({
            "id": i,
            "name": node_names[i],
            "rank": rank,
            "size": size,
            "color": color,
            "display_radius": math.sqrt(node_area),
        })

    n = len(normalized_ranks)
    links = [
        {"source": i, "target": j, "value": float(adjacency_matrix[i, j])}
        for i in range(n) for j in range(n)
        if adjacency_matrix[i, j] > 0
    ]

    graph_data = {"nodes": nodes, "links": links}
    color_config = {"min": color_range_min, "max": color_range_max}

    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>MY_TITLE</title>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; }
        svg { width: 100vw; height: 100vh; display: block; }
        .controls {
            position: absolute; top: 10px; left: 10px;
            background-color: rgba(255,255,255,0.95);
            padding: 15px; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-size: 12px; z-index: 1000;
            min-width: 250px; max-height: 90vh; overflow-y: auto;
        }
        .controls h3 { margin: 0 0 15px 0; font-size: 16px; color: #333; }
        .control-group { margin-bottom: 12px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        .control-group:last-child { border-bottom: none; }
        .control-group label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
        .control-group input[type="checkbox"] { margin-right: 8px; }
        .control-group select, .control-group input[type="range"],
        .control-group input[type="number"], .control-group input[type="color"] {
            width: 100%; padding: 4px; border: 1px solid #ccc;
            border-radius: 4px; box-sizing: border-box;
        }
        .control-group input[type="range"] { height: 20px; }
        .range-display { display: inline-block; margin-left: 8px; min-width: 40px; color: #666; font-size: 11px; }
        .links line { stroke: #999; }
        .nodes circle { stroke: #fff; }
        .node-labels { font-size: 10px; text-anchor: middle; pointer-events: none; display: none; }
        .tooltip {
            position: absolute; background-color: rgba(255,255,255,0.95);
            border: 1px solid #ddd; padding: 8px; border-radius: 4px;
            font-size: 12px; pointer-events: none; opacity: 0;
            transition: opacity 0.2s; max-width: 250px;
            white-space: normal; word-wrap: break-word;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
    </style>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
    <div class="controls">
        <h3>MY_TITLE</h3>
        <div class="control-group">
            <label><input type="checkbox" id="showLabels"> Show Node Names</label>
        </div>
        <div class="control-group">
            <label for="customColor">Node Color:</label>
            <input type="color" id="customColor" value="CUSTOM_COLOR">
        </div>
        <div class="control-group">
            <label for="minNodeRadius">Min Node Radius: <span class="range-display" id="minNodeRadiusDisplay">MIN_NODE_RADIUS</span></label>
            <input type="range" id="minNodeRadius" min="3" max="20" value="MIN_NODE_RADIUS" step="1">
        </div>
        <div class="control-group">
            <label for="maxNodeRadius">Max Node Radius: <span class="range-display" id="maxNodeRadiusDisplay">MAX_NODE_RADIUS</span></label>
            <input type="range" id="maxNodeRadius" min="10" max="50" value="MAX_NODE_RADIUS" step="1">
        </div>
        <div class="control-group">
            <label for="minStroke">Min Edge Width: <span class="range-display" id="minStrokeDisplay">MIN_STROKE</span></label>
            <input type="range" id="minStroke" min="0.1" max="2" value="MIN_STROKE" step="0.1">
        </div>
        <div class="control-group">
            <label for="maxStroke">Max Edge Width: <span class="range-display" id="maxStrokeDisplay">MAX_STROKE</span></label>
            <input type="range" id="maxStroke" min="1" max="10" value="MAX_STROKE" step="0.1">
        </div>
        <div class="control-group">
            <label for="edgeOpacity">Edge Opacity: <span class="range-display" id="edgeOpacityDisplay">EDGE_OPACITY</span></label>
            <input type="range" id="edgeOpacity" min="0.1" max="1" value="EDGE_OPACITY" step="0.05">
        </div>
        <div class="control-group">
            <label for="nodeStrokeWidth">Node Stroke Width: <span class="range-display" id="nodeStrokeWidthDisplay">NODE_STROKE_WIDTH</span></label>
            <input type="range" id="nodeStrokeWidth" min="0" max="5" value="NODE_STROKE_WIDTH" step="0.1">
        </div>
        <div class="control-group">
            <button id="saveSvg" style="width:100%;padding:8px;background-color:#007bff;color:white;border:none;border-radius:4px;cursor:pointer;font-size:14px;"
                    onmouseover="this.style.backgroundColor='#0056b3'"
                    onmouseout="this.style.backgroundColor='#007bff'">
                Save as SVG
            </button>
        </div>
    </div>

    <svg id="graph"></svg>
    <div class="tooltip" id="tooltip"></div>
    <script>
        const graph = GRAPH_DATA;
        const colorConfig = COLOR_CONFIG;
        const graphTitle = "MY_TITLE";
        let customColor = "CUSTOM_COLOR";

        document.getElementById('customColor').value = customColor;

        const svg = d3.select("#graph");
        const width = window.innerWidth;
        const height = window.innerHeight;
        const tooltip = d3.select("#tooltip");

        let minStroke = MIN_STROKE;
        let maxStroke = MAX_STROKE;
        let minNodeRadius = MIN_NODE_RADIUS;
        let maxNodeRadius = MAX_NODE_RADIUS;
        let edgeOpacity = EDGE_OPACITY;
        let nodeStrokeWidth = NODE_STROKE_WIDTH;

        const topMargin = TOP_MARGIN;
        const bottomMargin = BOTTOM_MARGIN;
        const verticalRange = height - topMargin - bottomMargin;

        const linkGroup = svg.append("g").attr("class", "links");
        const nodeGroup = svg.append("g").attr("class", "nodes");
        const labelGroup = svg.append("g").attr("class", "node-labels");

        graph.nodes.forEach(node => {
            node.yTarget = topMargin + node.rank * verticalRange;
            node.x = width / 2 + (Math.random() - 0.5) * 100;
            node.y = node.yTarget;
        });

        const simulation = d3.forceSimulation(graph.nodes)
            .force("link", d3.forceLink(graph.links).id(d => d.id).distance(60).strength(0.3))
            .force("charge", d3.forceManyBody().strength(-150))
            .force("center", d3.forceX(width / 2).strength(0.1))
            .force("y", d3.forceY(d => d.yTarget).strength(MY_VERTICAL_FORCE))
            .force("collision", d3.forceCollide().radius(d => d.display_radius + 5));

        let strokeScale = d3.scaleLinear()
            .domain([0, d3.max(graph.links, d => d.value) || 1])
            .range([minStroke, maxStroke]);

        function updateNodeSizes() {
            const minArea = minNodeRadius ** 2;
            const maxArea = maxNodeRadius ** 2;
            graph.nodes.forEach(node => {
                const area = minArea + node.size * (maxArea - minArea);
                node.display_radius = Math.sqrt(area);
            });
            nodes.attr("r", d => d.display_radius);
            labels.attr("dy", d => -d.display_radius - 2);
            simulation.force("collision", d3.forceCollide().radius(d => d.display_radius + 5));
            simulation.alpha(0.3).restart();
        }

        const links = linkGroup.selectAll("line").data(graph.links).enter().append("line")
            .attr("stroke-width", d => strokeScale(d.value))
            .attr("stroke-opacity", edgeOpacity);

        function getNodeColor(d) {
            const interp = d3.interpolateRgb("#ffffff", customColor);
            const v = colorConfig.min + d.color * (colorConfig.max - colorConfig.min);
            return interp(v);
        }

        const nodes = nodeGroup.selectAll("circle").data(graph.nodes).enter().append("circle")
            .attr("r", d => d.display_radius)
            .attr("fill", getNodeColor)
            .attr("stroke", "#fff")
            .attr("stroke-width", NODE_STROKE_WIDTH)
            .on("mouseover", function(event, d) {
                d3.select(this).attr("stroke", "#000").attr("stroke-width", nodeStrokeWidth + 0.5);
                tooltip.style("left", (event.pageX + 10) + "px")
                       .style("top", (event.pageY - 20) + "px")
                       .style("opacity", 1)
                       .html(`<strong>${d.name}</strong><br>Rank: ${d.rank.toFixed(2)}<br>Size: ${d.size.toFixed(2)}`);
            })
            .on("mouseout", function() {
                d3.select(this).attr("stroke", "#fff").attr("stroke-width", nodeStrokeWidth);
                tooltip.style("opacity", 0);
            })
            .call(d3.drag()
                .on("start", (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
                .on("drag",  (e, d) => { d.fx = e.x; d.fy = e.y; })
                .on("end",   (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

        function truncateText(text, maxLength = 10) {
            return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
        }

        const fontScale = d3.scaleLinear()
            .domain([d3.min(graph.nodes, d => d.display_radius), d3.max(graph.nodes, d => d.display_radius)])
            .range([8, 12]);

        const labels = labelGroup.selectAll("text").data(graph.nodes).enter().append("text")
            .text(d => truncateText(d.name))
            .attr("dy", d => -d.display_radius - 2)
            .style("font-size", d => fontScale(d.display_radius) + "px");

        document.getElementById('showLabels').addEventListener('change', function() {
            labelGroup.style('display', this.checked ? 'block' : 'none');
        });
        document.getElementById('customColor').addEventListener('input', function() {
            customColor = this.value; nodes.attr('fill', getNodeColor);
        });
        document.getElementById('minNodeRadius').addEventListener('input', function() {
            minNodeRadius = +this.value;
            document.getElementById('minNodeRadiusDisplay').textContent = minNodeRadius;
            updateNodeSizes();
        });
        document.getElementById('maxNodeRadius').addEventListener('input', function() {
            maxNodeRadius = +this.value;
            document.getElementById('maxNodeRadiusDisplay').textContent = maxNodeRadius;
            updateNodeSizes();
        });
        document.getElementById('minStroke').addEventListener('input', function() {
            minStroke = +this.value;
            document.getElementById('minStrokeDisplay').textContent = minStroke;
            strokeScale.range([minStroke, maxStroke]);
            links.attr("stroke-width", d => strokeScale(d.value));
        });
        document.getElementById('maxStroke').addEventListener('input', function() {
            maxStroke = +this.value;
            document.getElementById('maxStrokeDisplay').textContent = maxStroke;
            strokeScale.range([minStroke, maxStroke]);
            links.attr("stroke-width", d => strokeScale(d.value));
        });
        document.getElementById('edgeOpacity').addEventListener('input', function() {
            edgeOpacity = +this.value;
            document.getElementById('edgeOpacityDisplay').textContent = edgeOpacity;
            links.attr("stroke-opacity", edgeOpacity);
        });
        document.getElementById('nodeStrokeWidth').addEventListener('input', function() {
            nodeStrokeWidth = +this.value;
            document.getElementById('nodeStrokeWidthDisplay').textContent = nodeStrokeWidth.toFixed(1);
            nodes.attr("stroke-width", nodeStrokeWidth);
        });

        document.getElementById('saveSvg').addEventListener('click', function() {
            const svgEl = document.getElementById('graph');
            const svgCopy = svgEl.cloneNode(true);
            svgCopy.setAttribute('width', width);
            svgCopy.setAttribute('height', height);
            svgCopy.setAttribute('viewBox', `0 0 ${width} ${height}`);
            svgCopy.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
            if (!document.getElementById('showLabels').checked) {
                svgCopy.querySelectorAll('.node-labels').forEach(el => el.remove());
            }
            const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
            style.textContent = '.links line{stroke:#999} .nodes circle{stroke:#fff} .node-labels{font-size:10px;text-anchor:middle;font-family:Arial,sans-serif}';
            svgCopy.insertBefore(style, svgCopy.firstChild);
            const blob = new Blob([new XMLSerializer().serializeToString(svgCopy)], {type: 'image/svg+xml'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = `${graphTitle}.svg`;
            document.body.appendChild(a); a.click();
            document.body.removeChild(a); URL.revokeObjectURL(url);
        });

        simulation.on("tick", () => {
            graph.nodes.forEach(node => {
                node.x = Math.max(node.display_radius, Math.min(width - node.display_radius, node.x));
            });
            links.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                 .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
            nodes.attr("cx", d => d.x).attr("cy", d => d.y);
            labels.attr("x", d => d.x).attr("y", d => d.y);
        });

        window.addEventListener("resize", () => {
            const w = window.innerWidth, h = window.innerHeight;
            const vr = h - topMargin - bottomMargin;
            graph.nodes.forEach(node => { node.yTarget = topMargin + node.rank * vr; });
            simulation.force("y", d3.forceY(d => d.yTarget).strength(0.5))
                      .force("center", d3.forceX(w / 2).strength(0.1));
            simulation.alpha(0.3).restart();
        });
    </script>
</body>
</html>"""

    html_content = (html_content
        .replace("GRAPH_DATA", json.dumps(graph_data))
        .replace("COLOR_CONFIG", json.dumps(color_config))
        .replace("TOP_MARGIN", str(top_margin))
        .replace("BOTTOM_MARGIN", str(bottom_margin))
        .replace("MIN_NODE_RADIUS", str(min_node_radius))
        .replace("MAX_NODE_RADIUS", str(max_node_radius))
        .replace("MIN_STROKE", str(min_stroke))
        .replace("MAX_STROKE", str(max_stroke))
        .replace("EDGE_OPACITY", str(edge_opacity))
        .replace("NODE_STROKE_WIDTH", str(node_stroke_width))
        .replace("CUSTOM_COLOR", custom_color)
        .replace("MY_TITLE", Path(output_file).stem)
        .replace("MY_VERTICAL_FORCE", str(vertical_force))
    )

    with open(output_file, "w") as f:
        f.write(html_content)
    return output_file


# ---------------------------------------------------------------------------
# Main: generate one HTML per field
# ---------------------------------------------------------------------------
for field in FIELDS:
    print(f"  {field}...", end=" ", flush=True)

    adj, node_ids = create_field_user_chain_graph(field, K, seed=SEED)

    # Consensus field ranks for vertical positioning (negate: higher score = top)
    rank_series = (
        field_consensus[field_consensus["field"] == field]
        .set_index("venue_db_id")["score"]
    )
    ranks = -np.array(rank_series.reindex(node_ids).tolist())

    names = [id_to_name[nid] for nid in node_ids]
    sizes = np.array([consideration_counts[field][nid] for nid in node_ids])

    out = str(OUT_DIR / f"{field}.html")
    create_vertical_force_directed_graph(
        adj, ranks,
        node_names=names, node_sizes=sizes, node_colors=sizes,
        min_stroke=2, max_stroke=8, edge_opacity=0.5,
        node_stroke_width=3,
        color_range_min=0.5, color_range_max=1.0,
        min_node_radius=5, max_node_radius=25,
        vertical_force=5,
        custom_color=field_color_map[field],
        output_file=out,
    )
    print(f"saved → {Path(out).name}")

print(f"\nDone. HTML files written to {OUT_DIR}/")
