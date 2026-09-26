from functions.config import PLOTS_DIR, RESULTS_DIR
from functions.feasibility import read_feasibility
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import h5py
import pandas as pd
import seaborn as sns
from cycler import cycler


# widths the figures are included at: the full text block and a single column
TEXT_WIDTH_PT = 481.89
COLUMN_WIDTH_PT = 230.98

MARKER_SCALE = 0.7

OKABE_ITO = [
    "#56B4E9", # Sky Blue
    "#0072B2", # Blue
    "#E69F00", # Orange
    "#D55E00", # Vermilion
    "#009E73", # Bluish Green
    "#CC79A7", # Reddish Purple
    "#000000", # Black
    "#F0E442", # Yellow
]

EXPLICIT = "Explicit Construction"
UNINFORMED = "Uninformed LP Relaxation"
RAY = r"Ray Binary Search ($1 / \hat{\gamma}$)"
INFORMED = "Informed LP Algorithm"
OPTIMAL = "Optimal Solution"

ALGO_STYLES = {
    EXPLICIT: {"color": "#CC79A7", "marker": "s", "dash": (4, 1.5), "size": 4.0},
    UNINFORMED: {"color": "#0072B2", "marker": "D", "dash": (3, 1, 1, 1), "size": 4.0},
    RAY: {"color": "#E69F00", "marker": "P", "dash": (2, 1), "size": 5.5},
    INFORMED: {"color": "#D55E00", "marker": "o", "dash": "", "size": 5.5},
    OPTIMAL: {"color": "#009E73", "marker": "X", "dash": (1, 1), "size": 5.5},
}

RUN_TIME_LABEL = r"Quantum Run Time $\lVert\pmb{\lambda}\rVert_{\ell_1}$"


def set_style():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath,mathtools,amsthm} \usepackage{amssymb} \usepackage{dsfont} \usepackage{lmodern} \usepackage[T1]{fontenc}",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.prop_cycle": cycler(color=OKABE_ITO)
    })


def savefig_exact_width(fig, path, width_pt=TEXT_WIDTH_PT, pad_inches=None, dpi=300):
    """Save with the tight bounding box widened evenly on both sides to exactly width_pt."""
    if pad_inches is None:
        pad_inches = plt.rcParams["savefig.pad_inches"]

    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad_inches)

    delta = (width_pt / 72 - bbox.width) / 2
    bbox = Bbox.from_extents(bbox.x0 - delta, bbox.y0, bbox.x1 + delta, bbox.y1)

    fig.savefig(path, bbox_inches=bbox, pad_inches=0, dpi=dpi)


def load_attrs(results_file):
    with h5py.File(RESULTS_DIR / results_file, "r") as f:
        return dict(f.attrs)


def line_style(algo):
    dash = ALGO_STYLES[algo]["dash"]

    return "-" if dash == "" else (0, dash)


def algo_lineplot(ax, data, x, algo, **kwargs):
    # median over the runs, with the full range shaded
    props = ALGO_STYLES[algo]

    sns.lineplot(
        data=data,
        x=x,
        y="q_time",
        color=props["color"],
        marker=props["marker"],
        linestyle=line_style(algo),
        estimator=np.median,
        errorbar=("pi", 100),
        linewidth=1.2,
        markersize=props["size"] * MARKER_SCALE,
        err_kws={'alpha': 0.15},
        markeredgewidth=0.5,
        ax=ax,
        label=algo,
        **kwargs
    )


def algo_hline(ax, y, xmin, xmax, algo, label=None):
    ax.hlines(y=y, xmin=xmin, xmax=xmax, color=ALGO_STYLES[algo]["color"], linestyle=line_style(algo), linewidth=1.2, label=label or algo)


def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('black')

    ax.tick_params(axis='both', direction='out', length=3.5, width=0.8, colors='black')


def label_panel(ax, label, x_offset):
    ax.annotate(label, xy=(x_offset, 1.02), xycoords='axes fraction', fontsize=10, ha='right', va='bottom')


def k_title(k):
    return r"$k = \infty$" if np.isinf(k) else fr"$k = {k}$"


def legend_above(fig, handles, labels, ncol):
    legend = fig.legend(
        handles,
        labels,
        title="Algorithm",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=ncol,
        framealpha=0.8,
        edgecolor='black'
    )
    legend.get_frame().set_linewidth(0.8)


def complete_bipartite_plot():
    set_style()

    ratios = np.linspace(2, 4, 20)
    runs_per_ratio = 50
    results = "paper_results/complete_bipartite/complete_bipartite"

    q_time_explicit = load_attrs(f"{results}_explicit.hdf5")["q_time"]
    q_time_exact = load_attrs(f"{results}_exact.hdf5")["q_time"]
    q_time_ray = load_attrs(f"{results}_4.0_0_cut_poly.hdf5")["q_time_exp"]

    data = []
    for ratio in ratios:
        for run in range(runs_per_ratio):
            data.append([ratio, load_attrs(f"{results}_{ratio}_{run}_cut_poly.hdf5")["q_time"], INFORMED])
            data.append([ratio, load_attrs(f"{results}_{ratio}_{run}_eff_relax.hdf5")["q_time"], UNINFORMED])

    df = pd.DataFrame(data=data, columns=["s_m_ratio", "q_time", "algo"])

    fig, ax = plt.subplots(figsize=(3.35, 2.8))

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='gray')
    ax.set_ylim((0, 50))

    algo_hline(ax, q_time_explicit, 2, 4, EXPLICIT, label=r"Explicit Construction\textsuperscript{*}")
    algo_lineplot(ax, df[df["algo"] == UNINFORMED], "s_m_ratio", UNINFORMED)
    algo_hline(ax, q_time_ray, 2, 4, RAY)
    algo_lineplot(ax, df[df["algo"] == INFORMED], "s_m_ratio", INFORMED)
    algo_hline(ax, q_time_exact, 2, 4, OPTIMAL)

    ax.set_xlabel(r"$s/m$")
    ax.set_ylabel(RUN_TIME_LABEL)

    style_axes(ax)

    legend = ax.legend(title="Algorithm", loc="upper right", framealpha=0.8, edgecolor='black')
    legend.get_frame().set_linewidth(0.8)

    savefig_exact_width(fig, PLOTS_DIR / 'complete_bipartite.pdf', COLUMN_WIDTH_PT)
    plt.close(fig)


def edge_surpression_plot():
    set_style()

    n = 20
    graphs_per_density = 20
    max_edges = n * (n - 1) // 2

    data = []
    for m in range(10, max_edges + 1, 10):
        for r in range(graphs_per_density):
            results = f"paper_results/edge_surpression/n={n}_m={m}_instance={r}"
            cut_poly = load_attrs(f"{results}_cut_poly.hdf5")

            data.append([m / max_edges, cut_poly["q_time"], INFORMED])
            data.append([m / max_edges, cut_poly["q_time_exp"], RAY])
            data.append([m / max_edges, load_attrs(f"{results}_eff_relax.hdf5")["q_time"], UNINFORMED])
            data.append([m / max_edges, load_attrs(f"{results}_explicit.hdf5")["q_time"], EXPLICIT])
            data.append([m / max_edges, load_attrs(f"{results}_exact.hdf5")["q_time"], OPTIMAL])

    df = pd.DataFrame(data=data, columns=["edges_surpressed", "q_time", "algo"])

    fig, ax = plt.subplots(figsize=(3.35, 2.8))

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='gray')

    for algo in [EXPLICIT, UNINFORMED, RAY, INFORMED, OPTIMAL]:
        algo_lineplot(ax, df[df["algo"] == algo], "edges_surpressed", algo)

    ax.set_xlabel(r"Target Edge Density Ratio")
    ax.set_ylabel(RUN_TIME_LABEL)
    ax.set_ylim(bottom=0, top=35)

    style_axes(ax)

    legend = ax.legend(title="Algorithm", loc="upper right", framealpha=0.8, edgecolor='black')
    legend.get_frame().set_linewidth(0.8)

    savefig_exact_width(fig, PLOTS_DIR / 'edge_surpression.pdf', COLUMN_WIDTH_PT)
    plt.close(fig)


def hofstadter_load_data(ns, ks, runs):
    data = []

    for k, n, r in product(ks, ns, runs):
        results = f"paper_results/hofstadter/hofstadter_{n}_k={k}"
        cut_poly = load_attrs(f"{results}_cut_poly_{r}.hdf5")

        # q_time_exp is only written if one of the LPs was feasible
        if "q_time_exp" in cut_poly:
            data.append([k, n, RAY, r, cut_poly["q_time_exp"]])
        else:
            print(f"Ray binary search unsuccessful for n={n}, k={k}, r={r}")

        data.append([k, n, INFORMED, r, cut_poly["q_time"]])
        data.append([k, n, UNINFORMED, r, load_attrs(f"{results}_eff_relax_{r}.hdf5")["q_time"]])

    return pd.DataFrame(data=data, columns=["k", "n", "algo", "run", "q_time"])


def hofstadter_plot_combined():
    set_style()

    ks = [3, 4, np.inf]
    df = hofstadter_load_data(range(3, 21), ks, range(50))

    # the top row shows the full range, the bottom row zooms in on the small run times
    row_tops = [50, 5]
    subplot_labels = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']]

    fig, axs = plt.subplots(len(row_tops), len(ks), figsize=(6.8, 4.8), sharex=True, sharey="row")

    for irow, top in enumerate(row_tops):
        for ik, k in enumerate(ks):
            ax = axs[irow, ik]

            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='gray')

            for algo in [UNINFORMED, RAY, INFORMED]:
                algo_lineplot(ax, df[(df["algo"] == algo) & (df["k"] == k)], "n", algo, legend=False)

            ax.set_ylim(bottom=0, top=top)
            ax.set_xlabel(r"$L$" if irow == len(row_tops) - 1 else "")
            ax.set_ylabel(RUN_TIME_LABEL if ik == 0 else "")

            if irow == 0:
                ax.set_title(k_title(k), pad=4)

            label_panel(ax, subplot_labels[irow][ik], 0.09 if ik == 0 else 0.095)
            style_axes(ax)

    handles, labels = axs[0, 0].get_legend_handles_labels()

    plt.tight_layout()

    # the legend goes into the empty lower part of the bottom row, centred on the saved
    # area (which includes the y axis labels) rather than on the axes
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    legend = fig.legend(
        handles,
        labels,
        title="Algorithm",
        loc="lower center",
        bbox_to_anchor=(
            (tight.x0 + tight.x1) / 2 / fig.get_figwidth(),
            axs[-1, 0].get_position().y0 + 0.015
        ),
        bbox_transform=fig.transFigure,
        ncol=3,
        framealpha=0.8,
        edgecolor='black'
    )
    legend.get_frame().set_linewidth(0.8)

    savefig_exact_width(fig, PLOTS_DIR / "hofstadter_combined.pdf")
    plt.close(fig)


def chiral_clock_load_data(phi_ks, phi_steps, g_ks, gs, runs):
    tasks = [("phi", k, phi, f"phi={phi}_g=0") for k in phi_ks for phi in np.linspace(0, np.pi / k, phi_steps)]
    tasks += [("g", k, g, f"phi=0_g={g}") for k in g_ks for g in gs]

    data = []

    for sweep, k, x, instance in tasks:
        results = f"paper_results/chiral_clock/{instance}_k={k}"

        data.append([sweep, k, x, OPTIMAL, 0, load_attrs(f"{results}_exact.hdf5")["q_time"]])

        for r in runs:
            cut_poly = load_attrs(f"{results}_cut_poly_{r}.hdf5")
            if np.isfinite(cut_poly["q_time"]):
                data.append([sweep, k, x, RAY, r, cut_poly["q_time_exp"]])
                data.append([sweep, k, x, INFORMED, r, cut_poly["q_time"]])
            else:
                print(f"Informed LP algorithm infeasible for {instance}, k={k}, r={r}")

            eff_relax = load_attrs(f"{results}_eff_relax_{r}.hdf5")
            if np.isfinite(eff_relax["q_time"]):
                data.append([sweep, k, x, UNINFORMED, r, eff_relax["q_time"]])
            else:
                print(f"Uninformed relaxation infeasible for {instance}, k={k}, r={r}")

    return pd.DataFrame(data=data, columns=["sweep", "k", "x", "algo", "run", "q_time"])


def chiral_clock_phi_plot():
    set_style()

    ks = [3, 4, 5]
    df = chiral_clock_load_data(ks, 21, [], [], range(50))

    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.6), sharey=True)

    for ik, k in enumerate(ks):
        ax = axs[ik]

        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='gray')

        for algo in [UNINFORMED, RAY, INFORMED, OPTIMAL]:
            algo_lineplot(ax, df[(df["algo"] == algo) & (df["k"] == k)], "x", algo, legend=False)

        ax.set_xlim(0, np.pi / k)
        ax.set_xticks([0, np.pi / (2 * k), np.pi / k])
        ax.set_xticklabels([r"$0$", fr"$\pi / {2 * k}$", fr"$\pi / {k}$"])
        ax.set_xlabel(r"$\varphi$")
        ax.set_ylim(bottom=0, top=6)
        ax.set_ylabel(RUN_TIME_LABEL if ik == 0 else "")
        ax.set_title(k_title(k), pad=4)

        label_panel(ax, ['(a)', '(b)', '(c)'][ik], 0.09 if ik == 0 else 0.095)
        style_axes(ax)

    handles, labels = axs[0].get_legend_handles_labels()
    legend_above(fig, handles, labels, ncol=4)

    plt.tight_layout()

    savefig_exact_width(fig, PLOTS_DIR / "chiral_clock_phi.pdf")
    plt.close(fig)


def chiral_clock_g_plot():
    set_style()

    ks = [2, 3, 4, 5]
    gs = np.linspace(0, 2, 21)
    df = chiral_clock_load_data([], 0, ks, gs, range(50))

    fig, axs = plt.subplots(2, 2, figsize=(6.8, 5.2), sharex=True, sharey=True)

    for ik, k in enumerate(ks):
        ax = axs.flat[ik]

        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='gray')

        for algo in [UNINFORMED, RAY, INFORMED, OPTIMAL]:
            algo_lineplot(ax, df[(df["algo"] == algo) & (df["k"] == k)], "x", algo, legend=False)

        ax.set_xlim(gs[0], gs[-1])
        ax.set_xticks(np.linspace(gs[0], gs[-1], 5))
        ax.set_ylim(bottom=0, top=12)
        ax.set_xlabel(r"$g / J$" if ik >= 2 else "")
        ax.set_ylabel(RUN_TIME_LABEL if ik % 2 == 0 else "")
        ax.set_title(k_title(k), pad=4)

        label_panel(ax, ['(a)', '(b)', '(c)', '(d)'][ik], 0.06 if ik % 2 == 0 else 0.065)
        style_axes(ax)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend_above(fig, handles, labels, ncol=4)

    plt.tight_layout()

    savefig_exact_width(fig, PLOTS_DIR / "chiral_clock_g.pdf")
    plt.close(fig)


def feasibility_load_data(results_file):
    """Kept instances with their threshold r*, the median of s* / D over the repetitions."""
    tables = read_feasibility(results_file)

    instances = tables["instances"]
    instances = instances[instances["discarded"] == 0].copy()

    thresholds = tables["thresholds"]
    thresholds = thresholds[thresholds["instance_id"].isin(instances["instance_id"])]

    # a repetition without success within s_max counts as s* = inf
    r_star = thresholds["r_star"].fillna(np.inf).groupby(thresholds["instance_id"]).median()
    instances = instances.join(r_star, on="instance_id")

    return instances, thresholds


def feasibility_plot(ks=(2, 3, 4, np.inf), cap=630, out="feasibility.pdf",
                     width_pt=0.72 * TEXT_WIDTH_PT):
    """Success probability over s / D for the base ensemble at every k and for the
    interpolated (stress) instances at k = 4, binned by their saturation."""
    set_style()

    # fine enough for the inset
    ratios = np.linspace(1.0, 12.0, 4401)

    def curve(r_star, r_max):
        # empirical P[s* <= r D], continued flat beyond the r_max the arm was run to
        r = np.sort(np.nan_to_num(np.asarray(r_star, dtype=float), nan=np.inf))
        p = np.searchsorted(r, ratios, side="right") / len(r)
        return np.where(ratios > r_max, p[ratios <= r_max][-1], p)

    base_i, base_t = feasibility_load_data("paper_results/feasibility.hdf5")
    stress_i, stress_t = feasibility_load_data("paper_results/feasibility_stress.hdf5")

    stress_t = stress_t.merge(stress_i[["instance_id", "saturation"]], on="instance_id")

    fig, ax = plt.subplots(figsize=(4.8, 2.9))

    axins = ax.inset_axes([0.23, 0.36, 0.27, 0.56])

    for a in (ax, axins):
        a.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
        a.axvline(2, color="black", linestyle="--", linewidth=0.8)

    # the solid k = 4 curve first, so the dashed ones stay visible on top of it
    dashes = {4: "-", 2: (0, (1, 1.2)), 3: (0, (4, 1.5)), np.inf: (0, (4, 1.2, 1, 1.2))}
    base_handles = {}
    for kk in sorted(ks, key=lambda kk: kk != 4):
        keep = base_i[(base_i["k"] == kk) & (base_i["D"] <= cap)]
        r = base_t[base_t["instance_id"].isin(keep["instance_id"])]["r_star"]
        k_str = r"\infty" if np.isinf(kk) else f"{int(kk)}"
        p = curve(r, 3.0)
        (base_handles[kk],) = ax.plot(ratios, p, color="#D55E00", linewidth=1.3,
                                      linestyle=dashes[kk], label=fr"$k = {k_str}$")
        axins.plot(ratios, p, color="#D55E00", linewidth=1.3, linestyle=dashes[kk])

    # light to dark with growing saturation
    shades = sns.blend_palette(["#56B4E9", "#0072B2", "#002A4D"], 3)
    stress_handles = []
    for (lo, hi), colour in zip([(0.45, 0.6), (0.75, 0.85), (0.85, 0.93)], shades):
        sel = stress_t[(stress_t["saturation"] > lo) & (stress_t["saturation"] <= hi)]
        if len(sel):
            p = curve(sel["r_star"], 12.0)
            (h,) = ax.plot(ratios, p, color=colour, linewidth=1.5,
                           label=r"$\varsigma \approx %.2f$" % sel["saturation"].mean())
            axins.plot(ratios, p, color=colour, linewidth=1.5)
            stress_handles.append(h)

    ax.set_xlim(1.0, 12.0)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"$s / D$")
    ax.set_ylabel(r"Success Probability")

    axins.set_xlim(1.5, 2.6)
    axins.set_ylim(-0.03, 1.03)
    axins.tick_params(labelsize=7, labelleft=False)
    # drawn below the spines, which it would otherwise streak grey
    ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=1.0, linewidth=0.4, zorder=2.4)

    # one legend, split into the two ensembles by header rows without handles
    header = Line2D([], [], linestyle="none")
    headers = ["Base", r"Interpolated, $k = 4$"]
    handles = [header] + [base_handles[kk] for kk in ks] + [header] + stress_handles
    labels = [h.get_label() for h in handles]
    labels[0], labels[len(ks) + 1] = headers

    legend = ax.legend(
        handles,
        labels,
        title="Ensemble",
        loc="lower right",
        bbox_to_anchor=(0.99, 0.06),
        framealpha=0.8,
        edgecolor="black"
    )
    legend.get_frame().set_linewidth(0.8)

    # pull the headers flush left
    for row in legend._legend_handle_box.get_children()[0].get_children():
        handlebox, textbox = row.get_children()
        if textbox.get_text() in headers:
            handlebox.width = 0
            row.sep = 0

    for a in (ax, axins):
        style_axes(a)

    savefig_exact_width(fig, PLOTS_DIR / out, width_pt)
    plt.close(fig)
