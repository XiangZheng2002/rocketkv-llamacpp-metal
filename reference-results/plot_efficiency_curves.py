from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "efficiency_results.csv"

BUDGET_STYLES = {
    256: {"color": "#0072B2", "marker": "o"},
    512: {"color": "#E69F00", "marker": "s"},
    1024: {"color": "#009E73", "marker": "D"},
    2048: {"color": "#D55E00", "marker": "^"},
    4096: {"color": "#CC79A7", "marker": "v"},
}


def load_efficiency_results(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    arrays = [data['ctx_len'].values, data['method'].values, data['budget'].values]
    index = pd.MultiIndex.from_arrays(arrays, names=["ctx_len", "method", "budget"])
    df = pd.DataFrame(
        data[['decode_time (s)', 'peak_mem_prefill (GB)', 'peak_mem_decode (GB)']].values,
        index=index,
        columns=['decode_time', 'peak_mem_prefill', 'peak_mem_decode'],
    )
    return df


def compute_rocket_metrics(data: pd.DataFrame) -> pd.DataFrame:
    idx = pd.IndexSlice
    rocket_df = data.loc[idx[:, 'rocket', :], :].reset_index()
    baseline_df = data.loc[idx[:, 'baseline', :], :].reset_index()
    rocket_df = rocket_df.drop(columns=['method'])
    baseline_df = baseline_df.drop(columns=['method', 'budget'])
    df2 = pd.merge(rocket_df, baseline_df, on=['ctx_len'], suffixes=('_rocket', '_baseline'))
    df2['speedup'] = df2['decode_time_baseline'] / df2['decode_time_rocket']
    df2['memory_savings_pct'] = (
        (df2['peak_mem_decode_baseline'] - df2['peak_mem_decode_rocket'])
        / df2['peak_mem_decode_baseline']
        * 100.0
    )
    df2['budget'] = df2['budget'].astype(int)
    return df2.sort_values(['budget', 'ctx_len'])



def _style_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("Sequence length (tokens)")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([16384, 31744, 65536, 96256])
    ax.set_xticklabels(["16K", "32K", "64K", "96K"])


def _plot_budget_lines(ax: plt.Axes, rocket: pd.DataFrame, y_col: str) -> None:
    for budget, style in BUDGET_STYLES.items():
        series = rocket.loc[rocket["budget"] == budget]
        if series.empty:
            continue
        ax.plot(
            series["ctx_len"],
            series[y_col],
            label=f"RocketKV-{budget}",
            linewidth=2.2,
            markersize=7,
            **style,
        )
    ax.legend(frameon=False, loc="best")


def plot_decode_speedup(data: pd.DataFrame, output_path: Path | None = None) -> Path:
    rocket = compute_rocket_metrics(data)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _plot_budget_lines(ax, rocket, "speedup")
    ax.axhline(1.0, color="0.4", linestyle=":", linewidth=1.2)
    ax.set_ylabel("Decode Speedup")
    ax.set_title("Decode Speedup vs Sequence Length")
    _style_axes(ax)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    if output_path is None:
        output_path = ROOT / "decode_speedup_vs_sequence_length.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_memory_savings(data: pd.DataFrame, output_path: Path | None = None) -> Path:
    rocket = compute_rocket_metrics(data)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    _plot_budget_lines(ax, rocket, "memory_savings_pct")
    ax.set_ylabel("Memory savings (%)")
    ax.set_title("Memory savings vs sequence length")
    _style_axes(ax)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    if output_path is None:
        output_path = ROOT / "memory_savings_vs_sequence_length.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    data = load_efficiency_results()
    speedup_path = plot_decode_speedup(data)
    memory_path = plot_memory_savings(data)


if __name__ == "__main__":
    main()
