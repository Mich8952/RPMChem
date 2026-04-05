"""Run paired statistical tests on numerical evaluation results.

Compares two model result CSVs produced by run_test_numerical.py using
the Wilcoxon signed-rank test at multiple relative-error penalty caps.

Usage
-----
    python run_stat_test_on_numerical.py \
        --main-dir  analysis/results/numerical_qlora_ir.csv \
        --consistent-dir analysis/results/numerical_qlora_no_ir.csv \
        --model-to-use 1
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stat_classes import WilcoxonRunner

plt.style.use("ggplot")

# Maximum relative error cap; also assigned when a model fails to predict.
CAP_GLOB = 10


def relative_error(y_true, y_hat):
    return np.abs(y_hat - y_true) / np.abs(y_true)


def compute_err(y_true, y_hat, cap=CAP_GLOB):
    if pd.isna(y_hat):
        return cap
    return min(relative_error(y_true, y_hat), cap)


def penalty_analysis(df):
    """Plot mean relative error vs penalty cap for two models.

    Overlays a colour band showing where the Wilcoxon test is
    statistically significant (green) or not (red).
    """
    group_a = df[["model1_converted_value", "ground_truth"]]
    group_b = df[["model2_converted_value", "ground_truth"]]

    penalties = [2, 4, 7, 8, 10, 12, 14, 16, 18]

    per_pen_a = [
        np.mean(
            [
                compute_err(
                    group_a.iloc[i]["ground_truth"],
                    group_a.iloc[i]["model1_converted_value"],
                    cap=pen,
                )
                for i in range(len(group_a))
            ]
        )
        for pen in penalties
    ]

    per_pen_b = [
        np.mean(
            [
                compute_err(
                    group_b.iloc[i]["ground_truth"],
                    group_b.iloc[i]["model2_converted_value"],
                    cap=pen,
                )
                for i in range(len(group_b))
            ]
        )
        for pen in penalties
    ]

    stat_results = []
    for pen in penalties:
        errs_a = [
            compute_err(
                group_a.iloc[i]["ground_truth"],
                group_a.iloc[i]["model1_converted_value"],
                cap=pen,
            )
            for i in range(len(group_a))
        ]
        errs_b = [
            compute_err(
                group_b.iloc[i]["ground_truth"],
                group_b.iloc[i]["model2_converted_value"],
                cap=pen,
            )
            for i in range(len(group_b))
        ]
        runner = WilcoxonRunner(np.array(errs_a), np.array(errs_b))
        stat_results.append(runner.run_test())

    sig_bool = [r["significant"] for r in stat_results]

    plt.clf()
    plt.plot(penalties, per_pen_a, label="Model 1")
    plt.plot(penalties, per_pen_b, label="Model 2")

    x_dense = np.linspace(min(penalties) - 0.5, max(penalties) + 0.5, 600)
    sig_dense = np.interp(x_dense, penalties, np.array(sig_bool, dtype=float))
    rgb = np.stack(
        [1.0 - sig_dense, sig_dense, np.zeros_like(sig_dense)], axis=1
    )[None, :, :]
    y0, y1 = plt.gca().get_ylim()
    plt.gca().imshow(
        rgb,
        extent=[x_dense.min(), x_dense.max(), y0, y1],
        origin="lower",
        aspect="auto",
        alpha=0.18,
        zorder=0,
    )
    plt.grid()
    plt.title(
        "Mean Relative Error vs Penalty Cap\n"
        "Statistical Significance shown in Green"
    )
    plt.ylabel("Mean Relative Error")
    plt.xlabel("Penalty Cap (value assigned to failed predictions)")
    plt.legend()
    plt.show()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Paired Wilcoxon test on numerical model results"
    )
    parser.add_argument(
        "--main-dir",
        required=True,
        help="Path to the primary results CSV (model 2 column is used).",
    )
    parser.add_argument(
        "--consistent-dir",
        default=None,
        help=(
            "Optional path to a second results CSV whose model column "
            "is used as the baseline (model 1). When omitted the model 1 "
            "column from --main-dir is used directly."
        ),
    )
    parser.add_argument(
        "--model-to-use",
        choices=["1", "2"],
        default="1",
        help=(
            "Which model column to pull from --consistent-dir as the "
            "baseline. 1 = model1_*, 2 = model2_* (renamed to model1_*)."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    df = pd.read_csv(args.main_dir)

    if args.consistent_dir is None:
        if args.model_to_use == "2":
            df["model1_converted_value"] = df["model2_converted_value"]
            df["model1_ans"] = df["model2_ans"]
    else:
        if args.model_to_use == "1":
            df_m1 = pd.read_csv(args.consistent_dir)[
                ["prompt", "ground_truth", "model1_converted_value", "model1_ans"]
            ]
        else:
            df_m1 = (
                pd.read_csv(args.consistent_dir)[
                    [
                        "prompt",
                        "ground_truth",
                        "model2_converted_value",
                        "model2_ans",
                    ]
                ]
                .rename(
                    columns={
                        "model2_converted_value": "model1_converted_value",
                        "model2_ans": "model1_ans",
                    }
                )
            )

        df_m2 = df[["prompt", "ground_truth", "model2_converted_value", "model2_ans"]]
        df_m1["ground_truth"] = pd.to_numeric(
            df_m1["ground_truth"], errors="coerce"
        )
        df_m2["ground_truth"] = pd.to_numeric(
            df_m2["ground_truth"], errors="coerce"
        )
        df = df_m1.merge(df_m2, on=["prompt", "ground_truth"], how="inner")

    df = df[df["ground_truth"].notna()].reset_index(drop=True)

    print(f"N samples          : {len(df)}")
    print(
        f"Model 1 NaN rate   : "
        f"{df['model1_ans'].isna().sum() / len(df) * 100:.1f}%"
    )
    print(
        f"Model 2 NaN rate   : "
        f"{df['model2_ans'].isna().sum() / len(df) * 100:.1f}%"
    )

    err1 = df.apply(
        lambda r: compute_err(r["ground_truth"], r["model1_ans"]), axis=1
    )
    err2 = df.apply(
        lambda r: compute_err(r["ground_truth"], r["model2_ans"]), axis=1
    )

    print(f"Mean error Model 1 : {err1.mean():.4f}")
    print(f"Mean error Model 2 : {err2.mean():.4f}")

    runner = WilcoxonRunner(err1.values, err2.values)
    runner.check_assumptions()
    runner.run_test()

    penalty_analysis(df)