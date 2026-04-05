"""Run a paired t-test on semantic (BERTScore F1) evaluation results.

Compares two model result CSVs produced by run_test_semantics.py using
a paired Student's t-test on the bert_f1 columns.

Usage
-----
    python run_stat_test_on_semantics.py \
        --main-dir  analysis/results/semantics_qlora_no_ir.csv \
        --consistent-dir analysis/results/semantics_qlora_ir.csv \
        --model-to-use 1
"""

import argparse

import pandas as pd

from stat_classes import TTestRunner


def build_parser():
    parser = argparse.ArgumentParser(
        description="Paired t-test on semantic (BERTScore F1) model results"
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
            "is used as the baseline (model 1 / group A). When omitted "
            "the model 1 column from --main-dir is used directly."
        ),
    )
    parser.add_argument(
        "--model-to-use",
        choices=["1", "2"],
        default="1",
        help=(
            "Which model column to pull from --consistent-dir as group A. "
            "1 = bert_f1_model1, 2 = bert_f1_model2."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    df = pd.read_csv(args.main_dir)

    if args.consistent_dir is None:
        if args.model_to_use == "1":
            group_a = df["bert_f1_model1"].values
        else:
            group_a = df["bert_f1_model2"].values
        group_b = df["bert_f1_model2"].values
    else:
        if args.model_to_use == "1":
            df_m1 = pd.read_csv(args.consistent_dir)[
                ["prompt", "ground_truth_completion", "bert_f1_model1"]
            ].rename(columns={"bert_f1_model1": "group_a"})
        else:
            df_m1 = pd.read_csv(args.consistent_dir)[
                ["prompt", "ground_truth_completion", "bert_f1_model2"]
            ].rename(columns={"bert_f1_model2": "group_a"})

        df_m2 = df[
            ["prompt", "ground_truth_completion", "bert_f1_model2"]
        ].rename(columns={"bert_f1_model2": "group_b"})

        merged = df_m1.merge(
            df_m2, on=["prompt", "ground_truth_completion"], how="inner"
        )
        group_a = merged["group_a"].values
        group_b = merged["group_b"].values

    print(f"Group A mean BERTScore F1 : {group_a.mean():.4f}")
    print(f"Group B mean BERTScore F1 : {group_b.mean():.4f}")

    runner = TTestRunner(group_a, group_b)
    runner.check_assumptions()
    runner.run_test()