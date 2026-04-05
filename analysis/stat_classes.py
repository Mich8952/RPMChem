"""Statistical test runners for paired-sample comparisons.

Classes
-------
TTestRunner
    Paired Student's t-test.
WilcoxonRunner
    Wilcoxon signed-rank test (non-parametric alternative).
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import os
import sys

# Ensure stat_classes is found regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TTestRunner:
    """Paired Student's t-test between two equal-length samples."""

    def __init__(self, group_a, group_b, alpha=0.05):
        """
        Args:
            group_a: Array-like of numerical values for group A.
            group_b: Array-like of numerical values for group B.
            alpha: Significance level for hypothesis testing.
        """
        self.group_a = np.array(group_a)
        self.group_b = np.array(group_b)
        self.alpha = alpha

    def check_assumptions(self, bins=None):
        """Plot delta distribution and per-group histograms for visual inspection.

        If the delta histogram looks symmetric around the mean, the
        normality assumption is approximately satisfied.
        """
        deltas = self.group_a - self.group_b
        the_median = np.median(deltas)
        the_mean = np.mean(deltas)

        plt.clf()
        plt.hist(deltas, color="black", alpha=0.5, bins=bins)
        plt.axvline(
            x=the_median, label="median", linestyle="dashdot",
            color="blue", alpha=0.7,
        )
        plt.axvline(
            x=the_mean, label="mean", linestyle="dashed",
            color="red", alpha=0.7,
        )
        plt.legend()
        plt.xlabel("Delta")
        plt.ylabel("Freq")
        plt.show()

        plt.clf()
        fig, ax = plt.subplots(2, sharex=True, sharey=True)
        ax[0].hist(self.group_a, label="Group A")
        ax[1].hist(self.group_b, label="Group B")
        ax[0].legend()
        ax[1].legend()
        plt.show()

    def run_test(self, test_hypothesis="B>A"):
        """Run the paired t-test and print a summary.

        Args:
            test_hypothesis: Human-readable hypothesis label stored in
                the returned dict.

        Returns:
            dict with keys: test, hypothesis, observed_direction,
            t_statistic, p_value_two_tailed, alpha, significant.
        """
        if self.group_a.shape != self.group_b.shape:
            raise ValueError(
                "group_a and group_b must have the same shape for a paired test"
            )
        if self.group_a.size == 0:
            raise ValueError("group_a and group_b cannot be empty")

        t_stat, p_two = stats.ttest_rel(
            self.group_b, self.group_a, nan_policy="omit"
        )

        deltas = self.group_b - self.group_a
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size == 0:
            direction = "tie"
        else:
            delta = float(np.mean(deltas))
            if np.isclose(delta, 0.0):
                direction = "tie"
            elif delta > 0:
                direction = "group_b > group_a"
            else:
                direction = "group_a > group_b"

        is_significant = bool(np.isfinite(p_two) and p_two < self.alpha)

        print("Paired t-test (two-tailed)")
        print(f"  t-statistic : {t_stat}")
        print(f"  p-value     : {p_two}")
        print(f"  Direction   : {direction}")
        print(f"  Alpha       : {self.alpha}")
        print(f"  Significant : {is_significant}")

        return {
            "test": "paired_t_test",
            "hypothesis": test_hypothesis,
            "observed_direction": direction,
            "t_statistic": t_stat,
            "p_value_two_tailed": p_two,
            "alpha": self.alpha,
            "significant": is_significant,
        }


class WilcoxonRunner:
    """Wilcoxon signed-rank test (non-parametric paired test)."""

    def __init__(self, group_a, group_b, alpha=0.05):
        """
        Args:
            group_a: Array-like of numerical values for group A.
            group_b: Array-like of numerical values for group B.
            alpha: Significance level for hypothesis testing.
        """
        self.group_a = np.array(group_a)
        self.group_b = np.array(group_b)
        self.alpha = alpha

    def check_assumptions(self, bins=None):
        """Plot the delta distribution between groups."""
        deltas = self.group_a - self.group_b
        the_median = np.median(deltas)
        the_mean = np.mean(deltas)

        plt.clf()
        plt.hist(deltas, color="black", alpha=0.5, bins=bins)
        plt.axvline(
            x=the_median, label="median", linestyle="dashdot",
            color="blue", alpha=0.7,
        )
        plt.axvline(
            x=the_mean, label="mean", linestyle="dashed",
            color="red", alpha=0.7,
        )
        plt.legend()
        plt.xlabel("Delta")
        plt.ylabel("Freq")
        plt.show()

    def run_test(self, test_hypothesis="B>A"):
        """Run the Wilcoxon signed-rank test and print a summary.

        Args:
            test_hypothesis: Human-readable hypothesis label stored in
                the returned dict.

        Returns:
            dict with keys: test, hypothesis, observed_direction,
            statistic, p_value_two_tailed, alpha, significant.
        """
        if self.group_a.shape != self.group_b.shape:
            raise ValueError(
                "group_a and group_b must have the same shape for a paired test"
            )
        if self.group_a.size == 0:
            raise ValueError("group_a and group_b cannot be empty")

        try:
            statistic, p_value = stats.wilcoxon(
                self.group_b, self.group_a, alternative="two-sided"
            )
        except ValueError:
            statistic, p_value = 0.0, 1.0

        deltas = self.group_b - self.group_a
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size == 0:
            direction = "tie"
        else:
            delta = float(np.median(deltas))
            if np.isclose(delta, 0.0):
                delta = float(np.mean(deltas))
            if np.isclose(delta, 0.0):
                direction = "tie"
            elif delta > 0:
                direction = "group_b > group_a"
            else:
                direction = "group_a > group_b"

        is_significant = bool(np.isfinite(p_value) and p_value < self.alpha)

        print("Wilcoxon signed-rank test (two-tailed)")
        print(f"  Statistic   : {statistic:.6f}")
        print(f"  p-value     : {p_value:.6g}")
        print(f"  Direction   : {direction}")
        print(f"  Alpha       : {self.alpha}")
        print(f"  Significant : {is_significant}")

        return {
            "test": "wilcoxon_signed_rank",
            "hypothesis": test_hypothesis,
            "observed_direction": direction,
            "statistic": statistic,
            "p_value_two_tailed": p_value,
            "alpha": self.alpha,
            "significant": is_significant,
        }