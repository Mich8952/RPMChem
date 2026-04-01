import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

"""
Custom classes for running stat tests
"""

class TTestRunner: #paired student ttest
    def __init__(self, groupA, groupB, alpha=0.05):
        """groupA and groupB should be numerical"""
        self.groupA = np.array(groupA)
        self.groupB = np.array(groupB)

        self.alpha = alpha

    def check_assumptions(self, bins = None):
        deltas = self.groupA - self.groupB
        the_median = np.median(deltas)
        the_mean = np.mean(deltas)
        
        plt.clf()
        plt.hist(deltas,color='black',alpha=0.5, bins = bins)
        plt.axvline(x=the_median,label='median',linestyle='dashdot',color='blue',alpha=0.7)
        plt.axvline(x=the_mean,label='mean',linestyle='dashed',color='red',alpha=0.7)
        plt.legend()
        plt.xlabel("Delta")
        plt.ylabel("Freq")
        plt.show()


        """
        if it looks normal (i.e., symmetric wrt mean then we can proceed with t-test)
        """

        # now check for relatively equal variances (just plot both distros with the shared x axis)
        plt.clf()
        fig, ax = plt.subplots(2,sharex=True,sharey=True)
        ax[0].hist(self.groupA,label='GroupB')
        ax[1].hist(self.groupB,label='GroupA')
        ax[0].legend()
        ax[1].legend()
        plt.show()
    
    def run_test(self, test_hypothesis = "B>A"): 
        if self.groupA.shape != self.groupB.shape:
            raise ValueError("groupA and groupB must have the same shape for a paired test")
        if self.groupA.size == 0:
            raise ValueError("groupA and groupB cannot be empty")

        t_stat, p_two = stats.ttest_rel(self.groupB, self.groupA, nan_policy="omit")
        deltas = self.groupB - self.groupA
        deltas = deltas[np.isfinite(deltas)]
        if deltas.size == 0:
            direction = "tie"
        else:
            delta = float(np.mean(deltas))
            if np.isclose(delta, 0.0):
                direction = "tie"
            elif delta > 0:
                direction = "groupB > groupA"
            else:
                direction = "groupA > groupB"

        is_significant = bool(np.isfinite(p_two) and p_two < self.alpha)
        print("Paired t-test (two-tailed)")
        print(f"t-statistic: {t_stat}")
        print(f"p-value (two-tailed): {p_two}")
        print(f"Observed direction: {direction}")
        print(f"Alpha: {self.alpha}")
        print(f"Statistically significant: {is_significant}")

        return {
            "test": "paired_t_test",
            "hypothesis": test_hypothesis,
            "observed_direction": direction,
            "t_statistic": t_stat,
            "p_value_two_tailed": p_two,
            "alpha": self.alpha,
            "significant": is_significant,
        }


class WilcoxenRunner: # wilcoxon signed rank test
    def __init__(self, groupA, groupB, alpha=0.05):
        self.groupA = np.array(groupA)
        self.groupB = np.array(groupB)
        
        self.alpha = alpha

    def check_assumptions(self,bins = None):
        deltas = self.groupA - self.groupB
        the_median = np.median(deltas)
        the_mean = np.mean(deltas)
        plt.clf()
        plt.hist(deltas,color='black',alpha=0.5,bins=bins)
        plt.axvline(x=the_median,label='median',linestyle='dashdot',color='blue',alpha=0.7)
        plt.axvline(x=the_mean,label='mean',linestyle='dashed',color='red',alpha=0.7)
        plt.legend()
        plt.xlabel("Delta")
        plt.ylabel("Freq")
        plt.show()

    def run_test(self, test_hypothesis = "B>A"):
        if self.groupA.shape != self.groupB.shape:
            raise ValueError("groupA and groupB must have the same shape for a paired test")
        if self.groupA.size == 0:
            raise ValueError("groupA and groupB cannot be empty")

        try:
            statistic, p_value = stats.wilcoxon(self.groupB, self.groupA, alternative="two-sided")
        except ValueError:
            statistic, p_value = 0.0, 1.0

        deltas = self.groupB - self.groupA
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
                direction = "groupB > groupA"
            else:
                direction = "groupA > groupB"

        is_significant = bool(np.isfinite(p_value) and p_value < self.alpha)
        print("Wilcoxon signed-rank test (two-tailed)")
        print(f"Statistic: {statistic:.6f}")
        print(f"p-value (two-tailed): {p_value:.6g}")
        print(f"Observed direction: {direction}")
        print(f"Alpha: {self.alpha}")
        print(f"Statistically significant: {is_significant}")

        return {
            "test": "wilcoxon_signed_rank",
            "hypothesis": test_hypothesis,
            "observed_direction": direction,
            "statistic": statistic,
            "p_value_two_tailed": p_value,
            "alpha": self.alpha,
            "significant": is_significant,
        }
