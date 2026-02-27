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
        """
        if the test_hypothesis is B>A then its a one tailed test with the alternative hypothesis that groupB is larger than groupA
        elif its the other alt. hypothesis
        if its blank then dont implement yet (idk if i care about a two-tailed test yet)
        """

        hypothesis = test_hypothesis.strip().upper() 
        if hypothesis == "B>A":
            t_stat, p_two = stats.ttest_rel(self.groupB, self.groupA, nan_policy="omit") # we handle nans prior to this call anyways
            p_one = p_two / 2 if t_stat > 0 else 1 - (p_two / 2) # depending on the direction we need to adjust the p value to grab the right tail
            direction = "groupB > groupA"
        elif hypothesis == "A>B":
            t_stat, p_two = stats.ttest_rel(self.groupA, self.groupB, nan_policy="omit")
            p_one = p_two / 2 if t_stat > 0 else 1 - (p_two / 2)
            direction = "groupA > groupB"
        else:
            raise ValueError("test_hypothesis must be one of: 'B>A', 'A>B'")

        is_significant = bool(p_one < self.alpha)
        print(f"Paired t-test ({direction})")
        print(f"t-statistic: {t_stat}")
        print(f"p-value (one-tailed): {p_one}")
        print(f"Alpha: {self.alpha}")
        print(f"Statistically significant: {is_significant}")

        return {
            "test": "paired_t_test",
            "hypothesis": test_hypothesis,
            "t_statistic": t_stat,
            "p_value_one_tailed": p_one,
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

        hypothesis = test_hypothesis.strip().upper()
        if hypothesis == "B>A":
            statistic, p_value = stats.wilcoxon(self.groupB, self.groupA, alternative="greater")
            direction = "groupB > groupA"
        elif hypothesis == "A>B":
            statistic, p_value = stats.wilcoxon(self.groupA, self.groupB, alternative="greater")
            direction = "groupA > groupB"
        else:
            raise ValueError("test_hypothesis must be one of: 'B>A', 'A>B'")

        is_significant = bool(p_value < self.alpha)
        print(f"Wilcoxon signed-rank test ({direction})")
        print(f"Statistic: {statistic:.6f}") # just 6 digits, but idrc that much about the stat value as we only care about p value really
        print(f"p-value (one-tailed): {p_value:.6g}") # 6 sig figs because values can be very small
        print(f"Alpha: {self.alpha}")
        print(f"Statistically significant: {is_significant}")

        return {
            "test": "wilcoxon_signed_rank",
            "hypothesis": test_hypothesis,
            "statistic": statistic,
            "p_value_one_tailed": p_value,
            "alpha": self.alpha,
            "significant": is_significant,
        }
