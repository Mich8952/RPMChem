import sys
sys.path.append("/Users/michaelmurray/Documents/GitHub/RPMChem/analysis")

from StatClasses import TTestRunner, WilcoxenRunner
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CAP = 10  # maximum error value; also assigned when model fails to predict

def relative_error(y_true, y_hat):
    return np.abs(y_hat - y_true) / np.abs(y_true)

if __name__ == "__main__":
    df = pd.read_csv("/Users/michaelmurray/Documents/GitHub/RPMChem/analysis/results/numerical_comparison.csv")

    valid = df['ground_truth'].notna()
    df = df[valid].reset_index(drop=True)

    print(f"N samples: {len(df)}")
    print(f"Model1 NaN rate: {df['model1_ans'].isna().sum() / len(df) * 100:.1f}%")
    print(f"Model2 NaN rate: {df['model2_ans'].isna().sum() / len(df) * 100:.1f}%")

    def compute_err(y_true, y_hat):
        if pd.isna(y_hat):
            return CAP #\+ 0.1
        return min(relative_error(y_true, y_hat), CAP)

    err1 = df.apply(lambda r: compute_err(r['ground_truth'], r['model1_ans']), axis=1)
    err2 = df.apply(lambda r: compute_err(r['ground_truth'], r['model2_ans']), axis=1)


    print(f"\Mean relative error model1: {np.mean(err1):.4f}")
    print(f"Mean relative error model2: {np.mean(err2):.4f}")

    print(f"\std relative error model1: {np.std(err1):.4f}")
    print(f"std relative error model2: {np.std(err2):.4f}")
    
    print("")
    wcR = WilcoxenRunner(err1.values, err2.values)
    wcR.check_assumptions(bins=20)
    wcR.run_test(test_hypothesis="A>B")
