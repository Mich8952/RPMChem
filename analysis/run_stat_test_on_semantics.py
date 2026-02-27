from StatClasses import TTestRunner
import pandas as pd


if __name__ == "__main__":
    # first load in the file from results
    df = pd.read_csv("/Users/michaelmurray/Documents/GitHub/RPMChem/analysis/results/semantics_comparison.csv")

    groupA = df['bert_f1_model1'].values
    groupB = df['bert_f1_model2'].values

    print("")

    ttR = TTestRunner(groupA, groupB)
    ttR.check_assumptions()
    ttR.run_test(test_hypothesis="B>A")
