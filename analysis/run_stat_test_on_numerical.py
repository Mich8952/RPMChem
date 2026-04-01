import sys
sys.path.append("/Users/michaelmurray/Documents/GitHub/RPMChem/analysis")

from StatClasses import TTestRunner, WilcoxenRunner
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

CAP_glob = 10  # maximum error value; also assigned when model fails to predict

def relative_error(y_true, y_hat):
    return np.abs(y_hat - y_true) / np.abs(y_true)


def compute_err(y_true, y_hat, CAP = CAP_glob):
    if pd.isna(y_hat):
        return CAP #\+ 0.1
    return min(relative_error(y_true, y_hat), CAP)

def penality_analysis(df):
    A = df[["model1_converted_value","ground_truth"]]
    per_pen_A = []
    pens = [2,4,7,8,10,12,14,16,18]
    for pen in pens:
        errsA = []
        for i in range(len(A)):
            errsA.append(compute_err(A.iloc[i]['ground_truth'],A.iloc[i]['model1_converted_value'],CAP=pen))
        per_pen_A.append(np.mean(errsA))

    B = df[["model2_converted_value","ground_truth"]]
    per_pen_B = []
    for pen in pens:
        errsB = []
        for i in range(len(B)):
            errsB.append(compute_err(B.iloc[i]['ground_truth'],B.iloc[i]['model2_converted_value'],CAP=pen))
        per_pen_B.append(np.mean(errsB))

    # inefficient but okay for now (need to check sdtats on each one)
    stat_sig = []
    for pen in pens:
        
        errs_a = []
        errs_b = []
        for i in range(len(A)):
            errA = compute_err(A.iloc[i]['ground_truth'],A.iloc[i]['model1_converted_value'],CAP=pen)
            errB = compute_err(B.iloc[i]['ground_truth'],B.iloc[i]['model2_converted_value'],CAP=pen)
            errs_a.append(errA)
            errs_b.append(errB)
        wcR = WilcoxenRunner(np.array(errs_a), np.array(errs_b))
        stat_sig.append(wcR.run_test())
    
    sig_bool = [x['significant'] for x in stat_sig]

    plt.clf()
    plt.plot(pens,per_pen_A, label = "Model 1")
    plt.plot(pens,per_pen_B, label = "Model 2")
    x_dense = np.linspace(min(pens)-0.5, max(pens)+0.5, 600)
    sig_dense = np.interp(x_dense, pens, np.array(sig_bool, dtype=float))
    rgb = np.stack([1.0 - sig_dense, sig_dense, np.zeros_like(sig_dense)], axis=1)[None, :, :]
    y0, y1 = plt.gca().get_ylim()
    plt.gca().imshow(rgb, extent=[x_dense.min(), x_dense.max(), y0, y1], origin='lower', aspect='auto', alpha=0.18, zorder=0)
    plt.grid()
    
    plt.title("Mean Relative Error vs Penality Cap \n Statistical Significance shown in Green")
    plt.ylabel("Mean Relative Error")
    plt.xlabel("Penality Cap (Value assigned to failed predictions)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main_dir = "/Users/michaelmurray/Documents/GitHub/RPMChem/analysis/results/numerical_qlora_ir.csv"

    consistent_m_dir = "/Users/michaelmurray/Documents/GitHub/RPMChem/analysis/results/numerical_qlora_no_ir.csv"
    m_to_use = "1" # 1 means use model 1 from the consistent_m_dir and 2 means use model 2

    df = pd.read_csv(main_dir)
    if consistent_m_dir is None:
        if m_to_use == "1":
            pass
        elif m_to_use == "2":
            df["model1_converted_value"] = df["model2_converted_value"]
            df["model1_ans"] = df["model2_ans"]
        else:
            raise Exception("Please specify m_to_use as either '1' or '2'")
    else:
        if m_to_use == "1":
            df_m1 = pd.read_csv(consistent_m_dir)[["prompt", "ground_truth", "model1_converted_value", "model1_ans"]]
        elif m_to_use == "2":
            df_m1 = pd.read_csv(consistent_m_dir)[["prompt", "ground_truth", "model2_converted_value", "model2_ans"]].rename(columns={"model2_converted_value":"model1_converted_value","model2_ans":"model1_ans"})
        else:
            raise Exception("Please specify m_to_use as either '1' or '2'")
        df_m2 = df[["prompt", "ground_truth", "model2_converted_value", "model2_ans"]]
        df_m1["ground_truth"] = pd.to_numeric(df_m1["ground_truth"], errors="coerce")
        df_m2["ground_truth"] = pd.to_numeric(df_m2["ground_truth"], errors="coerce")
        df = df_m1.merge(df_m2, on=["prompt", "ground_truth"], how="inner")

    valid = df['ground_truth'].notna()
    df = df[valid].reset_index(drop=True)

    print(f"N samples: {len(df)}")
    print(f"Model1 NaN rate: {df['model1_ans'].isna().sum() / len(df) * 100:.1f}%")
    print(f"Model2 NaN rate: {df['model2_ans'].isna().sum() / len(df) * 100:.1f}%")

    err1 = df.apply(lambda r: compute_err(r['ground_truth'], r['model1_ans']), axis=1)
    err2 = df.apply(lambda r: compute_err(r['ground_truth'], r['model2_ans']), axis=1)


    print(f"\Mean relative error model1: {np.mean(err1):.4f}")
    print(f"Mean relative error model2: {np.mean(err2):.4f}")

    print(f"\std relative error model1: {np.std(err1):.4f}")
    print(f"std relative error model2: {np.std(err2):.4f}")
    
    print("")
    wcR = WilcoxenRunner(err1.values, err2.values)
    wcR.check_assumptions(bins=20)
    wcR.run_test()


    #penality_analysis(df)
