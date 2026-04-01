from StatClasses import TTestRunner
import pandas as pd


if __name__ == "__main__":
    main_dir = "/Users/michaelmurray/Documents/GitHub/RPMChem/analysis/results/semantics_qlora_no_ir.csv"
    
    # define comparison model
    consistent_m_dir = "/Users/michaelmurray/Documents/GitHub/RPMChem/analysis/results/semantics_qlora_ir.csv"
    m_to_use = "1" # 1 means use model 1 from the consistent_m_dir and 2 means use model 2

    df = pd.read_csv(main_dir)
    if consistent_m_dir is None:
        if m_to_use == "1":
            groupA = df['bert_f1_model1'].values
        elif m_to_use == "2":
            groupA = df['bert_f1_model2'].values
        else:
            raise Exception("Please specify m_to_use as either '1' or '2'")
        groupB = df['bert_f1_model2'].values
    else:
        if m_to_use == "1":
            df_m1 = pd.read_csv(consistent_m_dir)[['prompt', 'ground_truth_completion', 'bert_f1_model1']].rename(columns={'bert_f1_model1': 'groupA'})
        elif m_to_use == "2":
            df_m1 = pd.read_csv(consistent_m_dir)[['prompt', 'ground_truth_completion', 'bert_f1_model2']].rename(columns={'bert_f1_model2': 'groupA'})
        df_m2 = df[['prompt', 'ground_truth_completion', 'bert_f1_model2']].rename(columns={'bert_f1_model2': 'groupB'})
        merged = df_m1.merge(df_m2, on=['prompt', 'ground_truth_completion'], how='inner')
        groupA = merged['groupA'].values
        groupB = merged['groupB'].values

    print(groupA.mean(), groupB.mean())


    ttR = TTestRunner(groupA, groupB)
    ttR.check_assumptions()
    ttR.run_test()
