import json
import pandas as pd
import glob
import os

out_path = "output/harness/summary_output.txt"

with open(out_path, "w") as out:
    out.write("=== CLINICAL AUTOPSY REPORTS (JSON) ===\n")
    for model in ["zero_padded_ssm", "causal_transformer", "masr_ssm", "masr_mamba"]:
        json_path = f"output/harness/clinical_autopsy_report_{model}.json"
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
                out.write(f"Model: {model}\n")
                out.write(f"  Status: {data.get('status')}\n")
                out.write(f"  Predicted Crash Time: {data.get('predicted_crash_time')}\n")
                out.write(f"  Confidence Score: {data.get('confidence_score')}\n")
                ontology = data.get('anomaly_ontology', {})
                out.write(f"  Primary Latent Driver: {ontology.get('primary_latent_driver')}\n")
                trace = ontology.get('causal_trace', [])
                if len(trace) > 0:
                    out.write(f"  Top Flagged Input: {trace[0].get('flagged_input')} at {trace[0].get('time_step')}\n")
                out.write("\n")

    out.write("=== CLINICAL AUTOPSY METRICS (CSV) ===\n")
    metrics_df = pd.DataFrame()
    for file in glob.glob("output/harness/clinical_autopsy_metrics_*.csv"):
        df = pd.read_csv(file)
        metrics_df = pd.concat([metrics_df, df])
    if not metrics_df.empty:
        out.write(metrics_df.to_string(index=False) + "\n")
    out.write("\n")

    out.write("=== 01 BASELINE INTERPOLATION ===\n")
    df_interp = pd.read_csv("output/harness/01_baseline_interpolation.csv")
    out.write(df_interp.groupby('config/model_type')['mse'].mean().to_string() + "\n")

    out.write("\n=== 02 EXTRAPOLATION TEST ===\n")
    df_extrap = pd.read_csv("output/harness/02_extrapolation_test.csv")
    out.write(df_extrap.groupby('config/model_type')['mse'].mean().to_string() + "\n")

    out.write("\n=== 03 SENSOR DENSITY SWEEP ===\n")
    df_density = pd.read_csv("output/harness/03_sensor_density_sweep.csv")
    out.write(df_density.groupby(['config/model_type', 'config/density'])['mse'].mean().to_string() + "\n")

print(f"Summary written to {out_path}")
