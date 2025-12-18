import streamlit as st
import pandas as pd
import os
import yaml
from PIL import Image
import sys
import argparse

# Import main reproduction logic
# We need to add the current directory to sys.path to import main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import main

st.set_page_config(page_title="QCML Dashboard", layout="wide")

st.title("QCML Experiment Dashboard")

# --- Sidebar: Run Experiment ---
st.sidebar.header("Run New Experiment")

with st.sidebar.form("run_experiment_form"):
    dataset_type = st.selectbox("Dataset Type", ["sphere", "sphere_other", "cubic", "cubic_other", "campadelli_beta", "campadelli_n"])
    solver = st.selectbox("Solver", ["LBFGS", "analytic", "optax", "pseudo", "jaxopt"])
    parametrization = st.selectbox("Parametrization", ["upper", "pauli"])
    
    H_dim = st.number_input("H_dim", min_value=1, value=16)
    N_points = st.number_input("N_points", min_value=10, value=1000)
    epochs = st.number_input("Epochs", min_value=1, value=100)
    
    col1, col2 = st.columns(2)
    learning_rate = st.number_input("Learning Rate", value=0.9, format="%.4f")
    l2_lambda = st.number_input("L2 Lambda", value=1e-4, format="%.1e")
    
    noise_level = st.slider("Noise Level", 0.0, 1.0, 0.0, 0.05)
    seed = st.number_input("Seed", value=137)
    
    submitted = st.form_submit_button("Run Experiment")

    if submitted:
        config = {
            "solver": solver,
            "dataset_type": dataset_type,
            "parametrization": parametrization,
            "N_points": N_points,
            "H_dim": H_dim,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2_lambda": l2_lambda,
            "noise_level": noise_level,
            "seed": seed,
            "output_dir": "plots",
            # Defaults for optax or unused params
            "A_init_scale": 5.0, # You might want to expose this too
            "grad_clip_norm": 1.0,
            "transition_steps": 25,
            "decay_rate": 0.99
        }
        
        with st.spinner(f"Running {solver} on {dataset_type}..."):
            try:
                main.run_experiment(config)
                st.success("Experiment finished! Refreshing results...")
                st.rerun()
            except Exception as e:
                st.error(f"Experiment failed: {e}")
                import traceback
                st.code(traceback.format_exc())

# --- Main Area: View Results ---

def load_experiments(plots_dir="plots"):
    experiments = []
    if not os.path.exists(plots_dir):
        return pd.DataFrame()

    for root, dirs, files in os.walk(plots_dir):
        if "config.yaml" in files:
            config_path = os.path.join(root, "config.yaml")
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                
                # Flatten useful config items
                row = {
                    "Dataset": config.get("dataset_type"),
                    "Solver": config.get("solver"),
                    "H_dim": config.get("H_dim"),
                    "Noise": float(config.get("noise_level", 0.0)),
                    "LR": float(config.get("learning_rate", 0.0)),
                    "L2": float(config.get("l2_lambda", 0.0)),
                    "Path": root
                }
                experiments.append(row)
            except Exception:
                pass
    
    df = pd.DataFrame(experiments)
    if not df.empty:
        # Sort for better readability
        df = df.sort_values(by=["Dataset", "Solver", "H_dim", "Noise"])
    return df

st.header("Results Browser")

if st.button("Refresh Data"):
    st.rerun()

df_experiments = load_experiments()

if df_experiments.empty:
    st.info("No experiments found in `plots/`. Run an experiment from the sidebar!")
else:
    # Interactive Table
    # Use st.dataframe with key to allow selection? actually st.dataframe doesn't support select rows easily in older streamlit versions
    # But new st.dataframe or st.data_editor might. Let's strictly use st.dataframe.
    # A cleaner way is using st.selectbox to pick an experiment from the list ID
    
    # Create a display label
    df_experiments["ID"] = df_experiments.apply(
        lambda x: f"{x['Dataset']} | {x['Solver']} | H{x['H_dim']} | Noise {x['Noise']:.2f} | LR {x['LR']} | L2 {x['L2']:.1e}", axis=1
    )
    
    selected_exp_id = st.selectbox("Select Experiment to View", df_experiments["ID"].unique())
    
    if selected_exp_id:
        selected_row = df_experiments[df_experiments["ID"] == selected_exp_id].iloc[0]
        exp_path = selected_row["Path"]
        
        st.subheader(f"Results for: {selected_exp_id}")
        st.text(f"Path: {exp_path}")
        
        # Display Images
        # We look for standard filenames: point_cloud.png, mean_eigenvalues.png, I_dim_hist.png
        
        col1, col2 = st.columns(2)
        
        with col1:
            pc_path = os.path.join(exp_path, "point_cloud.png")
            if os.path.exists(pc_path):
                st.image(Image.open(pc_path), caption="Point Cloud Embedding")
            else:
                st.warning("point_cloud.png not found")
                
            idim_path = os.path.join(exp_path, "I_dim_hist.png")
            if os.path.exists(idim_path):
                st.image(Image.open(idim_path), caption="Intrinsic Dimension Histogram")
        
        with col2:
            me_path = os.path.join(exp_path, "mean_eigenvalues.png")
            if os.path.exists(me_path):
                st.image(Image.open(me_path), caption="Mean Eigenvalues (Quantum Metric)")
            else:
                st.warning("mean_eigenvalues.png not found")

        # Show raw config
        with st.expander("View Full Config"):
            with open(os.path.join(exp_path, "config.yaml"), "r") as f:
                st.code(f.read(), language="yaml")
