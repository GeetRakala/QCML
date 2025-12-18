import streamlit as st
import pandas as pd
import os
import yaml
from PIL import Image
import sys
import argparse
import shutil

# Import main reproduction logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import main

st.set_page_config(page_title="QCML Dashboard", layout="wide")
st.title("QCML Experiment Dashboard")

# --- Sidebar: Run Experiment ---
st.sidebar.header("Run New Experiment")

# No form to prevent "Enter" from triggering submission
st.sidebar.subheader("Core Parameters")
dataset_type = st.sidebar.selectbox("Dataset Type", ["sphere", "sphere_other", "cubic", "cubic_other", "campadelli_beta", "campadelli_n"])

# Dataset dimension info logic
dataset_dims_info = {
    "sphere": (3, 2),
    "sphere_other": (9, 6),
    "cubic": (18, 17),
    "cubic_other": (10, 5),
    "campadelli_beta": (40, 10),
    "campadelli_n": (72, 18)
}
if dataset_type in dataset_dims_info:
    e_dim_exp, i_dim_exp = dataset_dims_info[dataset_type]
    st.sidebar.info(f"""
    **Embedded dimension** = {e_dim_exp}  
    **Intrinsic dimension** = {i_dim_exp}
    """)

solver = st.sidebar.selectbox("Solver", ["LBFGS", "analytic", "optax", "pseudo", "jaxopt"])
parametrization = st.sidebar.selectbox("Parametrization", ["upper", "pauli"])

H_dim = st.sidebar.number_input("H_dim (Hilbert Dim)", min_value=1, value=16)
N_points = st.sidebar.number_input("N_points", min_value=10, value=1000)
epochs = st.sidebar.number_input("Epochs", min_value=1, value=100)
seed = st.sidebar.number_input("Seed", value=137)

st.sidebar.subheader("Hyperparameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    learning_rate = st.number_input("Learning Rate", value=0.9, format="%.4f")
    l2_lambda = st.number_input("L2 Lambda", value=1e-4, format="%.1e")
with col2:
    A_init_scale = st.number_input("Init Scale (A)", value=5.0)
    noise_level = st.slider("Noise Level", 0.0, 1.0, 0.0, 0.05)

st.sidebar.subheader("Optax Specific")
col3, col4 = st.sidebar.columns(2)
with col3:
    grad_clip_norm = st.number_input("Grad Clip", value=1.0)
with col4:
    transition_steps = st.number_input("Transition Steps", value=25)
    decay_rate = st.number_input("Decay Rate", value=0.99)

if st.sidebar.button("Run Experiment"):
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
        "A_init_scale": A_init_scale,
        "grad_clip_norm": grad_clip_norm,
        "transition_steps": transition_steps,
        "decay_rate": decay_rate
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
                
                # Robust extraction with type casting
                row = {
                    "Dataset": str(config.get("dataset_type", "unknown")),
                    "Solver": str(config.get("solver", "unknown")),
                    "Param": str(config.get("parametrization", "upper")),
                    "H_dim": int(config.get("H_dim", 0)),
                    "Noise": float(config.get("noise_level", 0.0)),
                    "LR": float(config.get("learning_rate", 0.0)),
                    "L2": float(config.get("l2_lambda", 0.0)),
                    "Scale": float(config.get("A_init_scale", 0.0)),
                    "Epochs": int(config.get("epochs", 0)),
                    "N_points": int(config.get("N_points", 0)),
                    "T_steps": int(config.get("transition_steps", 0)),
                    "Decay": float(config.get("decay_rate", 0.0)),
                    "Path": root
                }
                experiments.append(row)
            except Exception:
                pass
    
    df = pd.DataFrame(experiments)
    if not df.empty:
        # Sort for better readability
        sort_cols = [c for c in ["Dataset", "Solver", "H_dim", "Noise"] if c in df.columns]
        df = df.sort_values(by=sort_cols)
    return df

st.header("Results Browser")

if st.button("Refresh Data"):
    st.rerun()

df_experiments = load_experiments()

if df_experiments.empty:
    st.info("No experiments found in `plots/`. Run an experiment from the sidebar!")
else:
    # Interactive Table with Selection
    st.subheader("Select Experiment")
    
    # Configure grid options
    grid_df = df_experiments.drop(columns=["Path"]).copy()
    
    # Use st.dataframe with selection
    selection = st.dataframe(
        grid_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=300
    )
    
    # Access the selection state correctly
    selected_indices = selection.selection.rows
    
    if selected_indices:
        index = selected_indices[0]
        selected_row = df_experiments.iloc[index]
        exp_path = selected_row["Path"]
        
        st.divider()
        st.subheader(f"Results for: {selected_row['Dataset']} ({selected_row['Solver']})")
        st.caption(f"Path: {exp_path}")
        
        # Display Images
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
            
            qm_path = os.path.join(exp_path, "quantum_metric_spectra.png")
            if os.path.exists(qm_path):
                st.image(Image.open(qm_path), caption="Quantum Metric Spectra")

        # Show raw config
        with st.expander("View Full Config"):
            with open(os.path.join(exp_path, "config.yaml"), "r") as f:
                st.code(f.read(), language="yaml")
        
        st.divider()
        if st.button("Delete This Experiment", type="primary"):
            try:
                shutil.rmtree(exp_path)
                st.success(f"Deleted experiment")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting experiment: {e}")
    else:
        st.info("👆 Select an experiment from the table above to view details.")
