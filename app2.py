import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SoloCausal AI: Precision ITE Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Device Configuration
device = torch.device("cpu")

# --- 1. RealNVP Model Class (Core Engine) ---
class RealNVP(nn.Module):
    def __init__(self, dim=4, n_layers=4, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.prior = D.MultivariateNormal(torch.zeros(dim, device=device), torch.eye(dim, device=device))
        
        # Alternating binary masks for coupling layers
        mask = torch.zeros(dim, device=device)
        mask[::2] = 1
        masks = [mask] + [1 - mask] * (n_layers - 1)
        self.masks = torch.stack(masks[:n_layers])

        def net():
            return nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, dim)
            )
        
        self.scale_nets = nn.ModuleList([net() for _ in range(n_layers)])
        self.trans_nets = nn.ModuleList([net() for _ in range(n_layers)])

    def forward(self, x):
        log_det = torch.zeros(x.shape[0], device=device)
        z = x
        for i in range(len(self.scale_nets)):
            z_masked = z * self.masks[i]
            s = self.scale_nets[i](z_masked) * (1 - self.masks[i])
            t = self.trans_nets[i](z_masked) * (1 - self.masks[i])
            z = z_masked + (1 - self.masks[i]) * (z * torch.exp(s.clamp(-5, 5)) + t)
            log_det += s.sum(dim=1)
        return z, log_det
    
    def inverse(self, z):
        x = z
        for i in reversed(range(len(self.scale_nets))):
            x_masked = x * self.masks[i]
            s = self.scale_nets[i](x_masked) * (1 - self.masks[i])
            t = self.trans_nets[i](x_masked) * (1 - self.masks[i])
            x = x_masked + (1 - self.masks[i]) * ((x - t) * torch.exp(-s.clamp(-5, 5)))
        return x
    
    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_pz = self.prior.log_prob(z)
        return log_pz + log_det

# --- 2. Data Helper Functions ---

def generate_synthetic_basic(n_samples, var_names, code_formula):
    """Option 1-A: Basic Cross-sectional Data"""
    local_vars = {'N': n_samples, 'np': np, 'D': D.MultivariateNormal, 'torch': torch}
    try:
        exec(code_formula, globals(), local_vars)
        feature_names = [name.strip() for name in var_names.split(',')]
        
        if 'outcome' not in local_vars:
             st.error("Error: Code must define 'outcome'.")
             return None
        
        all_vars = []
        for name in feature_names:
            if name not in local_vars:
                st.error(f"Error: Variable '{name}' not found.")
                return None
            all_vars.append(local_vars[name])
        all_vars.append(local_vars['outcome'])
        
        data = np.column_stack(all_vars)
        final_var_names = feature_names + ['outcome']
        
        return pd.DataFrame(data, columns=final_var_names)
    except Exception as e:
        st.error(f"Generation Error: {e}")
        return None

def generate_clinical_longitudinal(n=500):
    """Option 1-B: 8-Week Longitudinal Data"""
    np.random.seed(42)

    ids = np.random.randint(10000000, 99999999, n)
    ages = np.random.randint(20, 80, n)
    sexes = np.random.randint(0, 2, n)
    doses = np.random.choice([0, 10, 20, 30, 40, 50], n) 

    weeks_data = []

    for i in range(n):
        age = ages[i]
        dose = doses[i]
        
        w1 = 5.0 + (age - 40) * 0.05 + np.random.normal(0, 1.0)
        w1 = np.clip(w1, 2, 9)
        
        patient_trajectory = [round(w1)]
        current_val = w1
        
        natural_worsening = 0.1 + (age / 100.0) * 0.05
        treatment_effect = (dose / 50.0) * 0.6 
        delta = natural_worsening - treatment_effect
        
        for t in range(7): 
            noise = np.random.normal(0, 0.4)
            current_val += delta + noise
            patient_trajectory.append(round(current_val, 2))
            
        weeks_data.append(patient_trajectory)

    cols = ['Week1', 'Week2', 'Week3', 'Week4', 'Week5', 'Week6', 'Week7', 'Week8']
    df_weeks = pd.DataFrame(weeks_data, columns=cols)
    
    df_final = pd.DataFrame({
        'Id': ids,
        'Age': ages,
        'Sex': sexes,
        'Dose': doses
    })
    
    df_final = pd.concat([df_final, df_weeks], axis=1)
    return df_final

def load_clinical_data(uploaded_file):
    """Option 2: File Upload"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        df = df.dropna().select_dtypes(include=[np.number])
        return df
    except Exception as e:
        st.error(f"File Loading Error: {e}")
        return None

# --- 3. Streamlit UI ---

st.title("🧬 SoloCausal AI")
st.markdown("### Precision ITE (Individual Treatment Effect) Platform")

# Session State
if 'model' not in st.session_state: st.session_state['model'] = None
if 'data_tensor' not in st.session_state: st.session_state['data_tensor'] = None
if 'col_config' not in st.session_state: st.session_state['col_config'] = {}
if 'sim_results' not in st.session_state: st.session_state['sim_results'] = None

# --- SIDEBAR: DATA SOURCE ---
st.sidebar.header("1. Data Source")
data_source = st.sidebar.radio("Main Source:", ("Option 1: Synthetic Simulation", "Option 2: Clinical Data Upload"))

df = None
sim_type = None

if data_source == "Option 1: Synthetic Simulation":
    sim_type = st.sidebar.radio("Simulation Type:", ("Type A: Basic (Cross-sectional)", "Type B: Longitudinal (8-Week Trial)"))
    
    if sim_type == "Type A: Basic (Cross-sectional)":
        st.sidebar.markdown("**Type A: Age, Sex, Dose -> Outcome**")
        N = st.sidebar.slider("Sample Size (N)", 500, 10000, 5000)
        var_input = st.sidebar.text_input("Features", "age, sex, dose")
        default_code_a = """
age = np.random.randint(40, 80, N)
sex = np.random.binomial(1, 0.5, N).astype(float)
dose = np.random.uniform(0, 100, N)
outcome = (100 - 0.5 * dose - 0.2 * age + 10.0 * sex + np.random.normal(0, 5.0, N))
"""
        code_formula = st.sidebar.text_area("Code Formula", default_code_a, height=150)
        
        if st.sidebar.button("Generate Type A Data"):
            df = generate_synthetic_basic(N, var_input, code_formula)
            if df is not None:
                st.sidebar.success(f"Generated {N} samples (Type A).")
                st.session_state['col_config'] = {
                    'pre': ['age', 'sex'],
                    'int': ['dose'],
                    'out': ['outcome'],
                    'all': df.columns.tolist()
                }
                st.session_state['data_tensor'] = torch.FloatTensor(df.values).to(device)
                st.session_state['model'] = None
                st.session_state['sim_results'] = None

    elif sim_type == "Type B: Longitudinal (8-Week Trial)":
        st.sidebar.markdown("**Type B: Age, Sex, Dose -> Week1...Week8**")
        N_b = st.sidebar.slider("Sample Size (N)", 500, 10000, 500)
        
        if st.sidebar.button("Generate Type B Data"):
            df = generate_clinical_longitudinal(N_b)
            if df is not None:
                st.sidebar.success(f"Generated {N_b} samples (Type B).")
                
                # Download Button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Generated Data (CSV)",
                    data=csv,
                    file_name="clinical_trial_data_8weeks.csv",
                    mime="text/csv",
                )

                training_cols = [c for c in df.columns if c != 'Id']
                st.session_state['col_config'] = {
                    'pre': ['Age', 'Sex'],
                    'int': ['Dose'],
                    'out': [f'Week{i}' for i in range(1, 9)],
                    'all': training_cols
                }
                st.session_state['data_tensor'] = torch.FloatTensor(df[training_cols].values).to(device)
                st.session_state['model'] = None
                st.session_state['sim_results'] = None
                st.dataframe(df.head(3))

elif data_source == "Option 2: Clinical Data Upload":
    st.sidebar.subheader("Clinical Data Loader")
    uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df = load_clinical_data(uploaded_file)
        if df is not None:
            st.sidebar.success("File loaded successfully.")

# --- VARIABLE CONFIGURATION ---
if df is not None and data_source == "Option 2: Clinical Data Upload":
    st.divider()
    st.subheader("2. Variable Configuration")
    
    all_cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    
    with c1:
        pre_cols = st.multiselect("1. Pre-treatment (Fixed)", all_cols, default=[])
    with c2:
        int_cols = st.multiselect("2. Interventions (Changeable)", all_cols, default=[])
    with c3:
        out_cols = st.multiselect("3. Outcomes (Predicted)", all_cols, default=[])

    if st.button("Confirm Configuration"):
        all_selected = pre_cols + int_cols + out_cols
        if not all_selected:
             st.error("Please select columns.")
        elif len(all_selected) != len(set(all_selected)):
            st.error("🚨 Duplicate columns detected!")
        else:
            ordered_cols = pre_cols + int_cols + out_cols
            ordered_data = df[ordered_cols].values
            
            st.session_state['col_config'] = {
                'pre': pre_cols,
                'int': int_cols,
                'out': out_cols,
                'all': ordered_cols
            }
            st.session_state['data_tensor'] = torch.FloatTensor(ordered_data).to(device)
            st.session_state['model'] = None
            st.session_state['sim_results'] = None
            st.success("Configuration Saved.")
            st.dataframe(df[ordered_cols].head(3))

# --- TRAINING ---
if st.session_state['data_tensor'] is not None:
    st.divider()
    config = st.session_state.get('col_config', {})
    
    if not config:
        st.warning("Variable configuration missing.")
    else:
        dim = len(config['all'])
        N_samples = st.session_state['data_tensor'].shape[0]
        
        st.subheader(f"3. Model Training (N={N_samples}, Dim={dim})")
        
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Epochs", 10, 5000, 50)
        with col2:
            lr = st.number_input("Learning Rate", 1e-5, 1e-2, 1e-3, format="%.5f")
        
        if st.button("Train SoloCausal Model"):
            data_tensor = st.session_state['data_tensor']
            
            # Simple complexity adjustment for small/large data
            if N_samples < 1000:
                n_layers, hidden_dim, wd = 2, 32, 1e-3
            else:
                n_layers, hidden_dim, wd = 4, 64, 0.0
            
            model = RealNVP(dim=dim, n_layers=n_layers, hidden_dim=hidden_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            
            train_loader = DataLoader(TensorDataset(data_tensor), batch_size=min(128, N_samples), shuffle=True)
            
            model.train()
            prog = st.progress(0)
            loss_hist = []
            
            for epoch in range(epochs):
                epoch_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    loss = -model.log_prob(batch[0]).mean()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                loss_hist.append(epoch_loss / len(train_loader))
                
                if epoch % 10 == 0: prog.progress((epoch+1)/epochs)
                
            prog.progress(1.0)
            st.session_state['model'] = model
            st.session_state['sim_results'] = None
            st.success("Model Trained!")
            
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.plot(loss_hist)
            ax.set_title("Training Loss")
            st.pyplot(fig)

# --- INFERENCE ---
if st.session_state['model'] is not None:
    st.divider()
    st.subheader("4. Counterfactual Simulation")
    
    model = st.session_state['model']
    data = st.session_state['data_tensor']
    config = st.session_state['col_config']
    var_names = config['all']
    dim = len(var_names)
    
    # 1. Select Patient
    max_id = data.shape[0] - 1
    pat_id = st.number_input(f"Select Patient ID (0~{max_id})", 0, max_id, 0)
    
    x_orig = data[pat_id:pat_id+1].to(device)
    orig_np = x_orig[0].cpu().numpy()
    
    # 2. Define Intervention
    intervention_candidates = config['int']
    if not intervention_candidates:
        st.error("No Intervention variables defined.")
    else:
        intervention_var = st.selectbox("Select Variable to Change", intervention_candidates)
        
        target_idx = var_names.index(intervention_var)
        current_val = float(orig_np[target_idx])
        target_val = st.number_input(f"Target Value for '{intervention_var}' (Current: {current_val:.2f})", value=current_val)
        
        steps = st.number_input("Optimization Steps", 100, 10000, 5000, 100)

        # 3. Run Simulation Logic
        if st.button("Run Simulation"):
            z_init, _ = model(x_orig)
            z = z_init.clone().detach().requires_grad_(True)
            opt = optim.Adam([z], lr=0.01)
            
            pre_indices = [var_names.index(c) for c in config['pre']]
            other_int_indices = [var_names.index(c) for c in config['int'] if c != intervention_var]
            fixed_indices = pre_indices + other_int_indices
            
            bar = st.progress(0)
            for i in range(steps):
                opt.zero_grad()
                x_pred = model.inverse(z)
                
                loss_fixed = 0
                if fixed_indices:
                    diff = x_pred[:, fixed_indices] - x_orig[:, fixed_indices]
                    loss_fixed = (diff**2).sum() * 1000
                
                loss_target = ((x_pred[:, target_idx] - target_val)**2).sum() * 1000
                loss_reg = 0.5 * (z**2).sum()
                
                loss = loss_fixed + loss_target + loss_reg
                loss.backward()
                opt.step()
                
                if i % 100 == 0: bar.progress((i+1)/steps)
            
            bar.progress(1.0)
            
            with torch.no_grad():
                x_cf = model.inverse(z)
                cf_np = x_cf[0].cpu().numpy()
            
            # Store results in Session State
            st.session_state['sim_results'] = {
                'orig_np': orig_np,
                'cf_np': cf_np,
                'target_idx': target_idx,
                'intervention_var': intervention_var,
                'pat_id': pat_id
            }

        # 4. Visualization & Metrics
        if st.session_state['sim_results'] is not None:
            res = st.session_state['sim_results']
            orig_res = res['orig_np']
            cf_res = res['cf_np']
            idx_tgt = res['target_idx']
            int_var_name = res['intervention_var']
            
            st.write("### Result Comparison")
            
            # Visualization Mode
            viz_mode = st.radio(
                "Visualization Mode:", 
                ["Individual Feature Bars (Default)", "Outcome Trajectory Plot (Longitudinal)"],
                horizontal=True
            )
            
            out_indices = [var_names.index(c) for c in config['out']]
            
            # --- MODE A: Trajectory Plot ---
            if viz_mode == "Outcome Trajectory Plot (Longitudinal)":
                if len(out_indices) < 2:
                    st.warning("⚠️ Trajectory plot requires multiple outcome variables (e.g., Week1-Week8). Showing bar charts instead.")
                else:
                    fig_traj, ax = plt.subplots(figsize=(12, 6))
                    orig_traj = orig_res[out_indices]
                    cf_traj = cf_res[out_indices]
                    weeks = range(1, len(out_indices) + 1)
                    week_labels = [var_names[i] for i in out_indices]
                    
                    # Safe retrieval of variables (prevents KeyError on reload)
                    int_var_name = res.get('intervention_var', 'Intervention')
                    idx_tgt = res.get('target_idx', 0)
                    patient_id_disp = res.get('pat_id', 'Unknown') # Safety fallback
                    
                    # Create labels for Legend
                    val_orig_int = orig_res[idx_tgt]
                    val_cf_int = cf_res[idx_tgt]
                    label_orig = f"       Factual ({int_var_name}={val_orig_int:.1f})"
                    label_cf = f"Counterfactual ({int_var_name}={val_cf_int:.1f})"
                    
                    ax.plot(weeks, orig_traj, 'o-', label=label_orig, color='skyblue', linewidth=3, markersize=8)
                    ax.plot(weeks, cf_traj, 'o--', label=label_cf, color='lightcoral', linewidth=3, markersize=8)
                    
                    ax.set_title(f"Outcome Trajectory: Original vs Counterfactual (Patient {patient_id_disp})", fontsize=16, fontweight='bold')
                    ax.set_xlabel("Time Points", fontsize=12)
                    ax.set_ylabel("Outcome Value", fontsize=12)
                    ax.set_xticks(weeks)
                    ax.set_xticklabels(week_labels, rotation=45)
                    ax.legend(fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.fill_between(weeks, orig_traj, cf_traj, color='gray', alpha=0.1)
                    
                    st.pyplot(fig_traj)

                    # Context Bars (Non-outcome variables)
                    st.write("#### Context & Intervention Variables")
                    other_indices = [i for i in range(dim) if i not in out_indices]
                    
                    if other_indices:
                        cols_per_row = 4
                        num_rows = (len(other_indices) + cols_per_row - 1) // cols_per_row
                        
                        # Handle single row case correctly for subplots
                        if num_rows == 1:
                            fig_ctx, axes_ctx = plt.subplots(1, cols_per_row, figsize=(20, 4))
                            axes_flat = axes_ctx if cols_per_row > 1 else [axes_ctx]
                        else:
                            fig_ctx, axes_ctx = plt.subplots(num_rows, cols_per_row, figsize=(20, 4 * num_rows))
                            axes_flat = axes_ctx.flatten()
                        
                        for k, i in enumerate(other_indices):
                            ax = axes_flat[k]
                            var_name = var_names[i]
                            vals = [orig_res[i], cf_res[i]]
                            bars = ax.bar(['Orig', 'CF'], vals)
                            bars[0].set_color('skyblue')
                            bars[1].set_color('lightcoral')
                            
                            # Local Y-Limits
                            scale_vals = [0, orig_res[i], cf_res[i]]
                            ymin, ymax = min(scale_vals), max(scale_vals)
                            span = ymax - ymin
                            if span == 0: span = 1.0
                            margin = span * 0.2
                            ax.set_ylim(ymin - margin, ymax + margin)

                            for bar in bars:
                                h = bar.get_height()
                                offset = margin * 0.05
                                pos = h + offset if h >= 0 else h - offset * 3
                                ax.text(bar.get_x() + bar.get_width()/2, pos, f"{h:.2f}", ha='center', va='bottom', fontsize=10)
                            
                            ax.set_title(var_name, fontsize=11, fontweight='bold')
                            if i == idx_tgt:
                                ax.set_title(f"{var_name} (Target)", color='blue', fontweight='bold')
                                for s in ax.spines.values(): s.set_edgecolor('blue'); s.set_linewidth(2)
                        
                        # Hide unused subplots
                        for k in range(len(other_indices), len(axes_flat)): 
                            axes_flat[k].axis('off')
                            
                        plt.tight_layout()
                        st.pyplot(fig_ctx)

            # --- MODE B: Individual Bars ---
            if viz_mode == "Individual Feature Bars (Default)" or (viz_mode == "Outcome Trajectory Plot (Longitudinal)" and len(out_indices) < 2):
                
                cols = st.columns(len(out_indices))
                for i, idx in enumerate(out_indices):
                    name = var_names[idx]
                    val_cf = cf_res[idx]
                    val_orig = orig_res[idx]
                    delta = val_cf - val_orig
                    with cols[i % len(cols)]: 
                        st.metric(f"{name}", f"{val_cf:.2f}", f"{delta:+.2f}")

                cols_per_row = 4
                num_rows = (dim + cols_per_row - 1) // cols_per_row
                fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(20, 5 * num_rows))
                
                if num_rows == 1 and cols_per_row == 1: axes_flat = [axes]
                elif hasattr(axes, 'flatten'): axes_flat = axes.flatten()
                else: axes_flat = axes
                
                colors = ['skyblue', 'lightcoral']
                labels = ['Orig', 'CF']

                for i in range(len(axes_flat)):
                    ax = axes_flat[i]
                    if i < dim: 
                        var_name = var_names[i]
                        vals = [orig_res[i], cf_res[i]]
                        bars = ax.bar(labels, vals)
                        bars[0].set_color(colors[0])
                        bars[1].set_color(colors[1])
                        ax.tick_params(axis='x', labelsize=19)
                        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
                        
                        # Robust Y-Limits Logic (Local to patient)
                        scale_values = [0, orig_res[i], cf_res[i]]
                        final_min = min(scale_values)
                        final_max = max(scale_values)
                        span = final_max - final_min
                        if span == 0: span = 1.0
                        margin = span * 0.2
                        ax.set_ylim(final_min - margin, final_max + margin)
                        
                        ax.set_title(var_name, fontsize=22)
                        
                        for bar in bars:
                            h = bar.get_height()
                            offset = margin * 0.05
                            pos = h + offset if h >= 0 else h - offset * 3
                            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, pos), ha='center', va='bottom', fontsize=18)
                        
                        if i == idx_tgt:
                            ax.set_title(f"{var_name} (Target)", color='blue', fontsize=22, fontweight='bold')
                            for s in ax.spines.values(): s.set_edgecolor('blue'); s.set_linewidth(2)
                        elif i in out_indices:
                            ax.set_title(f"{var_name} (Outcome)", color='red', fontsize=22, fontweight='bold')
                    else:
                        ax.axis('off')

                plt.tight_layout()
                st.pyplot(fig)

elif st.session_state['data_tensor'] is None:
    st.info("👈 Please start by selecting a Data Source in the Sidebar.")
