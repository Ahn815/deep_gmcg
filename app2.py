import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt

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
        
        # Alternating binary masks
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

def load_synthetic_data_original(n_samples, var_names, code_formula):
    """Restored Original Option 1 Logic"""
    local_vars = {'N': n_samples, 'np': np, 'D': D.MultivariateNormal, 'torch': torch}
    try:
        exec(code_formula, globals(), local_vars)
        feature_names = [name.strip() for name in var_names.split(',')]
        
        if 'outcome' not in local_vars:
             st.error("Error: Code must define 'outcome'.")
             return None, None
        
        all_vars = []
        for name in feature_names:
            if name not in local_vars:
                st.error(f"Error: Variable '{name}' not found.")
                return None, None
            all_vars.append(local_vars[name])
        all_vars.append(local_vars['outcome'])
        
        data = np.column_stack(all_vars)
        final_var_names = feature_names + ['outcome']
        
        # Create a DataFrame for easier handling
        df = pd.DataFrame(data, columns=final_var_names)
        return df
    except Exception as e:
        st.error(f"Generation Error: {e}")
        return None

def load_clinical_data(uploaded_file):
    """Option 2: Generic Loader"""
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

# --- SIDEBAR: DATA SOURCE ---
st.sidebar.header("1. Data Source")
data_option = st.sidebar.radio("Select Mode:", ("Option 1: Synthetic Simulation", "Option 2: Clinical Data Upload"))

df = None

if data_option == "Option 1: Synthetic Simulation":
    st.sidebar.subheader("Synthetic Generator (Original)")
    N = st.sidebar.slider("Sample Size (N)", 500, 10000, 5000)
    var_input = st.sidebar.text_input("Features (comma-separated)", "age, sex, dose")
    
    default_code = """
# Define independent variables (NumPy)
age = np.random.randint(40, 80, N)
sex = np.random.binomial(1, 0.5, N).astype(float)
dose = np.random.uniform(0, 100, N)

# Define Outcome (must be named 'outcome')
# Causal relationship with some noise
outcome = (250 - 0.02 * dose ** 2 - 0.2 * age + 8.0 * sex + np.random.normal(0, 5.0, N))
outcome = np.clip(outcome, 50, 300)
"""
    code_formula = st.sidebar.text_area("Generation Code", default_code, height=200)
    
    if st.sidebar.button("Generate Synthetic Data"):
        df = load_synthetic_data_original(N, var_input, code_formula)
        if df is not None:
            st.sidebar.success(f"Generated {N} samples.")
            # Auto-configure for Option 1
            all_cols = df.columns.tolist()
            # Assuming last is outcome, first few are pre-treatment, 'dose' is intervention
            # Simple heuristic for default Option 1
            st.session_state['col_config'] = {
                'pre': ['age', 'sex'],
                'int': ['dose'],
                'out': ['outcome'],
                'all': all_cols
            }
            # Save Tensor
            st.session_state['data_tensor'] = torch.FloatTensor(df.values).to(device)
            st.session_state['model'] = None

elif data_option == "Option 2: Clinical Data Upload":
    st.sidebar.subheader("Clinical Data Loader")
    uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df = load_clinical_data(uploaded_file)
        if df is not None:
            st.sidebar.success("File loaded successfully.")

# --- VARIABLE CONFIGURATION (Unified Logic) ---
# Only show manual config for Option 2, or if Option 1 needs tweaking
if df is not None and data_option == "Option 2: Clinical Data Upload":
    st.divider()
    st.subheader("2. Variable Configuration")
    st.info("Define the causal role of each column.")
    
    all_cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    
    with c1:
        pre_cols = st.multiselect("1. Pre-treatment (Fixed)", all_cols, default=[])
    with c2:
        int_cols = st.multiselect("2. Interventions (Changeable)", all_cols, default=[])
    with c3:
        out_cols = st.multiselect("3. Outcomes (Predicted)", all_cols, default=[])

    if st.button("Confirm Configuration"):
        if not pre_cols or not int_cols or not out_cols:
            st.error("Please assign columns to all categories.")
        else:
            # Reorder for safety: Pre -> Int -> Out
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
            st.success("Configuration Saved.")
            st.dataframe(df[ordered_cols].head(3))

# --- TRAINING ---
if st.session_state['data_tensor'] is not None:
    st.divider()
    
    # Get Config
    config = st.session_state.get('col_config', {})
    if not config:
        st.warning("Please configure variables first.")
    else:
        dim = len(config['all'])
        N_samples = st.session_state['data_tensor'].shape[0]
        
        st.subheader(f"3. Model Training (N={N_samples}, Dim={dim})")
        
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Epochs", 10, 5000, 300)
        with col2:
            lr = st.number_input("Learning Rate", 1e-5, 1e-2, 1e-3, format="%.5f")
        
        if st.button("Train SoloCausal Model"):
            data_tensor = st.session_state['data_tensor']
            
            # Smart complexity adjustment
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
    
    # 1. Select Patient
    max_id = data.shape[0] - 1
    pat_id = st.number_input(f"Select Patient ID (0~{max_id})", 0, max_id, 0)
    
    x_orig = data[pat_id:pat_id+1].to(device)
    orig_np = x_orig[0].cpu().numpy()
    
    # 2. Define Intervention
    # Only allow selecting variables defined as 'Interventions' in config
    intervention_candidates = config['int']
    if not intervention_candidates:
        st.error("No Intervention variables defined in configuration.")
    else:
        intervention_var = st.selectbox("Select Variable to Change", intervention_candidates)
        
        # Find index in the full tensor
        target_idx = var_names.index(intervention_var)
        
        current_val = float(orig_np[target_idx])
        target_val = st.number_input(f"Target Value for '{intervention_var}' (Current: {current_val:.2f})", value=current_val)
        
        steps = st.number_input("Optimization Steps", 100, 10000, 2000, 100)

        if st.button("Run Simulation"):
            # Optimization
            z_init, _ = model(x_orig)
            z = z_init.clone().detach().requires_grad_(True)
            opt = optim.Adam([z], lr=0.01)
            
            # Indices Logic
            # Fix Pre-treatment variables AND other Interventions (that we are not changing)
            # Outcomes are free to change
            
            # Indices of Pre-treatment variables
            pre_indices = [var_names.index(c) for c in config['pre']]
            
            # Indices of Intervention variables EXCEPT the target one
            other_int_indices = [var_names.index(c) for c in config['int'] if c != intervention_var]
            
            # Combine fixed indices
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
            
            # Result
            with torch.no_grad():
                x_cf = model.inverse(z)
                cf_np = x_cf[0].cpu().numpy()
            
            # Visualization
            st.write("### Result Comparison")
            
            # Identify Outcome indices for specific metrics
            out_indices = [var_names.index(c) for c in config['out']]
            
            # Metrics for Outcomes
            cols = st.columns(len(out_indices))
            for i, idx in enumerate(out_indices):
                name = var_names[idx]
                val_orig = orig_np[idx]
                val_cf = cf_np[idx]
                delta = val_cf - val_orig
                with cols[i % len(cols)]: # Wrap if many columns
                    st.metric(f"{name}", f"{val_cf:.2f}", f"{delta:+.2f}")
            
            # Bar Chart
            dim = len(var_names)
            fig, axes = plt.subplots(1, dim, figsize=(dim*3, 4))
            if dim == 1: axes = [axes]
            
            colors = ['skyblue', 'lightcoral']
            labels = ['Orig', 'CF']
            
            # Calculate Global Min/Max for plotting limits from training data
            data_min = st.session_state['data_tensor'].min(dim=0).values.cpu().numpy()
            data_max = st.session_state['data_tensor'].max(dim=0).values.cpu().numpy()

            for i, ax in enumerate(axes):
                vals = [orig_np[i], cf_np[i]]
                bars = ax.bar(labels, vals)
                bars[0].set_color(colors[0])
                bars[1].set_color(colors[1])
                
                # Set Limits
                margin = (data_max[i] - data_min[i]) * 0.1
                if margin == 0: margin = 1.0
                ax.set_ylim(data_min[i] - margin, data_max[i] + margin)
                
                ax.set_title(var_names[i])
                
                # Highlight
                if i == target_idx:
                    ax.set_title(f"{var_names[i]} (Target)", color='blue', fontweight='bold')
                    for s in ax.spines.values(): s.set_edgecolor('blue'); s.set_linewidth(2)
                elif i in out_indices:
                    ax.set_title(f"{var_names[i]} (Outcome)", color='red', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)

elif st.session_state['data_tensor'] is None:
    st.info("👈 Please start by selecting a Data Mode.")
