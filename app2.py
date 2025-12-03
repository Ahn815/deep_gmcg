import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Dynamic RealNVP Causal Inference Simulator", layout="wide")

# Device Configuration
device = torch.device("cpu")

# --- 1. RealNVP Model Class Definition ---

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

# --- 2. Dynamic Data Generation Function ---

def generate_dynamic_data(n_samples, var_names, code_formula):
    # Prepare environment for code execution
    local_vars = {'N': n_samples, 'np': np, 'D': D.MultivariateNormal, 'torch': torch}
    
    try:
        # Execute user-provided code
        # WARNING: This uses exec() and can run arbitrary code. Use with caution.
        exec(code_formula, globals(), local_vars)
        
        # Retrieve generated variables in order
        feature_names = [name.strip() for name in var_names.split(',')]
        
        if 'outcome' not in local_vars:
             st.error("Error: The code must define an 'outcome' variable.")
             return None, 0

        # Collect all generated features and the outcome
        all_vars = []
        for name in feature_names:
            if name in local_vars:
                all_vars.append(local_vars[name])
            else:
                st.error(f"Error: Feature variable '{name}' not found after executing the code.")
                return None, 0

        all_vars.append(local_vars['outcome'])
        
        # Final Tensor Creation
        data = np.column_stack(all_vars)
        data_tensor = torch.FloatTensor(data).to(device)
        
        # Calculate Dimension
        dim = data.shape[1] 

        # Store variable names and indices for CF inference
        st.session_state['var_names'] = feature_names + ['outcome']
        st.session_state['data_col_indices'] = {name: i for i, name in enumerate(st.session_state['var_names'])}
        
        return data_tensor, dim

    except Exception as e:
        st.error(f"Data Generation Error: {e}")
        return None, 0

# --- 3. Streamlit UI Start ---

st.title("💊 Dynamic RealNVP Causal Inference Simulator")
st.markdown("""
This app allows you to **dynamically define the variables and causal formula** for the simulation.
""")

# Initialize Session State
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'data_tensor' not in st.session_state:
    st.session_state['data_tensor'] = None

# --- Sidebar Configuration ---
st.sidebar.header("1. Data & Model Setup")
N = st.sidebar.slider("Number of Samples (N)", 1000, 20000, 5000, step=1000)
epochs = st.sidebar.number_input("Training Epochs", value=100, min_value=10)
lr = st.sidebar.number_input("Learning Rate", value=1e-4, format="%.5f")

st.sidebar.markdown("### Dynamic Variable Definition")
var_input = st.sidebar.text_input("Feature Variables (comma-separated)", "age, sex, dose")

default_code = """
# Define independent variables here (must use NumPy and N)
age = np.random.randint(40, 80, N)
sex = np.random.binomial(1, 0.5, N).astype(float)
dose = np.random.uniform(0, 100, N)

# Define Outcome (must be named 'outcome')
outcome = (250 - 0.02 * dose ** 2 - 0.2 * age + 8.0 * sex + np.random.normal(0, 0.1, N))
outcome = np.clip(outcome, 50, 300)
"""
code_formula = st.sidebar.text_area("Causal Formula Code (NumPy)", default_code, height=300)
st.sidebar.warning("Security Note: Code executed via 'exec' can be unsafe.")

# Data Generation Button
if st.sidebar.button("Generate Data & Initialize Model"):
    data_tensor, dim = generate_dynamic_data(N, var_input, code_formula)
    
    if data_tensor is not None:
        st.session_state['data_tensor'] = data_tensor
        st.session_state['dim'] = dim # Store DIM
        st.session_state['model'] = None
        st.sidebar.success(f"{N} data samples created. Dimension (D): {dim}")
    else:
        st.sidebar.error("Data generation failed. Check your code.")

# --- Main Area ---

# 2. Model Training (Uses dynamic DIM)
if st.session_state.get('data_tensor') is not None:
    dim = st.session_state['dim']
    st.subheader(f"2. Model Training (Dimension: D={dim})")
    
    if st.button("Start Model Training"):
        
        # Data loading setup
        train_size = int(0.8 * N)
        train_data, val_data = random_split(st.session_state['data_tensor'], [train_size, N - train_size])
        train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
        
        # Model initialization uses dynamic DIM
        model = RealNVP(dim=dim, n_layers=4, hidden_dim=12).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_history = []

        model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                loss = -model.log_prob(batch).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            
            epoch_loss /= len(train_loader.dataset)
            loss_history.append(epoch_loss)
            
            if epoch % 10 == 0 or epoch == 1:
                progress_bar.progress(epoch / epochs)
                status_text.text(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f}")
                
        st.session_state['model'] = model 
        st.success("Training Complete!")
        
        # Plot Loss Curve
        fig_loss, ax_loss = plt.subplots(figsize=(10, 3))
        ax_loss.plot(loss_history, label='Train Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Negative Log Likelihood')
        ax_loss.legend()
        st.pyplot(fig_loss)

# 3. Counterfactual Inference (Uses dynamic variable names and indices)
if st.session_state.get('model') is not None:
    st.divider()
    st.subheader("3. Counterfactual Inference")
    st.markdown("Simulate **'What would have happened if the value of one variable was changed?'**")

    # --- NEW: Optimization Steps Input ---
    steps = st.number_input(
        "Optimization Steps (Higher = More Accurate, Slower)", 
        min_value=100, 
        max_value=20000, 
        value=7000, 
        step=100
    )
    # ------------------------------------

    var_names = st.session_state['var_names']
    dim = st.session_state['dim']
    
    # Feature names (all except the last one, which is 'outcome')
    feature_names = var_names[:-1] 
    
    # Dynamic selection of the intervention variable
    intervention_var_name = st.selectbox(
        "Select Intervention Variable (Which variable to change?)",
        options=feature_names,
        index=feature_names.index('dose') if 'dose' in feature_names else 0
    )
    
    # Get the index of the variable to intervene on
    target_idx = st.session_state['data_col_indices'][intervention_var_name]

    col1, col2 = st.columns(2)
    with col1:
        patient_idx = st.number_input("Select Patient ID (0 ~ N-1)", 0, N-1, 100)
    with col2:
        # Simple max range for input
        target_val_input = st.number_input(f"Target Value for '{intervention_var_name}'", 0.0, 100.0, 90.0)

    if st.button("Generate Counterfactual Result"):
        model = st.session_state['model']
        data_tensor = st.session_state['data_tensor']
        
        x_orig = data_tensor[patient_idx:patient_idx+1].to(device)
        orig_vals = x_orig[0].cpu().numpy()
        
        # Find Latent z
        model.eval()
        with torch.no_grad():
            z_init, _ = model.forward(x_orig)
        
        z = z_init.clone().detach().requires_grad_(True)
        opt_cf = optim.Adam([z], lr=2e-3)

        cf_progress = st.progress(0)
        
        # Determine the outcome index (always the last one)
        outcome_idx = dim - 1
        
        # ⭐ FIXED INDICES: All indices EXCEPT the intervention variable AND the outcome variable
        fixed_indices = [i for i in range(dim) if i != target_idx and i != outcome_idx]
        
        # Counterfactual Optimization Loop
        for step in range(steps):
            opt_cf.zero_grad()
            x_pred = model.inverse(z)
            
            # 1. Loss for FIXED variables (Must remain close to original factual data)
            loss_fixed_components = []
            if fixed_indices:
                for i in fixed_indices:
                    loss_fixed_components.append((x_pred[:, i] - x_orig[:, i]) ** 2)
                
                loss_fixed = torch.stack(loss_fixed_components, dim=1).sum(dim=1) * 1e6 # High penalty
            else: 
                loss_fixed = 0
            
            # 2. Loss for TARGET variable (Must match the intervened value)
            loss_target = (x_pred[:, target_idx] - target_val_input) ** 2
            
            # 3. Regularization Loss (Keeps z close to the origin/prior mean)
            loss_reg = 0.5 * z.square().sum()
            
            # Total Loss
            loss = 1e5 * (loss_fixed + loss_target) + loss_reg
            
            loss.backward()
            opt_cf.step()
            
            if step % 100 == 0:
                cf_progress.progress((step + 1) / steps)
        
        cf_progress.progress(1.0)
        
        # Final Results
        with torch.no_grad():
            x_cf = model.inverse(z)
            cf_vals = x_cf[0].cpu().numpy()
            
        st.write("### Result Comparison")
        
        # Dynamic Metric Display
        cols = st.columns(dim)
        for i, name in enumerate(var_names):
            change = cf_vals[i] - orig_vals[i]
            delta_color = "off"
            
            if i == target_idx:
                delta_color = "normal"
            elif name == 'outcome':
                delta_color = "inverse"
            
            cols[i].metric(
                f"{name.capitalize()} (Orig)", 
                f"{orig_vals[i]:.2f}", 
                delta=f"{change:.2f}" if abs(change) > 1e-4 else "0.00",
                delta_color=delta_color
            )
        
        # Plot Graph (Simplified Bar Chart)
        fig, ax = plt.subplots(figsize=(12, 5))
        
        x_pos = np.arange(dim)
        width = 0.35
        
        ax.bar(x_pos - width/2, orig_vals, width, label='Original (Factual)', color='skyblue')
        ax.bar(x_pos + width/2, cf_vals, width, label='Counterfactual', color='lightcoral')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([name.capitalize() for name in var_names])
        ax.set_ylabel("Values")
        ax.set_title(f"Counterfactual: Changing {intervention_var_name} to {target_val_input:.2f} (Steps: {steps})")
        ax.legend()
        st.pyplot(fig)

elif st.session_state.get('data_tensor') is None:
    st.info("👈 Please define variables, enter the causal formula, and click 'Generate Data' first.")
