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
        
        # Alternating binary masks for coupling layers
        mask = torch.zeros(dim, device=device)
        mask[::2] = 1
        masks = [mask] + [1 - mask] * (n_layers - 1)
        self.masks = torch.stack(masks[:n_layers])

        # Network builder
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

# --- 2. Data Processing Functions ---

def load_synthetic_data(n_samples, var_names, code_formula):
    """Option 1: Generate Synthetic Data based on user formula"""
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
        return data, final_var_names
    except Exception as e:
        st.error(f"Generation Error: {e}")
        return None, None

def load_clinical_data(uploaded_file):
    """Option 2: Load and preprocess Clinical Trial Data"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Simple preprocessing: Drop NaNs
        df = df.dropna()
        
        # Only keep numeric columns (RealNVP works on continuous data)
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st.error("Error: No numeric columns found in the uploaded file.")
            return None
            
        return numeric_df
    except Exception as e:
        st.error(f"File Loading Error: {e}")
        return None

# --- 3. Streamlit UI Layout ---

st.title("🧬 SoloCausal AI")
st.markdown("### Precision ITE (Individual Treatment Effect) Estimation Platform")
st.markdown("""
**SoloCausal AI** goes beyond average clinical results to identify **individual causal effects**.
By simulating counterfactuals ("What if?"), we help optimize treatment strategies for every single patient.
""")

# Initialize Session State
if 'model' not in st.session_state: st.session_state['model'] = None
if 'data_tensor' not in st.session_state: st.session_state['data_tensor'] = None
if 'var_names' not in st.session_state: st.session_state['var_names'] = []

# --- SIDEBAR: DATA LOADING ---
st.sidebar.header("1. Data Source")
data_option = st.sidebar.radio(
    "Select Data Mode:",
    ("Option 1: Synthetic Simulation", "Option 2: Clinical Data Upload")
)

# Global variables to hold loaded data
data_array = None
var_names = []

if data_option == "Option 1: Synthetic Simulation":
    st.sidebar.subheader("Synthetic Generator")
    N = st.sidebar.slider("Sample Size (N)", 500, 10000, 5000)
    var_input = st.sidebar.text_input("Features (comma-separated)", "age, sex, dose")
    
    default_code = """
# Simulate an 8-week clinical trial
age = np.random.randint(40, 80, N)
sex = np.random.binomial(1, 0.5, N).astype(float)
# Accumulated dose over 8 weeks (0 to 800mg)
dose = np.random.uniform(0, 800, N) 

# Outcome: Reduction in symptoms (Higher is better)
# Non-linear effect of dose, influenced by age and sex
outcome = (100 
           + 0.2 * dose 
           - 0.0001 * (dose - 400)**2 
           - 0.5 * age 
           + 10.0 * sex 
           + np.random.normal(0, 10.0, N))
outcome = np.clip(outcome, 0, 200)
"""
    code_formula = st.sidebar.text_area("Generation Code (NumPy)", default_code, height=200)
    
    if st.sidebar.button("Generate Synthetic Data"):
        data_array, var_names = load_synthetic_data(N, var_input, code_formula)
        if data_array is not None:
            st.session_state['data_tensor'] = torch.FloatTensor(data_array).to(device)
            st.session_state['var_names'] = var_names
            st.session_state['model'] = None # Reset model
            st.success(f"Generated {N} samples. Variables: {var_names}")

elif data_option == "Option 2: Clinical Data Upload":
    st.sidebar.subheader("Clinical Data Loader")
    uploaded_file = st.sidebar.file_uploader("Upload Trial Data (CSV/Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df = load_clinical_data(uploaded_file)
        if df is not None:
            st.sidebar.write("### Select Columns")
            all_cols = df.columns.tolist()
            
            # User defines Features (X) and Outcome (Y)
            feature_cols = st.sidebar.multiselect("Features (e.g., Demographics, Dose)", all_cols, default=all_cols[:-1])
            outcome_col = st.sidebar.selectbox("Primary Outcome (Y)", all_cols, index=len(all_cols)-1)
            
            if st.sidebar.button("Load & Process Data"):
                if not feature_cols:
                    st.sidebar.error("Please select at least one feature.")
                else:
                    selected_cols = feature_cols + [outcome_col]
                    final_data = df[selected_cols].values
                    
                    st.session_state['data_tensor'] = torch.FloatTensor(final_data).to(device)
                    st.session_state['var_names'] = selected_cols
                    st.session_state['model'] = None # Reset model
                    st.success(f"Loaded {len(final_data)} patients. Dim: {len(selected_cols)}")
                    st.dataframe(df[selected_cols].head(3))

# --- MAIN AREA: TRAINING ---

if st.session_state['data_tensor'] is not None:
    data_tensor = st.session_state['data_tensor']
    var_names = st.session_state['var_names']
    dim = data_tensor.shape[1]
    N_samples = data_tensor.shape[0]
    
    st.divider()
    st.subheader(f"2. Model Training (N={N_samples}, Features={dim})")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.number_input("Training Epochs", 10, 5000, 300)
    with col2:
        lr = st.number_input("Learning Rate", 1e-5, 1e-2, 1e-3, format="%.5f")
    with col3:
        # Small Data Strategy
        complexity = st.selectbox("Model Complexity Strategy", ["Standard (Big Data)", "Lite (Small Data / Anti-Overfitting)"])
        
    if complexity == "Standard (Big Data)":
        n_layers, hidden_dim, weight_decay = 4, 64, 0.0
    else:
        # Lighter model with Regularization for small N
        n_layers, hidden_dim, weight_decay = 2, 32, 1e-3 

    if st.button("Initialize & Train SoloCausal Model"):
        # Train/Val Split
        train_size = int(0.8 * N_samples)
        if train_size < 1: train_size = 1 # Safety check
        
        train_data, val_data = random_split(data_tensor, [train_size, N_samples - train_size])
        
        # Batch size adjustment
        batch_size = min(128, train_size)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        # Init Model
        model = RealNVP(dim=dim, n_layers=n_layers, hidden_dim=hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # L2 Regularization
        
        # Training
        progress_bar = st.progress(0)
        status = st.empty()
        loss_hist = []
        
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
            loss_hist.append(epoch_loss)
            
            if epoch % 10 == 0:
                progress_bar.progress(epoch / epochs)
                status.text(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f}")
        
        progress_bar.progress(1.0)
        st.session_state['model'] = model
        st.success("Training Complete: Model is ready for ITE estimation.")
        
        # Loss Plot
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(loss_hist)
        ax.set_title("Training Loss (NLL)")
        ax.set_xlabel("Epoch")
        st.pyplot(fig)

# --- MAIN AREA: COUNTERFACTUAL INFERENCE ---

if st.session_state['model'] is not None:
    st.divider()
    st.subheader("3. Individual Treatment Effect (ITE) Analysis")
    st.info("Select a patient and simulate: 'How would their outcome change if we modified the treatment?'")
    
    var_names = st.session_state['var_names']
    data_tensor = st.session_state['data_tensor']
    model = st.session_state['model']
    dim = len(var_names)
    
    # Feature/Outcome identification
    feature_names = var_names[:-1] 
    outcome_name = var_names[-1]   
    
    # Global Min/Max for robust plotting
    data_min = data_tensor.min(dim=0).values.cpu().numpy()
    data_max = data_tensor.max(dim=0).values.cpu().numpy()
    
    # UI: Patient & Intervention Selection
    col1, col2 = st.columns(2)
    with col1:
        max_id = data_tensor.shape[0] - 1
        patient_idx = st.number_input(f"Select Patient ID (0 ~ {max_id})", 0, max_id, 0)
    with col2:
        steps = st.number_input("Optimization Steps", 100, 10000, 2000, 100)
    
    intervention_var = st.selectbox("Select Intervention Variable (e.g., Dose, Treatment Arm)", feature_names)
    
    # Indices
    target_idx = var_names.index(intervention_var)
    outcome_idx = dim - 1
    
    # Target Value Selection
    current_val = float(data_tensor[patient_idx, target_idx])
    target_val = st.number_input(
        f"Set Counterfactual Value for '{intervention_var}' (Current: {current_val:.2f})", 
        value=current_val
    )

    if st.button("Run SoloCausal Simulation"):
        x_orig = data_tensor[patient_idx:patient_idx+1].to(device)
        orig_vals = x_orig[0].cpu().numpy()
        
        # 1. Latent Space Search (Inversion)
        model.eval()
        with torch.no_grad():
            z_init, _ = model.forward(x_orig)
        
        z = z_init.clone().detach().requires_grad_(True)
        # Higher LR for faster convergence during inference
        opt_cf = optim.Adam([z], lr=0.01) 
        
        # Variables to fix (All except Target and Outcome)
        fixed_indices = [i for i in range(dim) if i != target_idx and i != outcome_idx]
        
        # 2. Optimization Loop
        cf_bar = st.progress(0)
        for step in range(steps):
            opt_cf.zero_grad()
            x_pred = model.inverse(z)
            
            # Loss: Fixed variables should stay same
            loss_fixed = 0
            if fixed_indices:
                diff = x_pred[:, fixed_indices] - x_orig[:, fixed_indices]
                loss_fixed = (diff ** 2).sum() * 1e3 
            
            # Loss: Target variable should match input
            loss_target = ((x_pred[:, target_idx] - target_val) ** 2) * 1e3
            
            # Loss: Regularization (stay in probable latent space)
            loss_reg = 0.5 * (z ** 2).sum()
            
            loss = loss_fixed + loss_target + loss_reg
            loss.backward()
            opt_cf.step()
            
            if step % 200 == 0: cf_bar.progress((step+1)/steps)
        
        cf_bar.progress(1.0)
        
        # 3. Final Prediction
        with torch.no_grad():
            x_cf = model.inverse(z)
            cf_vals = x_cf[0].cpu().numpy()
        
        # ITE Calculation
        factual_y = orig_vals[outcome_idx]
        cf_y = cf_vals[outcome_idx]
        ite = cf_y - factual_y
        
        # 4. Display Results
        st.write("### Simulation Results")
        
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Factual {outcome_name}", f"{factual_y:.2f}")
        m2.metric(f"Counterfactual {outcome_name}", f"{cf_y:.2f}")
        m3.metric("Individual Treatment Effect (ITE)", f"{ite:+.2f}", 
                  delta_color="normal" if ite == 0 else "inverse")
        
        # 5. Visualization (Subplots with fixed axis limits)
        fig, axes = plt.subplots(1, dim, figsize=(dim*3, 4))
        if dim == 1: axes = [axes]
        
        # Safe colors (standard names)
        colors = ['skyblue', 'lightcoral']
        labels = ['Factual', 'CF']
        
        for i, ax in enumerate(axes):
            vals = [orig_vals[i], cf_vals[i]]
            bars = ax.bar(labels, vals)
            
            # Manually set colors
            bars[0].set_color(colors[0])
            bars[1].set_color(colors[1])
            
            # Set Y-axis limits based on Global Min/Max of the dataset
            # Add 10% margin for better visuals
            margin = (data_max[i] - data_min[i]) * 0.1
            if margin == 0: margin = 1.0 # Prevent flat limit
            ax.set_ylim(data_min[i] - margin, data_max[i] + margin)
            
            ax.set_title(var_names[i])
            
            # Annotate values
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.2f}", ha='center', va='bottom', fontsize=9)
            
            # Highlighting
            if i == target_idx:
                ax.set_title(f"{var_names[i]} (Intervention)", color='blue', fontweight='bold')
                for s in ax.spines.values(): s.set_edgecolor('blue'); s.set_linewidth(2)
            elif i == outcome_idx:
                ax.set_title(f"{var_names[i]} (Outcome)", color='red', fontweight='bold')
                
        plt.tight_layout()
        st.pyplot(fig)

elif st.session_state['data_tensor'] is None:
    st.info("👈 Please start by selecting a Data Option in the Sidebar.")
