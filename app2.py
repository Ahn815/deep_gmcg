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
    page_title="SoloCausal AI: Longitudinal Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Device Configuration
device = torch.device("cpu")

# --- 1. RealNVP Model Class ---
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

def generate_longitudinal_data(n_samples):
    """Generates 8-week clinical trial data for demo."""
    np.random.seed(42)
    # 1. Pre-treatment (Demographics)
    age = np.random.randint(40, 80, n_samples)
    sex = np.random.binomial(1, 0.5, n_samples)
    baseline_severity = np.random.uniform(50, 80, n_samples) # Initial disease score
    
    # 2. Intervention (Doses Week 1-8)
    # Randomize doses to simulate different arms or adherence
    doses = []
    for w in range(8):
        # Some get 0 (placebo), some get 50, some get 100
        dose_w = np.random.choice([0, 50, 100], n_samples) 
        doses.append(dose_w)
    
    doses = np.stack(doses, axis=1) # (N, 8)
    
    # 3. Outcomes (Score Week 1-8)
    # Logic: Score decreases (improves) with Dose, increases with Age, correlated with previous week
    outcomes = []
    current_score = baseline_severity
    
    for w in range(8):
        dose_effect = 0.1 * doses[:, w]
        natural_recovery = 0.5 
        
        # Next week score = Current - Dose Effect - Recovery + Noise
        next_score = current_score - dose_effect - natural_recovery + np.random.normal(0, 1.0, n_samples)
        
        # Age penalty (Older recover slower)
        next_score += 0.01 * age
        
        next_score = np.clip(next_score, 0, 100)
        outcomes.append(next_score)
        current_score = next_score # Update for next step
        
    outcomes = np.stack(outcomes, axis=1) # (N, 8)

    # Combine into DataFrame
    cols = ['Age', 'Sex', 'Baseline'] + [f'Dose_W{i+1}' for i in range(8)] + [f'Outcome_W{i+1}' for i in range(8)]
    data = np.column_stack([age, sex, baseline_severity, doses, outcomes])
    
    return pd.DataFrame(data, columns=cols)

def load_clinical_data(uploaded_file):
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

st.title("📈 SoloCausal AI: Longitudinal Analysis")
st.markdown("""
**Simulate Multi-Week Clinical Trials.** Define Pre-treatment variables (Fixed), Intervention trajectory (Doses), and observe the counterfactual Outcome trajectory.
""")

# Session State Init
if 'model' not in st.session_state: st.session_state['model'] = None
if 'data_tensor' not in st.session_state: st.session_state['data_tensor'] = None
if 'col_config' not in st.session_state: st.session_state['col_config'] = {}

# --- SIDEBAR: DATA ---
st.sidebar.header("1. Data Source")
data_option = st.sidebar.radio("Mode:", ("Option 1: 8-Week Trial Demo", "Option 2: Upload Data"))

df = None

if data_option == "Option 1: 8-Week Trial Demo":
    N = st.sidebar.slider("Sample Size", 500, 5000, 1000)
    if st.sidebar.button("Generate Demo Data"):
        df = generate_longitudinal_data(N)
        st.sidebar.success(f"Generated {N} patients (8 Weeks).")

elif data_option == "Option 2: Upload Data":
    uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
    if uploaded_file:
        df = load_clinical_data(uploaded_file)
        if df is not None:
            st.sidebar.success("File loaded successfully.")

# --- COLUMN CONFIGURATION (CRITICAL STEP) ---
if df is not None:
    st.divider()
    st.subheader("2. Variable Configuration (Crucial Step)")
    st.info("Assign columns to their specific roles to define the Causal Structure.")
    
    all_cols = df.columns.tolist()
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**1. Pre-treatment (Fixed)**")
        st.caption("Baseline, Demographics (Age, Sex)")
        pre_cols = st.multiselect("Select Pre-treatment", all_cols, default=all_cols[:3] if data_option=="Option 1: 8-Week Trial Demo" else [])
        
    with c2:
        st.markdown("**2. Interventions (Changeable)**")
        st.caption("Doses, Treatment Arms (Week 1-8)")
        # Auto-detect 'Dose' columns for demo
        def_int = [c for c in all_cols if 'Dose' in c] if data_option=="Option 1: 8-Week Trial Demo" else []
        int_cols = st.multiselect("Select Interventions", all_cols, default=def_int)
        
    with c3:
        st.markdown("**3. Outcomes (Predicted)**")
        st.caption("Endpoints, Biomarkers (Week 1-8)")
        # Auto-detect 'Outcome' columns for demo
        def_out = [c for c in all_cols if 'Outcome' in c] if data_option=="Option 1: 8-Week Trial Demo" else []
        out_cols = st.multiselect("Select Outcomes", all_cols, default=def_out)

    if st.button("Confirm & Process Data"):
        if not pre_cols or not int_cols or not out_cols:
            st.error("Please select at least one column for each category.")
        else:
            # Reorder DataFrame: [Pre | Int | Out] for easier tensor indexing
            ordered_cols = pre_cols + int_cols + out_cols
            ordered_data = df[ordered_cols].values
            
            # Save config
            st.session_state['data_tensor'] = torch.FloatTensor(ordered_data).to(device)
            st.session_state['col_config'] = {
                'pre': pre_cols,
                'int': int_cols,
                'out': out_cols,
                'all': ordered_cols,
                # Store index ranges
                'idx_pre': list(range(0, len(pre_cols))),
                'idx_int': list(range(len(pre_cols), len(pre_cols)+len(int_cols))),
                'idx_out': list(range(len(pre_cols)+len(int_cols), len(ordered_cols)))
            }
            st.session_state['model'] = None
            st.success(f"Data Prepared. Dimension: {len(ordered_cols)}")
            st.dataframe(df[ordered_cols].head(3))

# --- TRAINING ---
if st.session_state['data_tensor'] is not None:
    st.divider()
    config = st.session_state['col_config']
    dim = len(config['all'])
    
    st.subheader(f"3. Model Training (Dim={dim})")
    
    if st.button("Train SoloCausal Model"):
        data_tensor = st.session_state['data_tensor']
        
        # Simple training config
        model = RealNVP(dim=dim, n_layers=4, hidden_dim=64).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        train_loader = DataLoader(TensorDataset(data_tensor), batch_size=64, shuffle=True)
        
        model.train()
        prog = st.progress(0)
        
        epochs = 200
        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                loss = -model.log_prob(batch[0]).mean()
                loss.backward()
                optimizer.step()
            if epoch % 20 == 0: prog.progress((epoch+1)/epochs)
            
        prog.progress(1.0)
        st.session_state['model'] = model
        st.success("Model Trained!")

# --- INFERENCE ---
if st.session_state['model'] is not None:
    st.divider()
    st.subheader("4. Counterfactual Simulation (Trajectory Analysis)")
    
    model = st.session_state['model']
    data = st.session_state['data_tensor']
    config = st.session_state['col_config']
    
    # 1. Select Patient
    pat_id = st.number_input("Select Patient ID", 0, data.shape[0]-1, 0)
    
    # Get Patient Data
    x_orig = data[pat_id:pat_id+1].to(device)
    orig_np = x_orig[0].cpu().numpy()
    
    # 2. Define Intervention Scenario
    st.markdown("### Set Counterfactual Scenario")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Intervention Plan:**")
        # Display current doses as a small chart
        orig_doses = orig_np[config['idx_int']]
        st.bar_chart(pd.DataFrame(orig_doses, index=config['int'], columns=["Original Dose"]))
        
    with col2:
        st.markdown("**New Intervention Target:**")
        target_mode = st.radio("Target Strategy:", ["Constant Dose (All Weeks)", "Percentage Change"])
        
        if target_mode == "Constant Dose (All Weeks)":
            target_val = st.number_input("Set Dose for ALL weeks:", value=float(np.mean(orig_doses)))
            target_vector = np.full_like(orig_doses, target_val)
        else:
            pct = st.slider("Change Dose by %:", -100, 100, 0)
            target_vector = orig_doses * (1 + pct/100)
            
        st.caption(f"Targeting: {target_vector}")

    if st.button("Run Simulation"):
        # Optimization
        z_init, _ = model(x_orig)
        z = z_init.clone().detach().requires_grad_(True)
        opt = optim.Adam([z], lr=0.01)
        
        steps = 1000
        bar = st.progress(0)
        
        # Tensor targets
        target_tensor = torch.FloatTensor(target_vector).to(device)
        
        for i in range(steps):
            opt.zero_grad()
            x_pred = model.inverse(z)
            
            # --- THE CORE LOGIC ---
            # 1. Fix Pre-treatment (Strict)
            loss_pre = ((x_pred[:, config['idx_pre']] - x_orig[:, config['idx_pre']])**2).sum() * 1000
            
            # 2. Move Interventions to Target (Strict)
            loss_int = ((x_pred[:, config['idx_int']] - target_tensor)**2).sum() * 1000
            
            # 3. Outcomes are FREE (No loss term)
            # This allows outcomes to change naturally based on Pre + Int changes
            
            # 4. Regularization
            loss_reg = 0.5 * (z**2).sum()
            
            loss = loss_pre + loss_int + loss_reg
            loss.backward()
            opt.step()
            
            if i % 100 == 0: bar.progress((i+1)/steps)
        
        bar.progress(1.0)
        
        # Results
        with torch.no_grad():
            x_cf = model.inverse(z)
            cf_np = x_cf[0].cpu().numpy()
            
        # Visualization
        st.write("### Trajectory Comparison")
        
        # Create DataFrame for plotting
        res_df = pd.DataFrame({
            "Week": range(1, len(config['idx_out']) + 1),
            "Original Outcome": orig_np[config['idx_out']],
            "Counterfactual Outcome": cf_np[config['idx_out']]
        })
        
        # Matplotlib Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(res_df["Week"], res_df["Original Outcome"], 'o-', label='Original', color='skyblue', linewidth=2)
        ax.plot(res_df["Week"], res_df["Counterfactual Outcome"], 'o-', label='Counterfactual', color='coral', linewidth=2, linestyle='--')
        
        ax.set_title(f"Clinical Outcome Trajectory (Patient {pat_id})")
        ax.set_xlabel("Time (Weeks)")
        ax.set_ylabel("Outcome Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Data Table
        st.write("#### Detailed Values")
        st.dataframe(res_df)
        
        # Pre-treatment Check
        st.write("#### Pre-treatment Variable Check (Should be Fixed)")
        pre_check = pd.DataFrame([orig_np[config['idx_pre']], cf_np[config['idx_pre']]], 
                                 columns=config['pre'], index=["Original", "Counterfactual"])
        st.dataframe(pre_check)

elif st.session_state['data_tensor'] is None:
    st.info("Start by selecting a Data Source.")
