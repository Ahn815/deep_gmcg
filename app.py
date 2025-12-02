import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="RealNVP Causal Inference", layout="wide")

# Device Configuration (Streamlit Cloud is usually CPU)
device = torch.device("cpu")

# --- 1. Model & Data Class Definition ---

class RealNVP(nn.Module):
    def __init__(self, dim=4, n_layers=10, hidden_dim=128):
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

# --- 2. Data Generation Function ---
def generate_data(n_samples):
    np.random.seed(42)
    age = np.random.randint(40, 80, n_samples)
    sex = np.random.binomial(1, 0.5, n_samples).astype(float)
    dose = np.random.uniform(0, 100, n_samples)

    # Causal model with noise
    outcome = (250 - 0.02 * dose ** 2 - 0.2 * age + 8.0 * sex + np.random.normal(0, 0.1, n_samples))
    outcome = np.clip(outcome, 50, 300)

    # Calculate Ground Truth (for comparison, noise-free)
    outcome_gt = (250 - 0.02 * 90 ** 2 - 0.2 * age + 8.0 * sex)

    data = np.column_stack([age, sex, dose, outcome])
    data_tensor = torch.FloatTensor(data).to(device)
    
    return data_tensor, age, sex, dose, outcome

# --- 3. Streamlit UI Start ---

st.title("💊 RealNVP Causal Inference Simulator")
st.markdown("""
This app uses a **RealNVP (Normalizing Flow)** model implemented in PyTorch to predict patient **Outcome** based on **Dose** and simulate **Counterfactual** scenarios.
""")

# Sidebar settings
st.sidebar.header("1. Experiment Settings")
N = st.sidebar.slider("Number of Samples (N)", 1000, 20000, 5000, step=1000)
epochs = st.sidebar.number_input("Training Epochs", value=100, min_value=10)
lr = st.sidebar.number_input("Learning Rate", value=1e-4, format="%.5f")

# Initialize Session State
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'data_tensor' not in st.session_state:
    st.session_state['data_tensor'] = None

# Data Generation Button
if st.sidebar.button("Generate Data & Reset"):
    data_tensor, age, sex, dose, outcome = generate_data(N)
    st.session_state['data_tensor'] = data_tensor
    st.session_state['data_full'] = (age, sex, dose, outcome)
    st.session_state['model'] = None # Reset model when data changes
    st.sidebar.success(f"Generated {N} data samples.")

# Show training UI only if data exists
if st.session_state['data_tensor'] is not None:
    data_tensor = st.session_state['data_tensor']
    
    st.subheader("2. Model Training")
    
    # Training Button
    if st.button("Start Model Training"):
        # Prepare Data Loaders
        train_size = int(0.8 * N)
        train_data, val_data = random_split(data_tensor, [train_size, N - train_size])
        train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=256, shuffle=False)

        # Initialize Model
        model = RealNVP(dim=4, n_layers=4, hidden_dim=12).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Progress Bar and Chart Containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_container = st.empty()
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
            
            # Update UI every 10 epochs or first epoch
            if epoch % 10 == 0 or epoch == 1:
                progress_bar.progress(epoch / epochs)
                status_text.text(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f}")
                
        st.session_state['model'] = model # Save trained model
        st.success("Training Complete!")
        
        # Plot Loss Curve
        fig_loss, ax_loss = plt.subplots(figsize=(10, 3))
        ax_loss.plot(loss_history, label='Train Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Negative Log Likelihood')
        ax_loss.legend()
        st.pyplot(fig_loss)

# Show inference UI only if the model is trained
if st.session_state['model'] is not None:
    st.divider()
    st.subheader("3. Counterfactual Inference")
    st.markdown("Simulate **'What would have happened if the dose was different?'** for a specific patient.")

    col1, col2 = st.columns(2)
    with col1:
        patient_idx = st.number_input("Select Patient ID (0 ~ N-1)", 0, N-1, 100)
    with col2:
        target_dose_input = st.number_input("Target Dose", 0.0, 100.0, 90.0)

    if st.button("Generate Counterfactual Result"):
        model = st.session_state['model']
        data_tensor = st.session_state['data_tensor']
        
        # Get Data
        x_orig = data_tensor[patient_idx:patient_idx+1].to(device)
        orig_vals = x_orig[0].cpu().numpy() # [Age, Sex, Dose, Outcome]
        
        # Calculate Ground Truth (using the formula)
        age_val, sex_val = orig_vals[0], orig_vals[1]
        gt_outcome = (250 - 0.02 * target_dose_input ** 2 - 0.2 * age_val + 8.0 * sex_val)
        gt_outcome = np.clip(gt_outcome, 50, 300)

        # Find Latent z
        model.eval()
        with torch.no_grad():
            z_init, _ = model.forward(x_orig)
        
        z = z_init.clone().detach().requires_grad_(True)
        opt_cf = optim.Adam([z], lr=2e-3)

        # Optimization Progress
        cf_progress = st.progress(0)
        
        # Counterfactual Optimization Loop
        steps = 7000
        for step in range(steps):
            opt_cf.zero_grad()
            x_pred = model.inverse(z)
            
            # Loss: (Age, Sex Fixed) + (Dose Target) + (Regularization)
            loss_fixed = ((x_pred[:, 0] - x_orig[:, 0]) ** 2 + (x_pred[:, 1] - x_orig[:, 1]) ** 2) * 1e6
            loss_dose = (x_pred[:, 2] - target_dose_input) ** 2
            loss_reg = 0.5 * z.square().sum()
            loss = 1e5 * (loss_fixed + loss_dose) + loss_reg
            
            loss.backward()
            opt_cf.step()
            
            if step % 100 == 0:
                cf_progress.progress((step + 1) / steps)
        
        cf_progress.progress(1.0)
        
        # Final Results
        with torch.no_grad():
            x_cf = model.inverse(z)
            cf_vals = x_cf[0].cpu().numpy()
            
        # Visualize Results
        st.write("### Result Comparison")
        
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Age (Fixed)", f"{orig_vals[0]:.0f}", delta=f"{cf_vals[0] - orig_vals[0]:.2f}", delta_color="off")
        m2.metric("Sex (Fixed)", f"{orig_vals[1]:.0f}", delta=f"{cf_vals[1] - orig_vals[1]:.2f}", delta_color="off")
        m3.metric("Dose (Changed)", f"{orig_vals[2]:.1f}", f"{cf_vals[2]:.1f} →")
        m4.metric("Outcome (Pred)", f"{orig_vals[3]:.1f}", f"{cf_vals[3]:.1f} (GT: {gt_outcome:.1f})")

        # Plot Graph
        fig, ax1 = plt.subplots(figsize=(10, 5))

        x_pos = [0, 2, 3]
        width = 0.35
        
        ax1.bar([p - width/2 for p in x_pos], orig_vals[[0, 2, 3]], width, label='Original (Factual)', color='skyblue')
        ax1.bar([p + width/2 for p in x_pos], cf_vals[[0, 2, 3]], width, label='Counterfactual', color='lightcoral')
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(['Age', 'Dose', 'Outcome'])
        ax1.set_ylabel("Values")
        ax1.set_title(f"Patient {patient_idx}: Dose {orig_vals[2]:.1f} -> {target_dose_input:.1f}")
        
        # Ground Truth Line
        ax1.axhline(y=gt_outcome, color='green', linestyle='--', linewidth=2, label=f'GT Outcome ({gt_outcome:.1f})')
        
        ax1.legend()
        st.pyplot(fig)

elif st.session_state['data_tensor'] is None:
    st.info("👈 Please click the 'Generate Data' button in the sidebar first.")
