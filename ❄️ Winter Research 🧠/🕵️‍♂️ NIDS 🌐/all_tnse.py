import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GINConv, GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data
from scapy.all import rdpcap, IP, TCP, UDP
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Function to extract features from a single PCAP file using `scapy`
def extract_features_from_pcap_scapy(pcap_file, sample=10):
    print(f"Processing: {pcap_file}")  # Debugging output

    try:
        packets = rdpcap(pcap_file)  # Read packets from PCAP
    except Exception as e:
        print(f"Error reading {pcap_file}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

    # Limit the number of packets processed (if sample size is set)
    packets = packets[:sample] if len(packets) > sample else packets

    data = []
    
    for packet in packets:
        if IP in packet:
            timestamp = packet.time  # Packet timestamp
            src_ip = packet[IP].src  # Source IP
            dst_ip = packet[IP].dst  # Destination IP
            protocol = "TCP" if TCP in packet else "UDP" if UDP in packet else "Other"
            length = len(packet)  # Packet length

            data.append([timestamp, src_ip, dst_ip, protocol, length])

    return pd.DataFrame(data, columns=['Timestamp', 'Source IP', 'Destination IP', 'Protocol', 'Length'])

# Function to process all PCAP files in given directories
def process_pcap_directories(pcap_directories, sample=10):
    all_pcap_data = []
    
    for pcap_directory in pcap_directories:
        if not os.path.isdir(pcap_directory):
            print(f"Skipping: {pcap_directory} (Not a valid directory)")
            continue
        
        # Get all .pcap files in the directory
        pcap_files = [os.path.join(pcap_directory, f) for f in os.listdir(pcap_directory) if f.endswith('.pcap')]

        if not pcap_files:
            print(f"No PCAP files found in: {pcap_directory}")
            continue

        for pcap_file in pcap_files:
            pcap_df = extract_features_from_pcap_scapy(pcap_file, sample=sample)
            if not pcap_df.empty:
                all_pcap_data.append(pcap_df)

    # Merge all extracted data into a single DataFrame
    return pd.concat(all_pcap_data, ignore_index=True) if all_pcap_data else pd.DataFrame()

# Example: List of directories containing PCAP files
pcap_directories = ['/x1/aabbas1/snap/ACI/pcap_combined/Combined_Pcaps/Spoofing Pcaps',
                    '/x1/aabbas1/snap/ACI/pcap_combined/Combined_Pcaps/Brute Force Pcaps',
                    '/x1/aabbas1/snap/ACI/pcap_combined/Combined_Pcaps/Recon Pcaps',
                    '/x1/aabbas1/snap/ACI/pcap_combined/Combined_Pcaps/Benign Pcaps',
                    '/x1/aabbas1/snap/ACI/pcap_combined/Combined_Pcaps/DoS_Pcaps'
]

# Process all PCAP files inside these directories with a sample limit
pcap_combined_df = process_pcap_directories(pcap_directories, sample=10)

# Step 1: Load and Merge the Dataset (CSV files)
dataset_paths = ['/x1/aabbas1/snap/CIC/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
	'/x1/aabbas1/snap/CIC/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    	'/x1/aabbas1/snap/CIC/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    	'/x1/aabbas1/snap/CIC/Monday-WorkingHours.pcap_ISCX.csv',
    	'/x1/aabbas1/snap/CIC/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    	'/x1/aabbas1/snap/CIC/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    	'/x1/aabbas1/snap/CIC/Tuesday-WorkingHours.pcap_ISCX.csv',
    	'/x1/aabbas1/snap/CIC/Wednesday-workingHours.pcap_ISCX.csv',
    	'/x1/aabbas1/snap/ACI/aci_csv/ACI-IoT-2023.csv',
    	'/x1/aabbas1/snap/ACI/aci_csv/ACI-IoT-2023-Payload.csv']

df_list = [pd.read_csv(path, low_memory=False).rename(columns=lambda x: x.strip().lower()) for path in dataset_paths]
merged_df = pd.concat(df_list, axis=0, ignore_index=True)

# Step 2: Merge the PCAP Data with CSV Data
if 'source ip' in merged_df.columns and 'source ip' in pcap_combined_df.columns:
    merged_df = merged_df.merge(pcap_combined_df, how='left', on='source ip')

# Step 3: Data Cleaning and Attack Label Mapping
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(merged_df[numeric_cols].median())

# Handle different spellings of 'Label' column
if 'label' in merged_df.columns and 'Label' in merged_df.columns:
    merged_df['Label'] = merged_df[['Label', 'label']].bfill(axis=1)['Label']
elif 'label' in merged_df.columns:
    merged_df.rename(columns={'label': 'Label'}, inplace=True)

if 'Label' not in merged_df.columns:
    raise KeyError("Error: The 'Label' column is missing from the dataset.")

merged_df['Label'] = merged_df['Label'].astype(str).fillna('Unknown').str.strip()

# Define attack mapping
attack_map = {
    'BENIGN': 'Benign',
    'DDoS': 'DDoS',
    'DoS Hulk': 'DoS Hulk',
    'DoS GoldenEye': 'DoS GoldenEye',
    'DoS slowloris': 'DoS Slowloris',
    'DoS Slowhttptest': 'DoS SlowHTTPTest',
    'PortScan': 'Port Scan',
    'FTP-Patator': 'Brute Force',
    'SSH-Patator': 'Brute Force',
    'Bot': 'Botnet',
    'Web Attack - Brute Force': 'Web Exploit',
    'Web Attack - XSS': 'Web Exploit',
    'Web Attack - Sql Injection': 'SQL Injection',
    'Infiltration': 'Infiltration',
    'Heartbleed': 'Heartbleed',
}

merged_df['Attack Type'] = merged_df['Label'].map(attack_map).fillna('Unknown')

# Encode attack labels
label_encoder = LabelEncoder()
merged_df['Attack Label'] = label_encoder.fit_transform(merged_df['Attack Type'])

# Step 4: Feature Selection
drop_cols = ['source ip', 'destination ip', 'Label', 'Attack Type']
merged_df.drop(columns=[col for col in drop_cols if col in merged_df.columns], inplace=True)

feature_cols = [col for col in ['flow duration', 'flow packets/s', 'flow bytes/s'] if col in merged_df.columns]

# Apply scaling only to the selected features
scaler = StandardScaler()
merged_df[feature_cols] = scaler.fit_transform(merged_df[feature_cols])

# Step 5: Undersampling
X = merged_df[feature_cols].values
y = np.ravel(merged_df['Attack Label'].values)

undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)
merged_df_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
merged_df_resampled['Attack Label'] = y_resampled

# Step 6: Computed Class Weights for Balanced Training
class_weights = compute_class_weight('balanced', classes=np.unique(y_resampled), y=y_resampled)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Step 7: Created Graph Using Feature-Based Similarity (KNN)
def create_feature_based_graph(df, k=12):
    num_nodes = len(df)
    node_features = df[feature_cols].values.astype(np.float32)
    node_labels = df['Attack Label'].values

    knn = NearestNeighbors(n_neighbors=k+1)
    knn.fit(node_features)
    neighbors = knn.kneighbors(node_features, return_distance=False)

    edge_list = [(i, neighbor) for i in range(num_nodes) for neighbor in neighbors[i][1:]]
    edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
    node_features = torch.tensor(node_features, dtype=torch.float)
    node_labels = torch.tensor(node_labels, dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index, y=node_labels)

graph_data = create_feature_based_graph(merged_df_resampled, k=5)

# Define models
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GIN, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        ))

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class MPNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MPNN, self).__init__(aggr='mean')
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = self.fc1(x)
        x = self.propagate(edge_index, x=x)
        x = F.relu(x)
        return F.log_softmax(self.fc2(x), dim=1)
    
    def message(self, x_j):
        return x_j

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Training Function
def train(model, graph_data, class_weights, epochs=20, lr=0.01, weight_decay=5e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    graph_data = graph_data.to(device)
    graph_data.y = graph_data.y.to(device)  # Ensure the labels are on the correct device

    # Move class_weights to the same device as the model
    class_weights = class_weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = criterion(out, graph_data.y)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

def test(model, graph_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    graph_data = graph_data.to(device)

    model.eval()
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index)
        pred = out.argmax(dim=1)  # Get the predicted class labels
        acc = (pred == graph_data.y).sum().item() / graph_data.y.size(0)  # Compute accuracy
        print(f"Test Accuracy: {acc:.4f}")

        # Print classification report
        print(classification_report(graph_data.y.cpu().numpy(), pred.cpu().numpy()))

        # Extract embeddings (before the final softmax layer)
        embeddings = out.cpu().numpy()
        labels = graph_data.y.cpu().numpy()

    return embeddings, labels

def plot_tsne(embeddings, labels, label_encoder, title="t-SNE Visualization", filename="tsne_plot"):
    # Map numerical labels back to attack types
    attack_labels = label_encoder.inverse_transform(labels)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))

    # Create a scatter plot with colors mapped to attack types
    unique_labels = np.unique(labels)
    colors = [plt.cm.jet(i / max(unique_labels)) for i in unique_labels]
    
    for label, color in zip(unique_labels, colors):
        idx = labels == label
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], color=color, label=label_encoder.inverse_transform([label])[0], alpha=0.7)

    # Create a legend with attack type names only
    plt.legend(title="Attack Type", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # Save the plot
    plt.savefig(f"{filename}.png", bbox_inches='tight')
    plt.close()
    
# Train and test each model
models = {
    'GraphSAGE': GraphSAGE(len(feature_cols), 64, len(np.unique(merged_df['Attack Label']))),
    'GIN': GIN(len(feature_cols), 64, len(np.unique(merged_df['Attack Label']))),
    'MPNN': MPNN(len(feature_cols), 64, len(np.unique(merged_df['Attack Label']))),
    'GCN': GCN(len(feature_cols), 64, len(np.unique(merged_df['Attack Label']))),
    'GAT': GAT(len(feature_cols), 64, len(np.unique(merged_df['Attack Label'])))
}

# Training and testing all models
for model_name, model in models.items():
    print(f"\nTraining {model_name} model...")
    trained_model = train(model, graph_data, class_weights)
    print(f"Testing {model_name} model...")
    embeddings, labels = test(trained_model, graph_data)

    # Visualize embeddings using t-SNE with attack labels
    plot_tsne(embeddings, labels, label_encoder, title=f"t-SNE Visualization ({model_name})", filename=f"tsne_{model_name}")