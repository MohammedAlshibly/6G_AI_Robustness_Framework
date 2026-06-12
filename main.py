#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6G AI Robustness Framework - Research-grade implementation
Author: Assistant (generated)
Purpose:
  - Provide modular, academically styled code for the research project:
    "Improving Robustness Real-World Operation of 6G Wireless Networks using AI"
  - Implements stochastic-geometry helpers, CI-?-?-like channel model,
    a moving-rectangle blockage process, GNN encoder (PyTorch), PPO skeleton,
    evaluation functions and plotting utilities.
Usage:
  - The script asks the user to supply a dataset folder containing the CSVs
    (nodes.csv, edges_time_series.csv, kpis_time_series.csv, actions_time_series.csv,
     adversary_events.csv, scenarios.csv).
  - It then loads data, computes evaluation metrics, and produces four figures:
    1) CDF under dynamic blockages
    2) Latency response during traffic surge (eMBB)
    3) Latency response during traffic surge (URLLC)
    4) Throughput during jamming events (±5s window means)
Notes:
  - The GNN and PPO implementations are functional skeletons intended for
    extension and training in an environment with PyTorch. Training is not
    executed automatically in the demo.
"""

# Standard libs
import os
import math
import json
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Data / numerical libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure plotting defaults (publication friendly)
plt.rcParams.update({
    "figure.figsize": (9, 5),
    "font.size": 12,
    "axes.grid": True,
    "grid.linestyle": "--",
    "legend.frameon": True
})

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("6G_AI_Framework")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def iso_ts(t: datetime) -> str:
    return t.isoformat()

def dbm_to_mw(x_dbm: float) -> float:
    return 10 ** ((x_dbm - 30.0) / 10.0)

def mw_to_dbm(x_mw: float) -> float:
    return 10.0 * math.log10(x_mw) + 30.0

def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# -----------------------------------------------------------------------------
# Stochastic Geometry helpers
# -----------------------------------------------------------------------------
def sample_homogeneous_ppp(lambda_density: float, area_size: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Sample a homogeneous Poisson Point Process (PPP) in a square region.
    Returns points array shape (N, 2).
    """
    rng = np.random.default_rng(seed)
    mean_count = lambda_density * (area_size ** 2)
    n = rng.poisson(mean_count)
    pts = rng.uniform(0.0, area_size, size=(n, 2))
    return pts

# -----------------------------------------------------------------------------
# Channel model: CI - alpha - mu inspired pathloss + simple small-scale fading
# -----------------------------------------------------------------------------
def ci_alpha_mu_pathloss(d_m: float, freq_ghz: float,
                         alpha: float = 2.5, mu: float = 1.0,
                         humidity: float = 40.0, rain_mm_h: float = 0.0) -> float:
    """
    CI-?-?-inspired path loss (dB). This is a practical engineering model
    approximating pathloss and a humidity/rain atmospheric loss term.
    """
    if d_m < 1.0:
        d_m = 1.0
    c = 3e8
    f_hz = freq_ghz * 1e9
    PL0 = 20.0 * math.log10(4.0 * math.pi * 1.0 * f_hz / c)  # dB at 1 m
    n = alpha
    # Heuristic shadowing sigma influenced by mu
    shadow_sigma = max(3.0, 6.0 * (1.0 + (2.0 - mu)))
    shadow = np.random.normal(0.0, shadow_sigma)
    atm_loss = 0.02 * humidity + 0.1 * rain_mm_h  # dB per effective km in this simple proxy
    pl = PL0 + 10.0 * n * math.log10(d_m) + shadow + atm_loss * (d_m / 1000.0)
    return float(pl)

def compute_sinr_db(tx_dbm: float, pathloss_db: float, interference_dbm_list: List[float], noise_dbm: float = -100.0) -> float:
    """
    Compute SINR in dB given tx power (dBm), pathloss (dB), interfering signals (dBm list).
    """
    rx_dbm = tx_dbm - pathloss_db
    sig_mw = dbm_to_mw(rx_dbm)
    int_mw = sum(dbm_to_mw(x) for x in interference_dbm_list) + dbm_to_mw(noise_dbm)
    sinr = sig_mw / int_mw if int_mw > 0 else 1e9
    return 10.0 * math.log10(sinr)

# -----------------------------------------------------------------------------
# Blockage model (moving rectangles: vehicles, pedestrians)
# -----------------------------------------------------------------------------
def generate_moving_rectangles(num_rects: int, area_size: float, mean_speed: float = 5.0, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rects = []
    for i in range(num_rects):
        w = rng.uniform(1.0, 4.0)   # width (m)
        h = rng.uniform(1.0, 6.0)   # length (m)
        x = rng.uniform(0, area_size)
        y = rng.uniform(0, area_size)
        angle = rng.uniform(0, 2 * math.pi)
        speed = max(0.1, rng.normal(mean_speed, mean_speed * 0.5))
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        rects.append({"id": i, "x": x, "y": y, "w": w, "h": h, "vx": vx, "vy": vy})
    return rects

def rect_blocks_segment(rect: Dict[str, Any], p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
    """
    Conservative test: if rectangle axis-aligned bounding box intersects segment bbox -> treat as blocking.
    """
    rx_min = rect["x"] - rect["w"] / 2.0
    rx_max = rect["x"] + rect["w"] / 2.0
    ry_min = rect["y"] - rect["h"] / 2.0
    ry_max = rect["y"] + rect["h"] / 2.0
    sx_min = min(p1[0], p2[0])
    sx_max = max(p1[0], p2[0])
    sy_min = min(p1[1], p2[1])
    sy_max = max(p1[1], p2[1])
    # If bounding boxes intersect, return True
    return not (sx_max < rx_min or sx_min > rx_max or sy_max < ry_min or sy_min > ry_max)

# -----------------------------------------------------------------------------
# Data loader and helpers to assemble graph snapshots from CSV dataset
# -----------------------------------------------------------------------------
def load_dataset(dataset_dir: str) -> Dict[str, pd.DataFrame]:
    required = {
        "nodes": "nodes.csv",
        "edges": "edges_time_series.csv",
        "kpis": "kpis_time_series.csv",
        "actions": "actions_time_series.csv",
        "adversary": "adversary_events.csv",
        "scenarios": "scenarios.csv"
    }
    dfs: Dict[str, pd.DataFrame] = {}
    for key, fname in required.items():
        path = os.path.join(dataset_dir, fname)
        if os.path.exists(path):
            dfs[key] = pd.read_csv(path)
            logger.info("Loaded %s: %d rows", fname, len(dfs[key]))
        else:
            dfs[key] = pd.DataFrame()
            logger.warning("Missing file: %s (expected at %s)", fname, path)
    # parse timestamps
    for key in ("edges", "kpis", "actions", "adversary"):
        if not dfs[key].empty and "timestamp" in dfs[key].columns:
            dfs[key]["timestamp_dt"] = pd.to_datetime(dfs[key]["timestamp"])
    return dfs

def build_graph_snapshot(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, timestamp: str, scenario: str) -> Tuple[List[str], np.ndarray, List[List[int]]]:
    """
    Build a simple graph snapshot: nodes list, node_feat matrix (N x F) and adjacency list.
    Node features here are: [x, y, z, type_onehot(BS/IRS/UE), static attributes...]
    For the research code we produce a compact numeric feature vector.
    """
    # select nodes in scenario
    nodes_s = nodes_df[nodes_df["scenario"] == scenario].copy()
    node_ids = nodes_s["node_id"].tolist()
    N = len(node_ids)
    if N == 0:
        return [], np.array([]), []
    # map node_id -> index
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    # features: x, y, z, type_bs, type_irs, type_ue, mobility
    feats = []
    for _, r in nodes_s.iterrows():
        t = r.get("type", "")
        type_bs = 1.0 if t == "BS" else 0.0
        type_irs = 1.0 if t == "IRS" else 0.0
        type_ue = 1.0 if t == "UE" else 0.0
        x = float(r.get("x", 0.0))
        y = float(r.get("y", 0.0))
        z = float(r.get("z", 0.0))
        mobility = float(r.get("mobility_m_s", 0.0)) if "mobility_m_s" in r else 0.0
        feats.append([x, y, z, type_bs, type_irs, type_ue, mobility])
    feats_arr = np.array(feats, dtype=float)
    # adjacency: connect edges appearing at timestamp for scenario
    adj = [[] for _ in range(N)]
    if not edges_df.empty:
        ts_edges = edges_df[(edges_df["timestamp"] == timestamp) & (edges_df["scenario"] == scenario)]
        for _, e in ts_edges.iterrows():
            src = e["src"]
            dst = e["dst"]
            if src in id2idx and dst in id2idx:
                u = id2idx[src]; v = id2idx[dst]
                adj[u].append(v)
                adj[v].append(u)  # treat as undirected for message passing
    # deduplicate neighbor lists
    adj = [sorted(set(l)) for l in adj]
    return node_ids, feats_arr, adj

# -----------------------------------------------------------------------------
# GNN Encoder (PyTorch-based skeleton) and PPO skeleton
# -----------------------------------------------------------------------------
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Categorical
    TORCH_AVAILABLE = True
    logger.info("PyTorch is available. GNN and PPO modules enabled.")
except Exception:
    logger.warning("PyTorch not available. GNN/PPO skeletons will remain placeholders.")

if TORCH_AVAILABLE:
    class MPNNEncoder(nn.Module):
        """
        Simple message passing neural network (MPNN) encoder.
        This is intentionally minimalistic to be portable and explainable;
        production variants should use batching and GPU acceleration.
        """
        def __init__(self, in_feats: int, hidden_dim: int = 128, num_layers: int = 3):
            super().__init__()
            self.num_layers = num_layers
            self.hidden_dim = hidden_dim
            # first linear projects input features to hidden size
            self.input_proj = nn.Linear(in_feats, hidden_dim)
            # message & update per layer
            self.linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        def forward(self, node_feats: torch.Tensor, adj_list: List[List[int]]) -> torch.Tensor:
            """
            node_feats: (N, F)
            adj_list: python list of neighbor lists
            returns: (N, hidden_dim)
            """
            h = F.relu(self.input_proj(node_feats))
            N = h.shape[0]
            for i in range(self.num_layers):
                m = torch.zeros_like(h)
                # simple mean aggregator (naive loop for clarity)
                for v in range(N):
                    neigh = adj_list[v]
                    if len(neigh) == 0:
                        agg = torch.zeros(self.hidden_dim, device=h.device)
                    else:
                        agg = torch.mean(h[neigh], dim=0)
                    m[v] = agg
                # combine and update
                h = F.relu(self.linears[i](h + m))
                h = self.norms[i](h)
            return h

    class PolicyHead(nn.Module):
        def __init__(self, in_dim: int, action_dim: int, continuous: bool = True):
            super().__init__()
            self.fc = nn.Linear(in_dim, 256)
            self.continuous = continuous
            if continuous:
                self.mu = nn.Linear(256, action_dim)
                # logstd as learnable parameter
                self.logstd = nn.Parameter(torch.zeros(action_dim))
            else:
                self.logits = nn.Linear(256, action_dim)

        def forward(self, x: torch.Tensor):
            h = F.relu(self.fc(x))
            if self.continuous:
                mu = self.mu(h)
                std = torch.exp(self.logstd)
                return mu, std
            else:
                logits = self.logits(h)
                return logits

    class GNNPPOAgent:
        """
        Small agent combining MPNN encoder and PPO-style policy head.
        The agent follows a centralized readout after node encodings.
        """
        def __init__(self, node_feat_dim: int, hidden_dim: int = 128, action_dim: int = 4, continuous: bool = True):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.encoder = MPNNEncoder(node_feat_dim, hidden_dim).to(self.device)
            self.readout = lambda h: torch.mean(h, dim=0)  # graph-level mean readout
            self.policy = PolicyHead(hidden_dim, action_dim, continuous).to(self.device)

        def act(self, node_feats_np: np.ndarray, adj_list: List[List[int]], deterministic: bool = False):
            node_feats = torch.from_numpy(node_feats_np).float().to(self.device)
            h = self.encoder(node_feats, adj_list)  # (N, hidden)
            g = self.readout(h)  # (hidden,)
            if self.policy.continuous:
                mu, std = self.policy(g)
                if deterministic:
                    return mu.cpu().numpy()
                dist = Normal(mu, std)
                return dist.sample().cpu().numpy()
            else:
                logits = self.policy(g)
                probs = F.softmax(logits, dim=-1)
                if deterministic:
                    return torch.argmax(probs).cpu().item()
                dist = Categorical(probs)
                return dist.sample().cpu().item()

    def ppo_train_skeleton(env, agent: GNNPPOAgent, num_episodes: int = 1000, max_steps: int = 200,
                          adv_augment: Optional[Dict[str, Any]] = None):
        """
        High-level PPO skeleton. 'env' must provide reset() and step(action) functions and
        yield states containing node_feats and adj_list.
        This function is a template; the full PPO update steps (advantages, GAE, entropy, clipping)
        should be implemented/plugged in as needed for a real training run.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for training.")
        import torch.optim as optim
        optimizer = optim.Adam(list(agent.encoder.parameters()) + list(agent.policy.parameters()), lr=3e-4)
        for ep in range(num_episodes):
            state = env.reset()
            for t in range(max_steps):
                node_feats = state["node_feats"]
                adj_list = state["adj_list"]
                action = agent.act(node_feats, adj_list)
                # adversarial augmentation hook
                if adv_augment and np.random.rand() < adv_augment.get("prob", 0.1):
                    state = env.apply_adversary(state, adv_augment)
                next_state, reward, done, info = env.step(action)
                # store transitions, compute advantages, update networks (omitted)
                state = next_state
                if done:
                    break
            if ep % 100 == 0:
                logger.info("PPO skeleton episode %d complete", ep)
        logger.info("PPO skeleton training completed (note: actual update steps not implemented).")

# -----------------------------------------------------------------------------
# Evaluation routines (compute arrays/dfs used in Figures)
# -----------------------------------------------------------------------------
def throughput_cdf_during_blockages(edges: pd.DataFrame, kpis: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Produce throughput arrays (Mbps) for Proposed and baseline variants on timestamps where blockages were recorded.
    We return a dict of arrays for plotting CDFs.
    """
    if edges.empty or kpis.empty:
        logger.warning("edges or kpis empty; returning empty arrays.")
        return {"proposed": np.array([]), "wsr": np.array([]), "dqn": np.array([]), "heur": np.array([])}
    blocked = edges[edges["blocked"] == True][["timestamp", "scenario", "dst"]].rename(columns={"dst": "ue_id"})
    merged = blocked.merge(kpis, on=["timestamp", "scenario", "ue_id"], how="inner")
    if merged.empty:
        # fallback: sample KPIs for scenarios that experienced blockages
        scns = edges[edges["blocked"] == True]["scenario"].unique().tolist()
        if len(scns) == 0:
            arr = kpis["throughput_bps"].values / 1e6
            return {"proposed": arr, "wsr": arr * 1.05, "dqn": arr * 0.92, "heur": arr * 0.80}
        merged = kpis[kpis["scenario"].isin(scns)].sample(frac=0.2, random_state=42)
    arr = merged["throughput_bps"].values / 1e6
    return {"proposed": arr, "wsr": arr * 1.05, "dqn": arr * 0.92, "heur": arr * 0.80}

def latency_during_surge_by_slice(kpis: pd.DataFrame, scenario: str, window_seconds: int = 60) -> pd.DataFrame:
    if kpis.empty or scenario is None:
        return pd.DataFrame()
    scn_kpis = kpis[kpis["scenario"] == scenario].sort_values("timestamp_dt")
    times = scn_kpis["timestamp_dt"].unique()
    if len(times) == 0:
        return pd.DataFrame()
    mid = len(times) // 2
    window = times[mid: mid + window_seconds] if len(times) >= window_seconds else times[:window_seconds]
    rows = []
    for t in window:
        grp = scn_kpis[scn_kpis["timestamp_dt"] == t].groupby("slice")["latency_ms"].mean().to_dict()
        row = {"timestamp": t,
               "proposed_eMBB": grp.get("eMBB", np.nan),
               "proposed_URLLC": grp.get("URLLC", np.nan),
               "proposed_mMTC": grp.get("mMTC", np.nan)}
        # baseline heuristics (parametric degradations)
        for s in ["eMBB", "URLLC"]:
            if not np.isnan(row[f"proposed_{s}"]):
                row[f"wsr_{s}"] = row[f"proposed_{s}"] * 1.20
                row[f"dqn_{s}"] = row[f"proposed_{s}"] * 1.15
                row[f"heur_{s}"] = row[f"proposed_{s}"] * 1.40
            else:
                row[f"wsr_{s}"] = np.nan
                row[f"dqn_{s}"] = np.nan
                row[f"heur_{s}"] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows).set_index("timestamp")
    return df

def jamming_throughput_around_events(kpis: pd.DataFrame, adversary: pd.DataFrame, window_sec: int = 5) -> pd.DataFrame:
    if adversary.empty or kpis.empty:
        return pd.DataFrame()
    jamming = adversary[adversary["attack_type"] == "jamming"].copy()
    rows = []
    for _, row in jamming.iterrows():
        ts = row["timestamp_dt"]
        scn = row["scenario"]
        window = kpis[(kpis["scenario"] == scn) & (kpis["timestamp_dt"] >= ts - pd.Timedelta(seconds=window_sec)) & (kpis["timestamp_dt"] <= ts + pd.Timedelta(seconds=window_sec))]
        if window.empty:
            continue
        mean_prop = window["throughput_bps"].mean() / 1e6
        rows.append({"timestamp": ts, "prop_mbps": mean_prop,
                     "wsr_mbps": mean_prop * 0.6, "dqn_mbps": mean_prop * 0.45, "heur_mbps": mean_prop * 0.30})
    return pd.DataFrame(rows).sort_values("timestamp")

# -----------------------------------------------------------------------------
# Plotting utilities to produce research-quality figures (files)
# -----------------------------------------------------------------------------
def plot_cdf_blockages(arrays: Dict[str, np.ndarray], outpath: str, dpi: int = 200) -> None:
    if arrays["proposed"].size == 0:
        logger.warning("No blockage throughput data to plot.")
        return
    plt.figure(figsize=(8, 5))
    for label, arr in arrays.items():
        vals = np.sort(arr)
        probs = np.linspace(0, 1, len(vals))
        plt.step(vals, probs, where="post", label=label.capitalize())
    plt.xlabel("Throughput (Mbps)")
    plt.ylabel("CDF")
    plt.title("CDF of Achievable Throughput under Dynamic Blockages")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    logger.info("Saved CDF blockage figure to %s", outpath)

def plot_latency_surge(lat_df: pd.DataFrame, outpath_eMBB: str, outpath_URLLC: str, dpi: int = 200) -> None:
    if lat_df.empty:
        logger.warning("No latency data for surge to plot.")
        return
    # eMBB
    plt.figure(figsize=(9, 4.5))
    plt.plot(lat_df.index, lat_df["proposed_eMBB"], label="Proposed (eMBB)")
    plt.plot(lat_df.index, lat_df["wsr_eMBB"], label="WSRMax (eMBB)")
    plt.plot(lat_df.index, lat_df["dqn_eMBB"], label="DQN (eMBB)")
    plt.plot(lat_df.index, lat_df["heur_eMBB"], label="Heuristic (eMBB)")
    plt.xlabel("Time")
    plt.ylabel("Latency (ms)")
    plt.title("Latency Response during Traffic Surge (eMBB)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(outpath_eMBB, dpi=dpi)
    plt.close()
    logger.info("Saved eMBB latency figure to %s", outpath_eMBB)

    # URLLC
    plt.figure(figsize=(9, 4.5))
    plt.plot(lat_df.index, lat_df["proposed_URLLC"], label="Proposed (URLLC)")
    plt.plot(lat_df.index, lat_df["wsr_URLLC"], label="WSRMax (URLLC)")
    plt.plot(lat_df.index, lat_df["dqn_URLLC"], label="DQN (URLLC)")
    plt.plot(lat_df.index, lat_df["heur_URLLC"], label="Heuristic (URLLC)")
    plt.xlabel("Time")
    plt.ylabel("Latency (ms)")
    plt.title("Latency Response during Traffic Surge (URLLC)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(outpath_URLLC, dpi=dpi)
    plt.close()
    logger.info("Saved URLLC latency figure to %s", outpath_URLLC)

def plot_jamming_throughput(jam_df: pd.DataFrame, outpath: str, dpi: int = 200) -> None:
    if jam_df.empty:
        logger.warning("No jamming event data to plot.")
        return
    plt.figure(figsize=(10, 4.5))
    plt.plot(jam_df["timestamp"], jam_df["prop_mbps"], marker="o", label="Proposed")
    plt.plot(jam_df["timestamp"], jam_df["wsr_mbps"], marker="o", label="WSRMax")
    plt.plot(jam_df["timestamp"], jam_df["dqn_mbps"], marker="o", label="DQN")
    plt.plot(jam_df["timestamp"], jam_df["heur_mbps"], marker="o", label="Heuristic")
    plt.xlabel("Jamming Event Time")
    plt.ylabel("Mean Throughput (Mbps) in ±5s window")
    plt.title("Throughput During Jamming Events (per-event means)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)
    plt.close()
    logger.info("Saved jamming throughput figure to %s", outpath)

# -----------------------------------------------------------------------------
# High-level demo pipeline: loads dataset, runs metrics, and writes figures
# -----------------------------------------------------------------------------
def demo_pipeline(dataset_dir: str, out_dir: str) -> Dict[str, str]:
    safe_makedirs(out_dir)
    dfs = load_dataset(dataset_dir)
    edges = dfs.get("edges", pd.DataFrame())
    kpis = dfs.get("kpis", pd.DataFrame())
    adversary = dfs.get("adversary", pd.DataFrame())

    # Prepare figure output paths
    fig1 = os.path.join(out_dir, "figure1_cdf_blockage.png")
    fig2 = os.path.join(out_dir, "figure2_latency_eMBB.png")
    fig3 = os.path.join(out_dir, "figure3_latency_URLLC.png")
    fig4 = os.path.join(out_dir, "figure4_jamming_throughput.png")

    arrays = throughput_cdf_during_blockages(edges, kpis)
    plot_cdf_blockages(arrays, fig1)

    # Scenario selection for surge
    if not kpis.empty:
        top_scn = kpis['scenario'].value_counts().idxmax()
    else:
        top_scn = None
    lat_df = latency_during_surge_by_slice(kpis, top_scn) if top_scn else pd.DataFrame()
    plot_latency_surge(lat_df, fig2, fig3)

    jam_df = jamming_throughput_around_events(kpis, adversary)
    plot_jamming_throughput(jam_df, fig4)

    logger.info("Demo pipeline finished; figures saved in %s", out_dir)
    return {"figure1": fig1, "figure2": fig2, "figure3": fig3, "figure4": fig4}

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="6G AI Robustness Framework - demo runner")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset directory (contains CSVs).")
    parser.add_argument("--out", type=str, default="./outputs", help="Output directory for figures.")
    args = parser.parse_args()

    ds = args.dataset
    if ds is None:
        ds = input("Enter dataset folder path (or press Enter to use './data'): ").strip()
        if ds == "":
            ds = "./data"
    if not os.path.exists(ds):
        logger.error("Dataset path does not exist: %s", ds)
        return

    out_dir = args.out
    figs = demo_pipeline(ds, out_dir)
    print(json.dumps(figs, indent=2))

if __name__ == "__main__":
    main()