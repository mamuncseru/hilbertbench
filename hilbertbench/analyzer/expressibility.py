"""
hilbertbench/analyzer/expressibility.py

Consumes a HilbertBench Parquet trace to compute the Expressibility 
of an ansatz using Kullback-Leibler (KL) Divergence against the Haar measure.
"""
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


class ExpressibilityAnalyzer:
    """
    Evaluates the Ansatz Expressibility axis (Thrust 1).
    Expects a trace generated in 'Active Mode' where parameters were uniformly sampled.
    """

    def __init__(self, run_dir: Path | str, num_qubits: int):
        self.run_dir = Path(run_dir)
        self.num_qubits = num_qubits
        self.parquet_path = self.run_dir / "events.parquet"
        self.catalog_path = self.run_dir / "catalog.json"
        
        # Dimension of the Hilbert space
        self.N = 2 ** self.num_qubits

        if not self.parquet_path.exists() or not self.catalog_path.exists():
            raise FileNotFoundError("Missing events.parquet or catalog.json. Ensure trace is valid.")

        self.trace_table = pq.read_table(self.parquet_path)
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)["artifacts"]

    def _load_numpy_artifact(self, artifact_ref: str) -> np.ndarray:
        """Safely loads physical artifacts using 2-char sharding."""
        meta = self.catalog[artifact_ref]
        hash_hex = artifact_ref.replace("sha256:", "")
        shard = hash_hex[:2]
        
        file_path = self.run_dir / "artifacts" / shard / f"{hash_hex}.npy"
        if not file_path.exists():
            file_path = self.run_dir / meta["file_path"] # Fallback
            
        return np.load(file_path, allow_pickle=True)

    def _compute_haar_probabilities(self, bin_edges: np.ndarray) -> np.ndarray:
        """
        Computes the theoretical Haar measure probabilities for the given bins.
        P_Haar(F) = (N-1)(1-F)^(N-2)
        """
        # Integrate the PDF over each bin
        p_haar = []
        for i in range(len(bin_edges) - 1):
            f_low = bin_edges[i]
            f_high = bin_edges[i + 1]
            
            # CDF of Haar measure: C(F) = 1 - (1 - F)^(N - 1)
            cdf_high = 1.0 - (1.0 - f_high) ** (self.N - 1)
            cdf_low = 1.0 - (1.0 - f_low) ** (self.N - 1)
            
            p_haar.append(cdf_high - cdf_low)
            
        return np.array(p_haar)

    def calculate_kl_divergence(self, num_bins: int = 75) -> Dict[str, Any]:
        """
        Computes the KL divergence between the empirical fidelity distribution
        of the trace and the theoretical Haar distribution.
        """
        df = self.trace_table.to_pandas()
        valid_spans = df[(df["status"] == "COMPLETED") & (df["outcome_ref"].notnull())]

        if valid_spans.empty:
            return {"kl_divergence": None, "status": "Insufficient Data"}

        # 1. Load all statevectors (assuming active mode ran statevector simulator)
        statevectors = []
        for _, row in valid_spans.iterrows():
            try:
                sv = self._load_numpy_artifact(row["outcome_ref"])
                # Handle varying tensor shapes depending on framework
                statevectors.append(np.ravel(sv))
            except Exception:
                continue

        if len(statevectors) < 2:
            return {"kl_divergence": None, "status": "Requires at least 2 samples"}

        # 2. Compute pairwise fidelities F = |<psi_i | psi_j>|^2
        fidelities = []
        num_samples = len(statevectors)
        
        # To avoid O(N^2) explosion on massive traces, sample pairs
        max_pairs = min(5000, num_samples * (num_samples - 1) // 2)
        
        for _ in range(max_pairs):
            idx1, idx2 = np.random.choice(num_samples, 2, replace=False)
            overlap = np.vdot(statevectors[idx1], statevectors[idx2])
            fidelity = np.abs(overlap) ** 2
            fidelities.append(float(fidelity))

        fidelities = np.array(fidelities)

        # 3. Create distributions
        bin_edges = np.linspace(0, 1, num_bins + 1)
        p_ansatz, _ = np.histogram(fidelities, bins=bin_edges, density=False)
        
        # Normalize empirical distribution
        p_ansatz = p_ansatz / np.sum(p_ansatz)
        
        # Get theoretical distribution
        p_haar = self._compute_haar_probabilities(bin_edges)

        # 4. Calculate KL Divergence (avoiding log(0) and div by 0)
        epsilon = 1e-10
        p_ansatz_safe = np.where(p_ansatz == 0, epsilon, p_ansatz)
        p_haar_safe = np.where(p_haar == 0, epsilon, p_haar)

        kl_div = np.sum(p_ansatz_safe * np.log(p_ansatz_safe / p_haar_safe))

        # Heuristic: KL < 0.1 is highly expressive, KL > 1.0 is highly rigid
        if kl_div < 0.1:
            status = "Highly Expressive (Matches Haar)"
        elif kl_div < 0.5:
            status = "Moderately Expressive"
        else:
            status = "Low Expressibility (Rigid Ansatz)"

        return {
            "kl_divergence": float(kl_div),
            "num_pairs_evaluated": max_pairs,
            "status": status,
            "histogram_ansatz": p_ansatz.tolist(),
            "histogram_haar": p_haar.tolist(),
            "bin_edges": bin_edges.tolist()
        }