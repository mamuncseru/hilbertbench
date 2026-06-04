"""
hilbertbench/analyzer/trainability.py

Consumes a HilbertBench Parquet trace and computes mechanistic diagnostics
for the Ansatz, specifically targeting Barren Plateau detection via outcome variance.
"""
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


class AnsatzAnalyzer:
    """
    Evaluates the Trainability axis (Thrust 1) by computing the statistical 
    properties of the quantum execution trajectory.
    """

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir)
        self.parquet_path = self.run_dir / "events.parquet"
        self.catalog_path = self.run_dir / "catalog.json"

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                "events.parquet not found. Please run the Parquet Storage Writer first."
            )
        if not self.catalog_path.exists():
            raise FileNotFoundError("catalog.json not found in the run directory.")

        if pq is None:
            raise ImportError("pyarrow is required for offline analysis.")

        # Load the highly compressed columnar trace
        self.trace_table = pq.read_table(self.parquet_path)
        
        # Load the artifact catalog to locate the physical .npy files
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)["artifacts"]

    def _load_numpy_artifact(self, artifact_ref: str) -> np.ndarray:
        """
        Retrieves a physical .npy array from the sharded artifacts directory
        using its sha256 reference.
        """
        if not artifact_ref or artifact_ref not in self.catalog:
            raise ValueError(f"Artifact {artifact_ref} not found in catalog.")
            
        meta = self.catalog[artifact_ref]
        # Account for the 2-character sharding structure we implemented
        hash_hex = artifact_ref.replace("sha256:", "")
        shard = hash_hex[:2]
        
        # In a real trace, the path is already stored in the catalog meta,
        # but we reconstruct it safely here just in case.
        file_path = self.run_dir / "artifacts" / shard / f"{hash_hex}.npy"
        
        if not file_path.exists():
            # Fallback to older non-sharded path if needed
            file_path = self.run_dir / meta["file_path"]
            
        return np.load(file_path, allow_pickle=True)

    def calculate_trajectory_variance(self) -> Dict[str, float]:
        """
        Analyzes the variance of all execution outcomes recorded in the trace.
        A variance approaching zero indicates the ansatz is in a Barren Plateau.
        """
        # 1. Query the Parquet table for all COMPLETED spans that have an outcome
        df = self.trace_table.to_pandas()
        valid_spans = df[(df["status"] == "COMPLETED") & (df["outcome_ref"].notnull())]
        
        if valid_spans.empty:
            return {"variance": 0.0, "num_evaluations": 0, "status": "Insufficient Data"}

        outcomes = []
        
        # 2. Iterate through the trace and load the physical physics data
        for _, row in valid_spans.iterrows():
            try:
                data = self._load_numpy_artifact(row["outcome_ref"])
                # Flatten the data in case of complex tensor returns
                # We want the variance across the entire observable trajectory
                outcomes.extend(np.ravel(data).tolist())
            except Exception as e:
                print(f"Warning: Failed to load artifact {row['outcome_ref']}: {e}")

        if not outcomes:
            return {"variance": 0.0, "num_evaluations": 0, "status": "No valid numeric outcomes"}

        outcomes_array = np.array(outcomes, dtype=float)
        
        # 3. Calculate statistical diagnostics
        variance = float(np.var(outcomes_array))
        std_dev = float(np.std(outcomes_array))
        
        # Realistic heuristic threshold for a 10-epoch Barren Plateau detection
        status = "Trainable" if variance > 0.005 else "Barren Plateau Detected"

        return {
            "variance": variance,
            "std_dev": std_dev,
            "num_evaluations": len(outcomes_array),
            "status": status
        }