import pandas as pd
import numpy as np
from ETIA.CausalLearning import CausalLearner
import traceback

# 1) Load your data (rows=samples, cols=variables; no missing values for FGES)
# Use an existing dataset from the repo
data_path = "causal_meta_dataset/dataset_000_config_000/data.npy"
data = np.load(data_path)
if data.ndim != 2:
    raise ValueError("Expected a 2D array from data.npy")
num_cols = data.shape[1]
columns = [f"X{i}" for i in range(num_cols)]
df = pd.DataFrame(data, columns=columns)

# 2) Pick data type to choose score:
#    - "continuous" -> SemBic
#    - "discrete"   -> BDeu
#    - "mixed"      -> CG
data_type = "continuous"  # change to "discrete" or "mixed" if needed

# 3) FGES hyperparameters
fges_params = {
    # Common ones:
    "penalty_discount": 2.0,     # lambda (higher → sparser)
    "max_degree": -1,            # -1 means unlimited
    "faithfulness_assumed": True,
    # Discrete-specific (if data_type == "discrete"):
    # "bdeu_ess": 10.0,          # equivalent sample size
}

def main() -> None:
    print("[ETIA TEST] Data shape:", df.shape)
    # 4) Create learner fixed to FGES
    learner = CausalLearner(
        data=df,
        algorithm="fges",            # force FGES
        data_type=data_type,          # drives score choice
        params=fges_params
    )

    # 5) Learn the graph
    print("[ETIA TEST] Running FGES...")
    results = learner.learn_model()
    print("[ETIA TEST] FGES completed.")

    # Results object commonly contains:
    # - results["graph"]: learned graph object
    # - results["edges"]: list of edges (u, v, orientation)
    # - results["adj_matrix"]: adjacency matrix (if provided)
    # - results["meta"]: runtime/meta info
    graph = results.get("graph")
    edges = results.get("edges", [])
    print("Num edges:", len(edges))
    for e in edges[:20]:
        print(e)

    # Optional: export adjacency matrix if available
    adj = results.get("adj_matrix")
    if adj is not None:
        import numpy as np
        np.savetxt("fges_adj_matrix.csv", adj, delimiter=",", fmt="%d")
        print("Saved adjacency matrix to fges_adj_matrix.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ETIA TEST] ERROR:", str(e))
        traceback.print_exc()