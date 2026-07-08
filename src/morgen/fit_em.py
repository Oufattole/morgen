import torch
import logging
from importlib.resources import files
from pathlib import Path

import hydra
from hydra.utils import instantiate
from meds_torchdata import MEDSTorchDataConfig
from omegaconf import DictConfig
from tqdm.auto import tqdm, trange
import time

# Import OmegaConf Resolvers

logger = logging.getLogger(__name__)

CONFIGS = files("morgen") / "configs"

MEDSTorchDataConfig.add_to_config_store("datamodule/config")

import logging

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

# Import OmegaConf Resolvers
from .utils import (
    gpus_available,
    hash_based_seed,
    int_prod,
    is_mlflow_logger,
    num_cores,
    num_gpus,
    oc_min,
    resolve_generation_context_size,
    save_resolved_config,
    sub,
)
from lightning.pytorch import seed_everything
from scipy.sparse import csr_array
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import tempfile


import numpy as np
import polars as pl
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
import os
from pyspark import StorageLevel

def kmeans_binary_init(pl_path: Path, X_train_csr, k, alpha=1e-4):
    # Ensure data is binary (0 or 1)
    # X_train_csr = (X_train_csr > 0).astype(np.float32) 
    start_time = time.time()
    total_threads = 64
    # 1. Force the partition count to 128 (your total logical threads)
    # 2. Persist to memory to ensure it doesn't re-calculate from 4 blocks later
    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.driver.memory", "100g") \
        .config("spark.executor.memory", "100g") \
        .config("spark.memory.fraction", "0.9") \
        .config("spark.default.parallelism", str(total_threads)) \
        .config("spark.sql.shuffle.partitions", str(total_threads)) \
        .config("spark.driver.cores", str(total_threads)) \
        .config("spark.scheduler.mode", "FAIR") \
        .config("spark.locality.wait", "0s") \
        .getOrCreate()
        

    to_vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
    df = spark.read.parquet(str(pl_path)) \
                .withColumn("features", to_vector_udf("values")) \
                .select("features").cache()
    df = df.repartition(total_threads).persist(StorageLevel.MEMORY_ONLY)

    # 3. CRITICAL: The 'count()' action MUST happen here to materialize 
    # the 128 partitions across your dual CPUs
    # df.count()
    duration_spark = time.time() - start_time
    print(f"Spark DataFrame Creation Time: {duration_spark:.2f}s")
    start_time = time.time()
    
    # Use Cosine distance for binary data initialization
    kmeans = KMeans(k=k, initMode="k-means||", maxIter=1, seed=np.random.randint(0, 1000)).setInitSteps(1).setDistanceMeasure("cosine")
    
    model = kmeans.fit(df)
    
    kmeans_fit_duration = time.time() - start_time
    print(f"Spark KMeans Fitting Time: {kmeans_fit_duration:.2f}")
    # Theta: The centroids are the Bernoulli success probabilities
    theta = np.array(model.clusterCenters())
    
    # CRITICAL: BMM Smoothing
    # Clip values to [alpha, 1-alpha] to prevent log(0)
    theta = np.clip(theta, alpha, 1 - alpha)
    
    # Pi: Standard cluster assignment count
    counts = model.transform(df).groupBy("prediction").count().collect()
    pi = np.zeros(k)
    for row in counts: pi[row['prediction']] = row['count']
    pi = (pi + alpha) / (pi.sum() + k * alpha) # Smoothed Pi
    
    return theta, pi

logger = logging.getLogger(__name__)

class BernoulliEM:
    def __init__(self, k, alpha=1e-4, device=torch.device("cuda")):
        self.k = k
        self.alpha = alpha
        self.theta = None
        self.pi = None
        self.device = device
        self.torch_dtype = torch.float64

    def fit(self, pl_path, X_train, X_val, max_iter=5, tol=1e-4, batch_size=2048, init_method="kmeans"):
        with torch.no_grad():
            device = self.device
            torch_dtype = self.torch_dtype
            
            start_init = time.time()
            N, D = X_train.shape
            if init_method == "kmeans":
                theta0_np, pi0_np = kmeans_binary_init(
                    pl_path, X_train, k=self.k, alpha=self.alpha,
                )
                self.theta = torch.from_numpy(theta0_np).to(device=device, dtype=torch_dtype)
                self.pi = torch.from_numpy(pi0_np).to(device=device, dtype=torch_dtype)
            else:
                rng = np.random.default_rng()
                self.theta = torch.from_numpy(X_train[rng.choice(N, self.k, replace=False)].toarray().astype(float)).to(device=device, dtype=torch_dtype)
                self.theta = torch.clip(self.theta, self.alpha, 1 - self.alpha)
                self.pi = torch.full((self.k,), 1.0 / self.k, device=device, dtype=torch_dtype)
            duration_init = time.time() - start_init
            
            best_theta = self.theta
            best_pi = self.pi
            best_metrics = dict(macro_auroc=-torch.inf)

            prev_ll = -torch.inf

            for i in range(max_iter):
                # --- E-STEP: precompute weights/bias ---
                log_theta = torch.log(self.theta)
                log_one_minus_theta = torch.log1p(-self.theta)
                weights = log_theta - log_one_minus_theta                # (K, D) 
                bias = log_one_minus_theta.sum(dim=1) + torch.log(self.pi) # (K,)   

                # Accumulate sufficient stats over blocks (full-batch EM)
                nj = torch.zeros((self.k,), device=device, dtype=torch_dtype)          # (K,)
                counts = torch.zeros((self.k, D), device=device, dtype=torch_dtype) # (K, D)
                ll_sum = 0.0

                start_e = time.time()
                for start in trange(0, N, batch_size, desc="E-Step Batch Progress"):
                    end = min(start + batch_size, N)
                    Xb = torch.from_numpy(X_train[start:end].toarray()).to(device=device, dtype=torch_dtype)   # sparse slice
                    B = end - start

                    # logits = Xb @ weights.T + bias   (B, K)
                    logits = Xb @ weights.T
                    logits += bias                         # in-place broadcast add

                    # Softmax
                    max_logits = torch.max(logits, dim=1, keepdims=True).values  # (B,1)
                    logits -= max_logits
                    torch.exp(logits, out=logits)
                    sum_exp = torch.sum(logits, dim=1, keepdims=True)     # (B,1)
                    logits /= sum_exp

                    responsibilities = logits  # (B,K)

                    # accumulate nj
                    nj += responsibilities.sum(dim=0)

                    # accumulate counts: counts += R^T X
                    # use (X^T R)^T to leverage sparse @ dense
                    xtr = (Xb.T @ responsibilities)        # (D, K)
                    counts += xtr.T          # (K, D)

                    # ll contribution for this block: sum_i logsumexp(raw_logits_i)
                    ll_sum += float(torch.sum(max_logits + torch.log(sum_exp)))
                duration_e = time.time() - start_e

                # --- M-STEP ---
                start_m = time.time()
                self.theta = (counts + self.alpha) / (nj[:, None] + 2 * self.alpha)
                self.pi = nj / N
                duration_m = time.time() - start_m

                current_ll = ll_sum / N
                if abs(current_ll - prev_ll) < tol:
                    break
                prev_ll = current_ll
                start_eval = time.time()
                metrics, clusters = self.compute_all_metrics(X_val)
                duration_eval = time.time() - start_eval
                auroc = metrics["macro_auroc"]
                print(f"Init time: {duration_init:.4f}s | E-step: {duration_e:.4f}s | M-step: {duration_m:.4f}s | Eval time: {duration_eval:.4f}s | AUROC {auroc:.4f}")
                if metrics["macro_auroc"] > best_metrics["macro_auroc"]:
                    best_theta = self.theta
                    best_pi = self.pi
                    best_metrics = metrics

            self.theta = best_theta
            self.pi = best_pi
            return metrics, clusters

    def compute_all_metrics(self, X_val):
        """Computes AUROC and AUPRC for every single column (feature) and for the specific groups in
        special_codes."""
        val_clusters = self.transform(X_val)
        # Predicted probabilities for every feature based on assigned cluster
        val_probs = self.theta.cpu().numpy()[val_clusters]

        N, D = X_val.shape
        aurocs = []
        auprcs = []

        # 1. Compute for ALL codes (Macro metrics)
        for d in range(D):
            if isinstance(X_val, csr_array):
                y_true = X_val[:, d].toarray().ravel()
            else:
                y_true = X_val[:, d]
            y_prob = val_probs[:, d]

            # Skip codes that never appear in the val set to avoid errors
            if len(np.unique(y_true)) > 1:
                aurocs.append(roc_auc_score(y_true, y_prob))
                auprcs.append(average_precision_score(y_true, y_prob))

        metrics = {
            "macro_auroc": np.mean(aurocs) if aurocs else 0.0,
            "macro_auprc": np.mean(auprcs) if auprcs else 0.0,
        }
        return metrics, val_probs

    def transform(self, X):
        if isinstance(X, csr_array):
            X = X.toarray()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).to(device=self.device, dtype=self.torch_dtype)
        weights = torch.log(self.theta) - torch.log1p(-self.theta)
        bias = torch.log1p(-self.theta).sum(dim=1) + torch.log(self.pi)
        logits = X @ weights.T + bias
        return torch.argmax(logits, dim=1).cpu().numpy()


def plot_auroc_grid(history_list, output_path):
    """Generates a 4x4 grid of AUROC curves for special codes."""
    # Convert list of dicts to a tidy dataframe
    df = pl.from_dicts(history_list).to_pandas()
    code_names = [c for c in df.columns if c != "iteration"]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    for i, name in enumerate(code_names):
        ax = axes[i]
        sns.lineplot(data=df, x="iteration", y=name, ax=ax, marker="o", color="teal")
        ax.set_title(f"AUROC: {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.set_ylim(0.5, 1.0)  # Standard AUROC range

    # Hide unused 16th subplot if you only have 15 codes
    if len(code_names) < 16:
        for j in range(len(code_names), 16):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def get_full_numpy_dataset(dataloader):
    """Collects all binary histograms from the MEDS dataloader into one array."""
    all_x = []
    logger.info("Loading full dataset into memory...")
    for batch in tqdm(dataloader, desc="Loading Batches"):
        # batch.histograms is usually (Batch, Vocab)
        # We only take non-empty windows and binarize them
        x = (batch.histograms[batch.histograms.sum(-1) != 0] > 0).float().cpu().numpy()
        if x.shape[0] > 0:
            all_x.append(x)
    return np.concatenate(all_x, axis=0)




import logging
from pathlib import Path
import numpy as np
import polars as pl
import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from lightning.pytorch import seed_everything
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(CONFIGS), config_name="_train_quantizer")
def train_analytic_quantizer(cfg: DictConfig):
    import multiprocessing

    multiprocessing.set_start_method("forkserver", force=True)
    seed_everything(0)
    output_dir = Path(cfg.output_dir)
    
    # We toggle this based on whether we are training new phenotypes 
    # or verifying a hot-swappable PyTorch model for the paper.
    do_fit = True 

    # Shared Metadata Loading (Minimal)
    datamodule = instantiate(cfg.datamodule)
    tensorized_dir = Path(datamodule.config.tensorized_cohort_dir)
    code_metadata_path = tensorized_dir / "metadata" / "codes.parquet"
    code_metadata_df = pl.read_parquet(code_metadata_path, columns=["code", "code/vocab_index"])
    
    vocab_size = cfg.quantizer_config.vocab_size
    special_codes = {}
    for name, regex in cfg.special_codes.items():
        idxs = code_metadata_df.filter(pl.col("code").str.contains(regex))["code/vocab_index"].to_list()
        special_codes[name] = [i for i in idxs if 0 <= i < vocab_size]

    if do_fit:
        # --- TRAINING BRANCH ---
        if cfg.cache_dir is None:
            raise ValueError("Cache directory is not set. Please set `cache_dir` in the config.")
        
        cache_dir = Path(cfg.cache_dir)
        if (cache_dir / "X_train.npy").exists() and (cache_dir / "X_val.npy").exists():
            print(f"--- Loading Cached training and validation data from {cfg.cache_dir} ---")
            X_train = np.load(cache_dir / "X_train.npy")
            X_val = np.load(cache_dir / "X_val.npy")
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"--- Caching training and validation data to {cfg.cache_dir} ---")
            datamodule.setup("fit")
            X_train = get_full_numpy_dataset(datamodule.train_dataloader())
            X_val = get_full_numpy_dataset(datamodule.val_dataloader())
            np.save(cache_dir /  "X_val.npy", X_val)
            np.save(cache_dir /  "X_train.npy", X_train)
            print(f"--- Saved training data to {cfg.cache_dir} ---")

        pl_path = cache_dir / "train.parquet"
        if not pl_path.exists():
            print(f"--- Saving training data to Polars Parquet at {pl_path} ---")
            start_time = time.time()
            # Polars: Concat into list for Spark
            pl.from_numpy(X_train.astype(np.float32)) \
                .select(pl.concat_list(pl.all()).alias("values")) \
                .unique() \
                .write_parquet(pl_path)
            duration_pl = time.time() - start_time
            print(f"Polars Parquet Save Time: {duration_pl:.2f}s")
        X_train = csr_array(X_train)
        
        print(f"--- Training on {X_train.shape[0]} examples with {X_train.shape[1]} features. ---")

        n_seeds = 3
        selection_metric = "macro_auroc"
        if selection_metric == "loss": raise NotImplementedError("Loss metric not implemented yet.")
        K = cfg.quantizer_config.n_embeddings

        best_model = None
        best_score = float("inf") if selection_metric == "loss" else -float("inf")
        best_metrics_summary = None
        best_special_results = None

        for seed in range(n_seeds):
            logger.info(f"--- Running EM with Seed {seed} ---")
            np.random.seed(seed)
            model = BernoulliEM(k=K)
            start_fit = time.time()
            global_metrics, clusters = model.fit(pl_path, X_train.astype(np.float32), X_val)
            duration_fit = time.time() - start_fit
            start_log = time.time()
            percent_used = 100 * len(np.unique(clusters)) / K

            print(f"--- AUROC: {global_metrics['macro_auroc']:.4f} | AUPRC: {global_metrics['macro_auprc']:.4f} | Percent Used: {percent_used:.2f}% ---\n")
            duration_log = time.time() - start_log
            special_aurocs, special_auprcs, special_results = [], [], {}

            current_metrics = {
                "macro_auroc": global_metrics["macro_auroc"],
                "macro_auprc": global_metrics["macro_auprc"],
                "special_macro_auroc": np.mean(special_aurocs) if special_aurocs else 0.0,
                "special_macro_auprc": np.mean(special_auprcs) if special_auprcs else 0.0,
                "percent_used": percent_used
            }

            score = current_metrics[selection_metric]
            if score > best_score:
                best_score, best_model, best_metrics_summary, best_special_results = score, model, current_metrics, special_results
            print(f"--- Fit Time: {duration_fit:.2f}s | Log Time: {duration_log:.2f}s ---\n")

        # Output & Save
        print("\nCOPY INTO GOOGLE SHEETS:\nMetric Group,AUROC,AUPRC")
        print(f"ALL CODES (Macro),{best_metrics_summary['macro_auroc']:.4f},{best_metrics_summary['macro_auprc']:.4f},{best_metrics_summary['percent_used']:.2f}")
        print(f"SPECIAL CODES (Macro),{best_metrics_summary['special_macro_auroc']:.4f},{best_metrics_summary['special_macro_auprc']:.4f}\n---,---,---")
        for name, res in best_special_results.items():
            print(f"{name},{res['auroc']:.4f},{res['auprc']:.4f}")

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "cluster_theta.npy", best_model.theta.cpu().numpy())
        np.save(output_dir / "cluster_pi.npy", best_model.pi.cpu().numpy())
        logger.info(f"Best model parameters saved to {output_dir}")

    # --- VERIFICATION BRANCH ---
    # from morgen.model.em_quantizer import EMQuantizer
    
    # # Only load Val data for verification
    # datamodule.setup("fit")
    # X_val = get_full_numpy_dataset(datamodule.val_dataloader())
    # X_val_torch = torch.from_numpy(X_val).float()

    # logger.info(f"Loading best parameters from {output_dir} for PyTorch verification...")
    # theta = np.load(output_dir / "cluster_theta.npy")
    # pi = np.load(output_dir / "cluster_pi.npy")
    
    # torch_model = EMQuantizer(theta=torch.from_numpy(theta).float(), pi=torch.from_numpy(pi).float())
    # torch_model.eval()

    # # Build equivalent NumPy model for comparison
    # numpy_model = BernoulliEM(k=cfg.quantizer_config.n_embeddings)
    # numpy_model.theta, numpy_model.pi = torch.from_numpy(theta).to(device=numpy_model.device, dtype=numpy_model.torch_dtype), torch.from_numpy(pi).to(device=numpy_model.device, dtype=numpy_model.torch_dtype)

    # with torch.no_grad():
    #     logits = torch_model.encode(X_val_torch)
    #     _, _, torch_indices = torch_model.quantize(logits)
    #     torch_val_probs = torch_model.theta[torch_indices].cpu().numpy()

    # logger.info("--- PyTorch EMQuantizer Alignment Verification ---")
    # verification_passed = True
    # for name, indices in special_codes.items():
    #     if not indices: continue
    #     y_true = (X_val[:, indices].sum(axis=1) > 0).astype(int)
        
    #     # NumPy scores
    #     val_clusters_np = numpy_model.transform(X_val)
    #     y_prob_np = 1 - np.prod(1 - theta[val_clusters_np][:, indices], axis=1)
    #     auc_np = roc_auc_score(y_true, y_prob_np)

    #     # PyTorch scores
    #     y_prob_torch = 1 - np.prod(1 - torch_val_probs[:, indices], axis=1)
    #     auc_torch = roc_auc_score(y_true, y_prob_torch)

    #     diff = abs(auc_np - auc_torch)
    #     status = "PASSED" if diff < 3e-2 else "FAILED"
    #     logger.info(f"{name:.<20} NP: {auc_np:.4f} | Torch: {auc_torch:.4f} | {status}")
    #     if status == "FAILED": verification_passed = False

    # if verification_passed:
    #     logger.info("Verification SUCCESS: PyTorch EMQuantizer is mathematically aligned with the Bernoulli EM solution.")
    # else:
    #     logger.error("Verification FAILURE: Discrepancy detected between implementations.")
