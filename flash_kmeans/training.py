"""K-Means training utilities used by TensorVS and Flash KMeans.

This module owns the dataset sampling, initialization, restart selection, and
both K-Means variants used by TensorVS:

* ``algorithm="lloyd"``: traditional nearest-center Lloyd iterations;
* ``algorithm="balanced"``: the cuVS-style rebalancing EM loop.  Coarse
  IVF training can additionally use cuVS's hierarchical balanced variant;
  PQ codebooks use the flat balanced variant, matching cuVS.

The low-level assignment and centroid-update kernels remain in the regular
Flash KMeans modules. Keeping this orchestration here means consumers such as
TensorVS do not need a second private K-Means implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import torch

from .kmeans_large import kmeans_largeN, kmeans_largeN_assign
from .torch_fallback import (
    batch_kmeans_Euclid_torch_native,
    euclid_assign_torch_native_chunked,
)

try:
    from .assign_euclid_triton import euclid_assign_triton
    from .kmeans_triton_impl import batch_kmeans_Euclid
except Exception:  # pragma: no cover - exercised without Triton installed
    euclid_assign_triton = None
    batch_kmeans_Euclid = None


_STAGE_SEED_STRIDE = 1_000_003
_MAX_TORCH_SEED = (1 << 63) - 1
_KMEANS_CPU_CHUNK_BYTES = 256 << 20
_INERTIA_CHUNK_BYTES = 64 << 20
_MAX_KMEANS_CPU_CHUNK_ROWS = 1 << 20
_CUVS_BALANCED_PRIMES = (
    29,
    71,
    113,
    173,
    229,
    281,
    349,
    409,
    463,
    541,
    601,
    659,
    733,
    809,
    863,
    941,
    1013,
    1069,
    1151,
    1223,
    1291,
    1373,
    1451,
    1511,
    1583,
    1657,
    1733,
    1811,
    1889,
    1987,
    2053,
    2129,
    2213,
    2287,
    2357,
    2423,
    2531,
    2617,
    2687,
    2741,
)

# Match Faiss ClusteringParameters defaults for the Euclidean IVF use case.
FAISS_DEFAULT_NITER = 25
FAISS_DEFAULT_NREDO = 1
FAISS_DEFAULT_MIN_POINTS_PER_CENTROID = 39
FAISS_DEFAULT_MAX_POINTS_PER_CENTROID = 256
FAISS_DEFAULT_SEED = 1234

KMEANS_INIT_RANDOM = "random"
KMEANS_INIT_KMEANS_PLUS_PLUS = "kmeans++"
KMEANS_ALGORITHM_LLOYD = "lloyd"
KMEANS_ALGORITHM_BALANCED = "balanced"
KMEANS_PLUS_PLUS_MAX_INIT_POINTS = 65_536


@dataclass(frozen=True)
class KMeansTrainingResult:
    """Best final-label K-Means result selected from multiple restarts."""

    labels: torch.Tensor
    centroids: torch.Tensor
    inertia: float
    seed: int


def derive_kmeans_seed(base_seed: int, stage: int, restart: int) -> int:
    """Derive stable, non-overlapping seeds for stages and restarts."""
    if stage < 0:
        raise ValueError(f"stage must be non-negative, got {stage}")
    if restart < 0:
        raise ValueError(f"restart must be non-negative, got {restart}")
    return (int(base_seed) + stage * _STAGE_SEED_STRIDE + restart) % _MAX_TORCH_SEED


def select_training_indices(
    num_rows: int,
    *,
    k: int,
    max_points_per_centroid: int | None,
    trainset_fraction: float | None,
    seed: int,
    stage: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Select deterministic rows for a K-Means training subset."""
    if num_rows <= 0:
        raise ValueError(f"num_rows must be positive, got {num_rows}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if trainset_fraction is not None:
        trainset_fraction = float(trainset_fraction)
        if not 0.0 < trainset_fraction <= 1.0:
            raise ValueError(
                "trainset_fraction must be in (0, 1] when provided, "
                f"got {trainset_fraction}"
            )
        max_training_rows = min(
            int(num_rows),
            max(int(k), math.ceil(int(num_rows) * trainset_fraction)),
        )
    elif max_points_per_centroid is None:
        max_training_rows = int(num_rows)
    else:
        if max_points_per_centroid <= 0:
            raise ValueError(
                "max_points_per_centroid must be positive when provided, "
                f"got {max_points_per_centroid}"
            )
        max_training_rows = min(int(num_rows), int(k) * int(max_points_per_centroid))

    target_device = torch.device(device)
    if max_training_rows >= num_rows:
        return torch.arange(int(num_rows), dtype=torch.int64, device=target_device)

    # Keep sampling independent of the target device so CPU and CUDA builds
    # receive the same subset for a given seed and stage.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_kmeans_seed(seed, stage, 0))
    sample_indices = torch.randperm(
        int(num_rows),
        generator=generator,
        device="cpu",
    )[:max_training_rows]
    if target_device.type != "cpu":
        sample_indices = sample_indices.to(device=target_device)
    return sample_indices


def _rows_for_byte_budget(
    d: int,
    *,
    byte_budget: int,
    max_rows: int | None = None,
) -> int:
    """Return a positive row chunk whose FP32 payload fits a byte budget."""
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    rows = max(1, int(byte_budget) // (int(d) * torch.float32.itemsize))
    if max_rows is not None:
        rows = min(rows, int(max_rows))
    return rows


def _kmeans_cpu_chunk_rows(d: int) -> int:
    """Bound CPU-to-GPU K-Means chunks by bytes and row count."""
    return _rows_for_byte_budget(
        d,
        byte_budget=_KMEANS_CPU_CHUNK_BYTES,
        max_rows=_MAX_KMEANS_CPU_CHUNK_ROWS,
    )


def _select_training_data(
    data: torch.Tensor,
    *,
    k: int,
    max_points_per_centroid: int | None,
    trainset_fraction: float | None,
    seed: int,
    stage: int,
) -> torch.Tensor:
    """Select the deterministic subset used by one K-Means stage."""
    sample_indices = select_training_indices(
        int(data.shape[0]),
        k=k,
        max_points_per_centroid=max_points_per_centroid,
        trainset_fraction=trainset_fraction,
        seed=seed,
        stage=stage,
        device=data.device,
    )
    if sample_indices.numel() >= data.shape[0]:
        return data
    return data.index_select(0, sample_indices)


def _random_initial_centroids(
    data: torch.Tensor,
    *,
    k: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Choose uniformly random, non-repeating initial centers."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % _MAX_TORCH_SEED)
    indices = torch.randperm(
        int(data.shape[0]),
        generator=generator,
        device="cpu",
    )[:k]
    if data.device.type != "cpu":
        indices = indices.to(device=data.device)
    return data.index_select(0, indices).to(
        device=device,
        dtype=torch.float32,
        copy=False,
    ).contiguous()


def _kmeans_plus_plus_training_data(
    data: torch.Tensor,
    *,
    k: int,
    seed: int,
    stage: int,
) -> torch.Tensor:
    """Bound the candidate rows used by scalable K-Means++ seeding."""
    num_rows = int(data.shape[0])
    init_rows = min(num_rows, max(3 * int(k), KMEANS_PLUS_PLUS_MAX_INIT_POINTS))
    if init_rows >= num_rows:
        return data

    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_kmeans_seed(seed, stage, 0))
    sample_indices = torch.randperm(
        num_rows,
        generator=generator,
        device="cpu",
    )[:init_rows]
    if data.device.type != "cpu":
        sample_indices = sample_indices.to(device=data.device)
    return data.index_select(0, sample_indices)


def _squared_distance_to_centroid(
    data: torch.Tensor,
    centroid: torch.Tensor,
) -> torch.Tensor:
    """Compute row-wise squared L2 distances without a large temporary."""
    if data.ndim != 2 or centroid.ndim != 1 or data.shape[1] != centroid.shape[0]:
        raise ValueError(
            "Expected data [n, d] and centroid [d], got "
            f"{tuple(data.shape)} and {tuple(centroid.shape)}"
        )
    chunk_rows = _rows_for_byte_budget(
        int(data.shape[1]),
        byte_budget=_KMEANS_CPU_CHUNK_BYTES,
        max_rows=int(data.shape[0]),
    )
    distances = torch.empty((data.shape[0],), dtype=torch.float32, device=data.device)
    centroid_norm = torch.dot(centroid, centroid)
    for start in range(0, int(data.shape[0]), chunk_rows):
        end = min(start + chunk_rows, int(data.shape[0]))
        data_chunk = data[start:end]
        distances[start:end] = (
            (data_chunk * data_chunk).sum(dim=1)
            + centroid_norm
            - 2.0 * torch.mv(data_chunk, centroid)
        ).clamp_min_(0.0)
    return distances


def _kmeans_plus_plus_initial_centroids(
    data: torch.Tensor,
    *,
    k: int,
    seed: int,
    stage: int,
    device: torch.device,
) -> torch.Tensor:
    """Choose initial centers with scalable K-Means++ seeding."""
    candidate_data = _kmeans_plus_plus_training_data(
        data,
        k=k,
        seed=seed,
        stage=stage,
    ).to(device=device, dtype=torch.float32, copy=False).contiguous()
    num_rows = int(candidate_data.shape[0])
    if num_rows < k:
        raise ValueError(f"KMeans++ requires at least k={k} rows, got {num_rows}")

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) % _MAX_TORCH_SEED)
    centers = torch.empty(
        (k, candidate_data.shape[1]),
        dtype=torch.float32,
        device=device,
    )
    first_index = torch.randint(num_rows, (1,), generator=generator, device=device)
    centers[0] = candidate_data[first_index[0]]
    min_distances = _squared_distance_to_centroid(candidate_data, centers[0])

    for center_idx in range(1, int(k)):
        total_distance = min_distances.sum()
        if not torch.isfinite(total_distance) or total_distance <= 0:
            next_index = torch.randint(num_rows, (1,), generator=generator, device=device)[0]
        else:
            next_index = torch.multinomial(
                min_distances / total_distance,
                num_samples=1,
                replacement=True,
                generator=generator,
            )[0]
        centers[center_idx] = candidate_data[next_index]
        min_distances = torch.minimum(
            min_distances,
            _squared_distance_to_centroid(candidate_data, centers[center_idx]),
        )
    return centers.contiguous()


def _initial_centroids(
    data: torch.Tensor,
    *,
    k: int,
    seed: int,
    stage: int,
    device: torch.device,
    init_method: str,
) -> torch.Tensor:
    normalized_method = str(init_method).strip().lower().replace("_", "-")
    if normalized_method in {"kmeans++", "k-means++", "kmeans-plus-plus"}:
        return _kmeans_plus_plus_initial_centroids(
            data,
            k=k,
            seed=seed,
            stage=stage,
            device=device,
        )
    if normalized_method == KMEANS_INIT_RANDOM:
        return _random_initial_centroids(data, k=k, seed=seed, device=device)
    raise ValueError(
        "Unsupported KMeans init_method: "
        f"{init_method!r}; expected 'kmeans++' or 'random'"
    )


def _assign_kmeans_labels(
    data: torch.Tensor,
    centroids: torch.Tensor,
    *,
    d: int,
    use_triton: bool,
    device: torch.device,
) -> torch.Tensor:
    """Assign rows to fixed centers through Flash KMeans backends."""
    if (
        use_triton
        and device.type == "cuda"
        and euclid_assign_triton is not None
        and data.device.type == "cpu"
        and data.shape[0] > _kmeans_cpu_chunk_rows(d)
    ):
        return kmeans_largeN_assign(
            data,
            centroids,
            BLOCK_N=_kmeans_cpu_chunk_rows(d),
            device=device,
            dtype=torch.float32,
        )

    data_device = data.to(device=device, dtype=torch.float32, copy=False).contiguous()
    centroids_device = centroids.to(
        device=device,
        dtype=torch.float32,
        copy=False,
    ).contiguous()
    x = data_device.unsqueeze(0)
    x_sq = (x * x).sum(dim=-1)
    if use_triton and device.type == "cuda" and euclid_assign_triton is not None:
        labels = euclid_assign_triton(x, centroids_device.unsqueeze(0), x_sq)
    else:
        labels = euclid_assign_torch_native_chunked(
            x,
            centroids_device.unsqueeze(0),
            x_sq,
            chunk_size_N=_kmeans_cpu_chunk_rows(d),
            chunk_size_K=1024,
        )
    return labels.squeeze(0)


def _run_kmeans_once(
    data: torch.Tensor,
    *,
    d: int,
    k: int,
    niter: int,
    use_triton: bool,
    device: torch.device,
    seed: int,
    stage: int,
    init_method: str,
) -> torch.Tensor:
    """Run one traditional Lloyd K-Means restart."""
    init_centroids = _initial_centroids(
        data,
        k=k,
        seed=seed,
        stage=stage,
        device=device,
        init_method=init_method,
    )
    if (
        use_triton
        and device.type == "cuda"
        and euclid_assign_triton is not None
        and data.device.type == "cpu"
        and data.shape[0] > _kmeans_cpu_chunk_rows(d)
    ):
        _, centroids = kmeans_largeN(
            data,
            k,
            max_iters=niter,
            tol=0.0,
            init_centroids=init_centroids,
            device=device,
            dtype=torch.float32,
            BLOCK_N=_kmeans_cpu_chunk_rows(d),
        )
        return centroids.to(dtype=torch.float32, device=device, copy=False)

    data_device = data.to(device=device, dtype=torch.float32, copy=False).contiguous()
    x = data_device.unsqueeze(0)
    init_batch = init_centroids.unsqueeze(0)
    if use_triton and device.type == "cuda" and batch_kmeans_Euclid is not None:
        _, centroids, _ = batch_kmeans_Euclid(
            x,
            k,
            max_iters=niter,
            tol=0.0,
            init_centroids=init_batch,
            verbose=False,
        )
    else:
        _, centroids, _ = batch_kmeans_Euclid_torch_native(
            x,
            k,
            max_iters=niter,
            tol=0.0,
            init_centroids=init_batch,
            verbose=False,
            chunk_size_N=_kmeans_cpu_chunk_rows(d),
            chunk_size_K=1024,
        )
    return centroids.squeeze(0).to(dtype=torch.float32, device=device, copy=False)


def _assigned_inertia(
    data: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int | None = None,
) -> float:
    """Compute assigned squared L2 cost without an N-by-K temporary."""
    if chunk_size is None:
        chunk_size = _rows_for_byte_budget(
            int(data.shape[1]),
            byte_budget=_INERTIA_CHUNK_BYTES,
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    total = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, data.shape[0], chunk_size):
        end = min(start + chunk_size, data.shape[0])
        data_chunk = data[start:end].to(dtype=torch.float32, device=device, copy=False)
        label_chunk = labels[start:end].to(dtype=torch.int64, device=device, copy=False)
        residual = centroids[label_chunk]
        residual.sub_(data_chunk).square_()
        total.add_(residual.sum(dtype=torch.float64))
    return float(total.item())


def _recompute_centers_and_sizes(
    data: torch.Tensor,
    labels: torch.Tensor,
    *,
    k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cluster means and sizes for the balanced EM loop."""
    data_device = data.to(device=device, dtype=torch.float32, copy=False).contiguous()
    labels_device = labels.to(device=device, dtype=torch.int64, copy=False)
    centers = torch.zeros((k, data_device.shape[1]), dtype=torch.float32, device=device)
    centers.scatter_add_(
        0,
        labels_device.unsqueeze(1).expand(-1, data_device.shape[1]),
        data_device,
    )
    sizes = torch.bincount(labels_device, minlength=k)
    centers.div_(sizes.clamp_min(1).to(dtype=centers.dtype).unsqueeze(1))
    return centers, sizes


def _adjust_balanced_centers(
    data: torch.Tensor,
    labels: torch.Tensor,
    sizes: torch.Tensor,
    centers: torch.Tensor,
    *,
    balancing_threshold: float,
    prime: int,
) -> bool:
    """Pull undersized centers using cuVS's deterministic source-row walk."""
    n_rows = int(data.shape[0])
    n_clusters = int(centers.shape[0])
    average = n_rows // n_clusters
    if average <= 0:
        return False

    small = torch.nonzero(
        sizes <= float(average) * float(balancing_threshold),
        as_tuple=False,
    ).flatten()
    if small.numel() == 0:
        return False

    labels_device = labels.to(device=centers.device, dtype=torch.int64, copy=False)
    # cuVS advances a coprime prime walk through the rows and accepts the
    # first row whose current cluster has at least the average size.  The
    # vectorized form below has the same row order and avoids a host/device
    # synchronization for every undersized cluster.
    offsets = torch.arange(
        1,
        n_rows + 1,
        dtype=torch.int64,
        device=centers.device,
    )
    candidate_rows = (int(prime) * offsets).remainder(n_rows)
    eligible_rows = candidate_rows[
        sizes[labels_device[candidate_rows]] >= average
    ]
    if eligible_rows.numel() < small.numel():  # pragma: no cover - averaging guarantees enough
        return False
    source_rows = eligible_rows[: small.numel()]
    source_vectors = data.index_select(0, source_rows).to(
        device=centers.device,
        dtype=centers.dtype,
        copy=False,
    )
    weights = sizes.index_select(0, small).to(dtype=centers.dtype).clamp_max(7.0)
    centers[small] = (
        centers[small] * weights.unsqueeze(1) + source_vectors
    ) / (weights.unsqueeze(1) + 1.0)
    return True


def _run_balanced_kmeans_once(
    data: torch.Tensor,
    *,
    k: int,
    niter: int,
    use_triton: bool,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run cuVS's non-hierarchical balanced K-Means builder."""
    data_device = data.to(device=device, dtype=torch.float32, copy=False).contiguous()
    n_rows = int(data_device.shape[0])
    labels = torch.arange(n_rows, device=device, dtype=torch.int64).remainder(k)
    centers, sizes = _recompute_centers_and_sizes(
        data_device,
        labels,
        k=k,
        device=device,
    )

    return _run_balanced_em(
        data_device,
        centers,
        niter=niter,
        balancing_pullback=2,
        balancing_threshold=0.25,
        use_triton=use_triton,
        device=device,
        seed=seed,
        initial_labels=labels,
        initial_sizes=sizes,
    )


def _run_balanced_em(
    data: torch.Tensor,
    centers: torch.Tensor,
    *,
    niter: int,
    balancing_pullback: int,
    balancing_threshold: float,
    use_triton: bool,
    device: torch.device,
    seed: int,
    initial_labels: torch.Tensor | None = None,
    initial_sizes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run cuVS's balancing/expectation/maximization loop from given centers.

    cuVS uses the same loop in two places: the flat builder initializes labels
    modulo ``k`` and computes centers from them, while the hierarchical IVF
    builder uses already-built fine centers and starts with an ordinary
    expectation step.  Keeping those two entry points in one implementation
    makes the balancing cadence and extra-iteration rule identical.
    """
    if niter <= 0:
        raise ValueError(f"niter must be positive, got {niter}")
    if balancing_pullback <= 0:
        raise ValueError(
            f"balancing_pullback must be positive, got {balancing_pullback}"
        )

    data_device = data.to(device=device, dtype=torch.float32, copy=False).contiguous()
    centers = centers.to(device=device, dtype=torch.float32, copy=False).contiguous()
    if initial_labels is None:
        labels = torch.zeros(data_device.shape[0], dtype=torch.int64, device=device)
        sizes = torch.zeros(centers.shape[0], dtype=torch.int64, device=device)
    else:
        labels = initial_labels.to(device=device, dtype=torch.int64, copy=False)
        if initial_sizes is None:
            _, sizes = _recompute_centers_and_sizes(
                data_device,
                labels,
                k=int(centers.shape[0]),
                device=device,
            )
        else:
            sizes = initial_sizes.to(device=device, dtype=torch.int64, copy=False)

    target_iterations = int(niter)
    balancing_counter = int(balancing_pullback)
    balancing_step = 0
    iteration = 0
    while iteration < target_iterations:
        # cuVS skips rebalancing on the first iteration.  If rebalancing moves
        # any center, it periodically adds one extra EM iteration.
        if iteration > 0 and _adjust_balanced_centers(
            data_device,
            labels,
            sizes,
            centers,
            balancing_threshold=balancing_threshold,
            prime=next(
                prime
                for prime in _CUVS_BALANCED_PRIMES[balancing_step:]
                if int(data_device.shape[0]) % prime != 0
            ),
        ):
            balancing_step = min(
                balancing_step + 1,
                len(_CUVS_BALANCED_PRIMES) - 1,
            )
            if balancing_counter >= balancing_pullback:
                balancing_counter -= balancing_pullback
                target_iterations += 1
            balancing_counter += 1

        labels = _assign_kmeans_labels(
            data_device,
            centers,
            d=int(data_device.shape[1]),
            use_triton=use_triton,
            device=device,
        )
        centers, sizes = _recompute_centers_and_sizes(
            data_device,
            labels,
            k=int(centers.shape[0]),
            device=device,
        )
        iteration += 1
    return labels, centers


def _arrange_fine_clusters(
    n_clusters: int,
    n_mesoclusters: int,
    n_rows: int,
    mesocluster_sizes: torch.Tensor,
) -> list[int]:
    """Match cuVS's proportional fine-cluster allocation."""
    sizes = [int(value) for value in mesocluster_sizes.detach().cpu().tolist()]
    fine_clusters: list[int] = []
    lists_remaining = int(n_clusters)
    nonempty_remaining = sum(size > 0 for size in sizes)
    rows_remaining = int(n_rows)

    for meso_idx in range(n_mesoclusters):
        if meso_idx < n_mesoclusters - 1:
            size = sizes[meso_idx]
            if size == 0:
                count = 0
            else:
                nonempty_remaining -= 1
                count = int(
                    float(lists_remaining * size) / float(rows_remaining) + 0.5
                )
                count = min(count, lists_remaining - nonempty_remaining)
                count = max(count, 1)
        else:
            count = lists_remaining
        fine_clusters.append(count)
        lists_remaining -= count
        rows_remaining -= sizes[meso_idx]

    if sum(fine_clusters) != n_clusters:
        raise RuntimeError(
            "Balanced KMeans fine-cluster allocation did not cover all clusters: "
            f"{sum(fine_clusters)} != {n_clusters}"
        )
    return fine_clusters


def _run_balanced_hierarchical_kmeans_once(
    data: torch.Tensor,
    *,
    k: int,
    niter: int,
    use_triton: bool,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run cuVS's hierarchical balanced K-Means builder used for IVF.

    cuVS chooses ``round(sqrt(k))`` mesoclusters, trains each mesocluster with
    the flat balanced builder, trains proportional fine clusters inside each
    mesocluster, and then fine-tunes all fine centers together.  The final
    labels are nearest-center predictions, just like cuVS's subsequent
    ``kmeans::balanced::predict`` call used while adding vectors to IVF.
    """
    data_device = data.to(device=device, dtype=torch.float32, copy=False).contiguous()
    n_rows = int(data_device.shape[0])
    n_mesoclusters = min(k, int(math.sqrt(k) + 0.5))
    n_mesoclusters = max(n_mesoclusters, 1)

    meso_labels, _meso_centers = _run_balanced_kmeans_once(
        data_device,
        k=n_mesoclusters,
        niter=niter,
        use_triton=use_triton,
        device=device,
        seed=seed,
    )
    meso_sizes = torch.bincount(meso_labels, minlength=n_mesoclusters)
    fine_cluster_counts = _arrange_fine_clusters(
        k,
        n_mesoclusters,
        n_rows,
        meso_sizes,
    )

    # cuVS caps the amount of data used per mesocluster when an intermediate
    # cluster is unexpectedly large.  This is a training-time cap only; the
    # final IVF assignment still predicts every input row.
    max_meso_size = int(meso_sizes.max().item()) if meso_sizes.numel() else 0
    balanced_max_meso_size = math.ceil(2 * n_rows / max(n_mesoclusters, 1))
    if max_meso_size > balanced_max_meso_size:
        max_meso_size = balanced_max_meso_size

    fine_centers = torch.empty(
        (k, int(data_device.shape[1])),
        dtype=torch.float32,
        device=device,
    )
    center_offset = 0
    for meso_idx, fine_count in enumerate(fine_cluster_counts):
        if fine_count == 0:
            continue
        meso_rows = torch.nonzero(meso_labels == meso_idx, as_tuple=False).flatten()
        meso_rows = meso_rows[:max_meso_size]
        if meso_rows.numel() < fine_count:
            raise RuntimeError(
                "Balanced KMeans mesocluster has fewer training rows than fine "
                f"clusters ({meso_rows.numel()} < {fine_count})"
            )
        meso_data = data_device.index_select(0, meso_rows)
        _fine_labels, local_centers = _run_balanced_kmeans_once(
            meso_data,
            k=fine_count,
            niter=niter,
            use_triton=use_triton,
            device=device,
            seed=seed,
        )
        fine_centers[center_offset : center_offset + fine_count] = local_centers
        center_offset += fine_count

    # This is intentionally shorter than the two hierarchical construction
    # stages, matching cuVS's max(n_iters / 10, 2) final fine-tuning pass.
    final_labels, final_centers = _run_balanced_em(
        data_device,
        fine_centers,
        niter=max(int(niter) // 10, 2),
        balancing_pullback=5,
        balancing_threshold=0.2,
        use_triton=use_triton,
        device=device,
        seed=seed,
    )
    return final_labels, final_centers


def _normalize_algorithm(algorithm: str) -> str:
    normalized = str(algorithm).strip().lower().replace("_", "-")
    if normalized in {"traditional", "lloyd-kmeans", "k-means"}:
        normalized = KMEANS_ALGORITHM_LLOYD
    elif normalized in {"balanced-kmeans", "cuvs-balanced"}:
        normalized = KMEANS_ALGORITHM_BALANCED
    if normalized not in {KMEANS_ALGORITHM_LLOYD, KMEANS_ALGORITHM_BALANCED}:
        raise ValueError(
            "Unsupported KMeans algorithm: "
            f"{algorithm!r}; expected 'lloyd' or 'balanced'"
        )
    return normalized


def fit_best_kmeans(
    data: torch.Tensor,
    *,
    d: int,
    k: int,
    niter: int = FAISS_DEFAULT_NITER,
    use_triton: bool = True,
    device: torch.device | str = "cpu",
    seed: int = FAISS_DEFAULT_SEED,
    n_init: int = FAISS_DEFAULT_NREDO,
    stage: int = 0,
    max_points_per_centroid: int | None = FAISS_DEFAULT_MAX_POINTS_PER_CENTROID,
    min_points_per_centroid: int = FAISS_DEFAULT_MIN_POINTS_PER_CENTROID,
    trainset_fraction: float | None = None,
    init_method: str = KMEANS_INIT_KMEANS_PLUS_PLUS,
    training_data: torch.Tensor | None = None,
    training_indices: torch.Tensor | None = None,
    algorithm: str = KMEANS_ALGORITHM_LLOYD,
    balanced_hierarchical: bool = False,
) -> KMeansTrainingResult:
    """Fit traditional or cuVS-style Balanced K-Means.

    The selected training subset is used for fitting and restart selection;
    final labels are always assigned for every row in ``data``.  When
    ``balanced_hierarchical`` is true, the balanced algorithm uses cuVS's
    hierarchical builder.  This is intended for the coarse IVF quantizer;
    PQ codebooks use the flat builder by default.
    """
    if data.ndim != 2:
        raise ValueError(f"KMeans data must be 2D, got shape {tuple(data.shape)}")
    if data.shape[1] != d:
        raise ValueError(f"KMeans data dimension {data.shape[1]} does not match d={d}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if data.shape[0] < k:
        raise ValueError(f"KMeans requires at least k={k} rows, got {data.shape[0]}")
    if niter <= 0:
        raise ValueError(f"niter must be positive, got {niter}")
    if n_init <= 0:
        raise ValueError(f"n_init must be positive, got {n_init}")
    if min_points_per_centroid < 0:
        raise ValueError(
            "min_points_per_centroid must be non-negative, "
            f"got {min_points_per_centroid}"
        )
    normalized_algorithm = _normalize_algorithm(algorithm)

    target_device = torch.device(device)
    selected_training_indices: torch.Tensor | None = None
    if training_data is None:
        selected_training_indices = select_training_indices(
            int(data.shape[0]),
            k=k,
            max_points_per_centroid=max_points_per_centroid,
            trainset_fraction=trainset_fraction,
            seed=seed,
            stage=stage,
            device=data.device,
        )
        training_data = data.index_select(0, selected_training_indices)
    else:
        if training_data.ndim != 2 or training_data.shape[1] != d:
            raise ValueError(
                "training_data must have shape [n, d] with the same d as data, "
                f"got {tuple(training_data.shape)} and d={d}"
            )
        if training_indices is not None:
            training_indices = training_indices.to(dtype=torch.int64, device=data.device)
            if training_indices.ndim != 1 or training_indices.numel() != training_data.shape[0]:
                raise ValueError(
                    "training_indices must be a 1D index for every training_data row, "
                    f"got {tuple(training_indices.shape)} for {training_data.shape[0]} rows"
                )
            selected_training_indices = training_indices
        if training_data.shape[0] < k:
            raise ValueError(
                f"KMeans training_data requires at least k={k} rows, "
                f"got {training_data.shape[0]}"
            )
    if (
        min_points_per_centroid > 0
        and training_data.shape[0] < k * min_points_per_centroid
    ):
        warnings.warn(
            f"KMeans training has {training_data.shape[0]} points for {k} "
            f"centroids; Faiss recommends at least "
            f"{k * min_points_per_centroid} points.",
            UserWarning,
            stacklevel=2,
        )

    best: KMeansTrainingResult | None = None
    for restart in range(int(n_init)):
        run_seed = derive_kmeans_seed(seed, stage, restart)
        if normalized_algorithm == KMEANS_ALGORITHM_BALANCED:
            if balanced_hierarchical:
                labels, centroids = _run_balanced_hierarchical_kmeans_once(
                    training_data,
                    k=k,
                    niter=niter,
                    use_triton=use_triton,
                    device=target_device,
                    seed=run_seed,
                )
            else:
                labels, centroids = _run_balanced_kmeans_once(
                    training_data,
                    k=k,
                    niter=niter,
                    use_triton=use_triton,
                    device=target_device,
                    seed=run_seed,
                )
        else:
            centroids = _run_kmeans_once(
                training_data,
                d=d,
                k=k,
                niter=niter,
                use_triton=use_triton,
                device=target_device,
                seed=run_seed,
                stage=stage,
                init_method=init_method,
            )
            labels = _assign_kmeans_labels(
                training_data,
                centroids,
                d=d,
                use_triton=use_triton,
                device=target_device,
            )

        inertia = _assigned_inertia(
            training_data,
            labels,
            centroids,
            device=target_device,
        )
        if best is None or inertia < best.inertia:
            best = KMeansTrainingResult(
                labels=labels.clone(),
                centroids=centroids.clone(),
                inertia=inertia,
                seed=run_seed,
            )

    if best is None:  # pragma: no cover - guarded by n_init validation
        raise RuntimeError("KMeans training produced no result")
    # cuVS predicts labels from the final centers after ``fit``.  In
    # particular, it does not reuse the labels produced by an intermediate
    # balancing pass.  Always doing the final nearest-center assignment here
    # is important when the fit used a sampled subset: preserving those stale
    # subset labels can create severely unbalanced IVF lists.
    final_labels = _assign_kmeans_labels(
        data,
        best.centroids,
        d=d,
        use_triton=use_triton,
        device=target_device,
    ).to(dtype=torch.int64)

    return KMeansTrainingResult(
        labels=final_labels,
        centroids=best.centroids,
        inertia=best.inertia,
        seed=best.seed,
    )


__all__ = [
    "FAISS_DEFAULT_NITER",
    "FAISS_DEFAULT_NREDO",
    "FAISS_DEFAULT_MIN_POINTS_PER_CENTROID",
    "FAISS_DEFAULT_MAX_POINTS_PER_CENTROID",
    "FAISS_DEFAULT_SEED",
    "KMEANS_INIT_RANDOM",
    "KMEANS_INIT_KMEANS_PLUS_PLUS",
    "KMEANS_ALGORITHM_LLOYD",
    "KMEANS_ALGORITHM_BALANCED",
    "KMeansTrainingResult",
    "derive_kmeans_seed",
    "select_training_indices",
    "fit_best_kmeans",
]
