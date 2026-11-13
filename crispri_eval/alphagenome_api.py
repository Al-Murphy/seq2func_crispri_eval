"""
crispri_eval.alphagenome_api
============================

AlphaGenome 2 through DeepMind's hosted AlphaGenome API, behind the same call the
benchmark scripts make on the AlphaGenome PyTorch port.

The scripts call ``outputs = model(x, organism_index)`` and read
``outputs[head][1]`` as a ``(B, L, n_tracks)`` tensor, then select tracks and take
the exon readout. :class:`AlphaGenomeAPIModel` answers that call by sending each
sequence in the batch to the API (in parallel threads), so the windows, shuffles,
strand-matched track selection and exon readout are shared code with the PyTorch
path and only the forward pass differs.

AlphaGenome 2 is conditioned on a cell type: each request names one condition
(an ontology CURIE; K562 is ``EFO:0002067``), which selects that condition's
learned embedding inside the model, and the response holds only that condition's
tracks. The track list for a condition is read once, from a 16 kb probe
prediction, and every later response is checked against it.

The response includes tracks the model imputes for assays never measured in that
condition (``imputation_mode != "observed"``; K562 RNA-seq has 9 measured and 3
imputed). By default only measured tracks are read (``observed_only=True``).

Requires the ``alphagenome`` client (>= 2.0.0, ``alphagenome.models.v2``) and an
API key in a file (default ``~/.config/alphagenome/api_key``). The key is read
from that file only; it is never printed or written to outputs.
"""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

#: Benchmark modality -> attribute of the API ``Output`` and ``OutputType`` name.
HEADS = {"rna_seq": "RNA_SEQ", "cage": "CAGE"}
#: gRPC status codes worth retrying (rate limits, transient server faults).
RETRY_CODES = {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "INTERNAL", "ABORTED", "UNKNOWN"}
DEFAULT_KEY_FILE = "~/.config/alphagenome/api_key"
K562_CURIE = "EFO:0002067"

_BASES = np.frombuffer(b"ACGTN", dtype="S1")


def onehot_to_str(x):
    """(L, 4) one-hot in ACGT order -> DNA string; all-zero columns (padding / N) -> 'N'."""
    a = np.asarray(x, dtype=np.float32)
    idx = a.argmax(axis=-1)
    idx[a.sum(axis=-1) == 0] = 4
    return _BASES[idx].tobytes().decode("ascii")


def strand_bundle(track_strands, keep=None):
    """Strand-matched track indices, the rule the PyTorch scripts apply to their metadata.

    The benchmark reverse-complements minus-strand genes, so the gene is always on '+'
    in the input: read '+' and unstranded ('.') tracks forward, '-' and '.' on a
    reverse-complement pass. ``keep`` (one bool per track) restricts the choice to
    those tracks, e.g. to measured ones; indices still refer to the full track list.
    """
    strands = [str(s) for s in track_strands]
    keep = [True] * len(strands) if keep is None else [bool(k) for k in keep]
    all_inds = [i for i, k in enumerate(keep) if k]
    plus = [i for i in all_inds if strands[i] in ("+", ".")]
    minus = [i for i in all_inds if strands[i] in ("-", ".")]
    return {"all": all_inds, "plus": plus or all_inds, "minus": minus or all_inds}


def observed_mask(track_metadata):
    """True for measured tracks (``imputation_mode == "observed"``); all True if the column is absent."""
    if "imputation_mode" not in track_metadata.columns:
        return [True] * len(track_metadata)
    return list(track_metadata["imputation_mode"].astype(str) == "observed")


class AlphaGenomeAPIModel:
    """Callable stand-in for the AlphaGenome PyTorch model, backed by the AlphaGenome 2 API."""

    def __init__(self, modality="rna_seq", condition=K562_CURIE, model_version="ALL_FOLDS",
                 api_key_file=DEFAULT_KEY_FILE, concurrency=4, max_retries=8, observed_only=True):
        from alphagenome.models.v2 import dna_client  # imported here: optional dependency

        if modality not in HEADS:
            raise ValueError("modality must be one of {}".format(sorted(HEADS)))
        with open(os.path.expanduser(api_key_file)) as fh:
            key = fh.read().strip()
        self._dc = dna_client
        self.observed_only = observed_only
        self.modality = modality
        self.condition = condition
        self.model_version = model_version
        self.output_type = getattr(dna_client.OutputType, HEADS[modality])
        self.client = dna_client.create(key, model_version=dna_client.ModelVersion[model_version])
        self.max_retries = max_retries
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(concurrency)))
        self.n_calls = 0
        self.n_retries = 0
        # Read the condition's track list once from a small probe prediction.
        probe = "ACGT" * (dna_client.SEQUENCE_LENGTH_16KB // 4)
        _, md = self._predict(probe)
        self.track_metadata = md.reset_index(drop=True)
        self._track_key = self._key(self.track_metadata)
        if len(self.track_metadata) == 0:
            raise RuntimeError("AlphaGenome API returned no {} tracks for condition {}".format(
                modality, condition))

    @staticmethod
    def _key(md):
        return tuple(zip(md["name"].astype(str), md["strand"].astype(str)))

    def eval(self):
        return self

    def track_bundle(self):
        """Strand-matched indices into this condition's tracks (measured ones only, unless
        ``observed_only=False``), plus a description."""
        obs = observed_mask(self.track_metadata)
        keep = obs if self.observed_only else None
        bundle = strand_bundle(self.track_metadata["strand"], keep=keep)
        desc = ("AlphaGenome 2 API | condition {} | {} tracks ({} measured, {} imputed) | {} | "
                "strand-aware (+/.: {}, -/.: {})").format(
            self.condition, len(self.track_metadata), sum(obs), len(obs) - sum(obs),
            "measured only" if self.observed_only else "measured + imputed",
            len(bundle["plus"]), len(bundle["minus"]))
        return bundle, desc

    def _predict(self, seq):
        """One API prediction with retries; returns (values (L, n_tracks) float32, metadata)."""
        import grpc

        for attempt in range(self.max_retries + 1):
            try:
                out = self.client.predict_sequence(
                    seq, requested_outputs=[self.output_type], condition=self.condition)
                td = getattr(out, self.modality)
                self.n_calls += 1
                return np.asarray(td.values, dtype=np.float32), td.metadata
            except grpc.RpcError as err:
                code = err.code().name if hasattr(err, "code") and err.code() else "UNKNOWN"
                if code not in RETRY_CODES or attempt == self.max_retries:
                    raise
                self.n_retries += 1
                wait = min(600.0, 5.0 * 2 ** attempt) * (0.5 + random.random())
                print("AlphaGenome API {} (attempt {}/{}); retrying in {:.0f}s".format(
                    code, attempt + 1, self.max_retries, wait), flush=True)
                time.sleep(wait)

    def __call__(self, x, organism_index=None):
        """(B, L, 4) one-hot -> {modality: (None, (B, L, n_tracks) float32 tensor)}."""
        seqs = [onehot_to_str(xi) for xi in x.detach().cpu().numpy()]
        results = list(self.pool.map(self._predict, seqs))
        for _, md in results:
            if self._key(md.reset_index(drop=True)) != self._track_key:
                raise RuntimeError("AlphaGenome API track list changed between calls")
        vals = np.stack([v for v, _ in results])
        return {self.modality: (None, torch.from_numpy(vals))}
