"""
crispri_eval.datasets
=====================

PyTorch ``Dataset`` classes for two CRISPRi enhancer-knockdown screens:

* :class:`GasperiniDataset` — Gasperini et al. (2019), pooled CRISPRi at
  scale; high-confidence enhancer-gene pairs and significant pairs only.
* :class:`FulcoDataset`     — Fulco et al. (2019), CRISPRi-FlowFISH,
  aligned to the Avsec et al. / Karollus et al. SequenceModelBenchmark
  Zenodo tables.

Both datasets centre a model's receptive field on each gene's TSS, encode
the wild-type sequence, then either dinucleotide-shuffle or random-permute
the enhancer slice ``N`` times to generate matched CRISPRi-perturbed
sequences. Per-item outputs:

    x_WT        (4, L) one-hot wild-type sequence
    x_crispri   (N, 4, L) one-hot perturbed sequences
    y_delta     measured RNA-decrease effect (scalar)
    ... + locus metadata (tss, enh_loc, strand, enh_dist)

Reference column conventions match the published SequenceModelBenchmark
(``ziga_additional_columns.tsv``) so results are directly comparable.
"""

import glob
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pysam
import torch

from .dataset_utils import (
    dinucleotide_shuffle,
    get_genome,
    one_hot_encode_dna,
    seq_loader,
)


class GasperiniDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str = "./metadata/",
        sequence_length: int  = 196_608,
        seed: int = 101,
        N: int = 25,
        enhancer_bp: int = 2000,
        genome_build: str = "hg38",
        high_confidence_subset: bool = True,
        crispri_perturb_mode: str = "dinucleotide",
        min_enh_dist: int = 990,
        tta_shifts=(0,),
    ):
        """
        Data loader for Gasperini et al. CRISPRi data.
        Parameters N and enhancer_bp match the analysys of Karrolus et al., 2023.
        
        Parameters:
            - data_path: Path to the data directory with the precomputed Gasperini gene-enhancer pairs, enahncer liftover locations (if necessary) and the gene locations.
            - sequence_length: Length of the sequence to be loaded. The window is centred on the
              gene TSS (matching the SequenceModelBenchmark / Avsec et al. approach); pairs are kept
              if the enhancer midpoint lies within half this length of the TSS.
            - seed: Seed for the dinucleotide shuffling. Default 101.
            - N: Number of times to dinucleotide shuffle the sequence. Default 25. Note this will increase the number of samples returnedby N.
            - enhancer_bp: Number of base pairs to consider for the enhancer. Default 2000.
            - genome_build: Genome build to use. Default hg38.
            - high_confidence_subset: Whether to use the high confidence subset of the dataset. Default True.
            - crispri_perturb_mode: How to scramble the enhancer window for CRISPRi negatives.
              ``"dinucleotide"`` (default): preserve dinucleotide frequencies (Altschul–Erickson).
              ``"shuffle_bases"``: keep the **same** multiset of bases in the perturbed window
              (same counts of A/C/G/T/N as the WT enhancer slice) but draw a **uniform random
              permutation** of their order along the sequence (destroys motifs and dinucleotide
              structure). Alias: ``"random_permute"`` (same behaviour).
        
        Returns:
            - x_WT: pytorch The WT sequence of the gene.
            - x_crispri: pytorch The CRISPRi sequence of the gene.
            - y_delta: The expression level change of the gene.    
            - gene_id: The ID of the gene.
            - gene_name: The name of the gene.
            - enh_loc: The location of the enhancer.
            - tss: Genomic position of the TSS (Gene start for + strand, Gene end for − strand).
            - tss_seq_index: Index along the sequence axis (0..sequence_length-1) where the TSS falls
              in the **model input** tensor (after reverse-complement for − strand). With TSS-centred
              windowing this will always be at or very near L//2 (sequence centre).
            - strand: The strand of the gene.
            - min_enh_dist: Minimum enhancer–TSS distance (bp, exclusive) to keep.
              Pairs with ``enh_dist <= min_enh_dist`` are dropped before inference.
              SequenceModelBenchmark uses 990 bp to remove near-promoter elements.
              Default 0 (no lower filter).
            - enh_dist: The distance of the enhancer midpoint to the TSS.
            - enh_middle: Genomic midpoint of the perturbed enhancer.
        """
        #child class has a object of parent's class and often has more attribute that parent's class doesn't have
        super().__init__()
        
        
        self.data_path = data_path
        self.seed = seed
        #enf resolution - must be 128bp for enformer
        assert sequence_length % 128 == 0
        
        self.sequence_length = sequence_length
        self.N = N
        self.enhancer_bp = enhancer_bp
        
        # Store genome build info instead of initializing file handles here
        assert genome_build in ["hg38", "hg19"], "Genome build must be one of hg38, hg19"
        self.genome_build = genome_build
        self.genome = None
        self.genome_ref = None
        self.chr_sizes = None
        self.high_confidence_subset = high_confidence_subset
        self.min_enh_dist = int(min_enh_dist)
        # Test-time augmentation shifts (bp); each re-fetches a real-flank window
        # at seq_start + shift. (0,) = no shift augmentation (default).
        self.tta_shifts = tuple(int(s) for s in tta_shifts) or (0,)
        _valid = ("dinucleotide", "shuffle_bases", "random_permute")
        if crispri_perturb_mode not in _valid:
            raise ValueError(
                "crispri_perturb_mode must be one of {}, got {!r}".format(_valid, crispri_perturb_mode)
            )
        # Canonical internal name (random_permute is a backward-compatible alias)
        self.crispri_perturb_mode = (
            "shuffle_bases" if crispri_perturb_mode == "random_permute" else crispri_perturb_mode
        )
        
        #get Gasperini data
        self._prepare_gasperini_data()
        self._filt_enh_receptive_field()
    
    def _init_genome_loaders(self):
        """Initialize genome loaders per worker process to avoid file handle sharing issues."""
        if self.genome is None:
            self.genome = seq_loader(build=self.genome_build, model_receptive_field=self.sequence_length)
        
        if self.genome_ref is None:
            self.genome_ref = pysam.Fastafile(get_genome(build=self.genome_build))
            self.chr_sizes = dict(zip(self.genome_ref.references, self.genome_ref.lengths))
    
    def _prepare_gasperini_data(self):
        assert os.path.exists(self.data_path), "Data path does not exist"
        all_files = glob.glob(self.data_path+f"/*")
        assert self.data_path+"Gasperini_sign_enhancer_gene_pairs.csv" in all_files, "Gasperini gene-enhancer pairs file not found"
        self.crispri_data = pd.read_csv(self.data_path+"Gasperini_sign_enhancer_gene_pairs.csv")
        if self.high_confidence_subset:
            self.crispri_data = self.crispri_data[self.crispri_data['high_confidence_subset'] == True]
        
        if self.genome_build=="hg38":        
            assert self.data_path+"Gasperini_gene_tss_hg38_biomart_GRCh38_p14.csv" in all_files, "Gasperini gene tss file not found"
            assert self.data_path+"Gasperini_enh_hg38_ucscLiftOver.bed" in all_files, "Gasperini enhancer locations hg38 file not found"
            #get lifted over enhancers
            lifted_over_enhancers = pd.read_csv(self.data_path+"Gasperini_enh_hg38_ucscLiftOver.bed", sep='\t',header=None)
            #set name
            lifted_over_enhancers.columns = ['enh_loc']
            self.crispri_data = pd.concat([self.crispri_data, lifted_over_enhancers], axis=1)
            #get gene tss
            gasp_gene_tss = pd.read_csv(self.data_path+"Gasperini_gene_tss_hg38_biomart_GRCh38_p14.csv", sep=',')
            #join
            self.crispri_data = pd.merge(self.crispri_data, gasp_gene_tss, left_on='ENSG', right_on='Gene stable ID', how='inner')
            
        elif self.genome_build=="hg19":
            assert self.data_path+"Gasperini_gene_tss_hg19_biomart_GRCh37_p13.csv" in all_files, "Gasperini gene tss file not found"
            #format enhancer locations
            self.crispri_data['enh_loc'] = self.crispri_data['chr.candidate_enhancer']+":"+self.crispri_data['start.candidate_enhancer'].astype(str)+"-"+self.crispri_data['stop.candidate_enhancer'].astype(str)
            #get gene tss
            gasp_gene_tss = pd.read_csv(self.data_path+"Gasperini_gene_tss_hg19_biomart_GRCh37_p13.csv", sep=',')
            #join
            self.crispri_data = pd.merge(self.crispri_data, gasp_gene_tss, left_on='ENSG', right_on='Gene stable ID', how='inner')            
        else:
            raise ValueError(f"Genome build must be one of hg38, hg19, got {self.genome_build}")
        
        self.crispri_data.reset_index(drop=True, inplace=True)

        # TSS (5' transcription start): Ensembl "Gene start" is the minimum genomic coordinate and
        # equals the TSS for + strand; for − strand the TSS is at "Gene end" (see DecimaGeneExpressionDataset).
        gs = self.crispri_data["Gene start (bp)"].astype(int)
        ge = self.crispri_data["Gene end (bp)"].astype(int)
        st = self.crispri_data["Strand"].astype(int)
        self.crispri_data["tss_pos"] = np.where(st >= 1, gs, ge)

        print("Number of validated Gasperini Enhancer-Gene pairs:", self.crispri_data.shape[0])
        
    def __len__(self):
        # Return total number of samples
        return self.crispri_data.shape[0]
    
    def _filt_enh_receptive_field(self):
        # Filter for enhancers within the receptive field of the model.
        # The sequence window is centred on the TSS; we require the enhancer midpoint
        # to lie within [tss - L/2, tss + L/2) (matching SequenceModelBenchmark).
        enh_parts = self.crispri_data["enh_loc"].str.split(":", expand=True)
        enh_coords = enh_parts[1].str.split("-", expand=True)
        enh_start = enh_coords[0].astype(int)
        enh_end = enh_coords[1].astype(int)

        self.crispri_data["enh_len"] = (enh_end - enh_start).abs()
        # Not always in order (enh_end can precede enh_start in some rows)
        self.crispri_data["enh_middle"] = (enh_start + enh_end) // 2
        self.crispri_data["enh_dist"] = abs(
            self.crispri_data["tss_pos"] - self.crispri_data["enh_middle"]
        )

        # TSS-centred window: enhancer midpoint must lie within half the window.
        self.crispri_data = self.crispri_data[
            self.crispri_data["enh_dist"] <= self.sequence_length // 2
        ]

        # Exclude near-promoter elements (SequenceModelBenchmark uses > 990 bp).
        if self.min_enh_dist > 0:
            n_before = self.crispri_data.shape[0]
            self.crispri_data = self.crispri_data[
                self.crispri_data["enh_dist"] > self.min_enh_dist
            ]
            print(
                "min_enh_dist={} bp: removed {} near-promoter pairs, {} remaining.".format(
                    self.min_enh_dist, n_before - self.crispri_data.shape[0], self.crispri_data.shape[0]
                )
            )

        assert self.crispri_data.shape[0] > 0, "No enhancers found within the receptive field of the model"
        print("Number of validated Gasperini Enhancer-Gene pairs within the receptive field of the model:", self.crispri_data.shape[0])
    
    def _reverse_complement_dna(self, seq):
        # seq shape: (N, seq_len, 4)  dim1=positions, dim2=channels [ACGT]
        # Reverse complement = flip positions (dim 1) AND complement bases (dim 2, ACGT→TGCA)
        return torch.flip(seq, dims=[1, 2])

    def _crispri(self, seq, cCRE_i, seq_window_start_genomic, crispri_effect_size_i, pair_idx=0):
        seq = seq.swapaxes(1, 2)
        start_pos = cCRE_i - crispri_effect_size_i // 2
        rel_pos = max(0, start_pos - seq_window_start_genomic)
        # Silence the region: dinucleotide shuffle (default) or uniform shuffle of bases (ablation).
        silenced_seq = seq[:, :, rel_pos : rel_pos + crispri_effect_size_i]
        L = silenced_seq.shape[-1]
        all_dinucl_shuffled_seqs = []

        if self.crispri_perturb_mode == "dinucleotide":
            silenced_seq = dinucleotide_shuffle(
                silenced_seq, allow_N=True, n=self.N
            )  # (1, N, 4, L)
            for n_i in range(self.N):
                silenced_seq_i = silenced_seq[:, n_i, :, :]
                seq_crispri_i = torch.concat(
                    [seq[:, :, 0:rel_pos], silenced_seq_i, seq[:, :, rel_pos + crispri_effect_size_i :]],
                    axis=2,
                )
                seq_crispri_i = seq_crispri_i.swapaxes(1, 2)
                all_dinucl_shuffled_seqs.append(seq_crispri_i)
        else:
            # shuffle_bases: same nucleotides as WT enhancer slice, random order (uniform permutation
            # of the L positions' one-hot columns). Multiset of bases unchanged; N/padding columns stay
            # all-zero when moved. Implemented as silenced_seq[:, :, perm] with perm ~ randperm(L).
            dev = silenced_seq.device
            for n_i in range(self.N):
                g = torch.Generator()
                g.manual_seed(int(self.seed) + int(pair_idx) * 1_000_003 + n_i)
                perm = torch.randperm(L, generator=g).to(device=dev, dtype=torch.long)
                silenced_seq_i = silenced_seq[:, :, perm]
                seq_crispri_i = torch.concat(
                    [seq[:, :, 0:rel_pos], silenced_seq_i, seq[:, :, rel_pos + crispri_effect_size_i :]],
                    axis=2,
                )
                seq_crispri_i = seq_crispri_i.swapaxes(1, 2)
                all_dinucl_shuffled_seqs.append(seq_crispri_i)

        seq_crispri = torch.cat(all_dinucl_shuffled_seqs, axis=0)
        return seq_crispri

    def _tss_seq_index(self, tss_genomic, seq_start, strand_sign, chrom_scaffold_name):
        """
        Position of the TSS along the sequence axis (0..L-1) after get_seq_start + swapaxes,
        then strand orientation: for − strand genes we report the index in the **final** tensor
        (after reverse-complement), i.e. the index Enformer/AlphaGenome should use for readout.

        Mirrors ``seq_loader.get_seq_start`` padding, then maps i_pre -> L-1-i_pre on − strand.
        """
        L = self.sequence_length
        pad_N_strt = max(seq_start * -1, 0)
        start = max(0, seq_start)
        chrom = "chr" + str(chrom_scaffold_name)
        chrom_len = self.genome.genome_dat.get_reference_length(chrom)
        end = min(seq_start + self.sequence_length + self.genome.mod, chrom_len)

        if tss_genomic < start:
            i_pre = 0
        elif tss_genomic >= end:
            i_pre = L - 1
        else:
            i_pre = pad_N_strt + (tss_genomic - start)
        i_pre = int(max(0, min(L - 1, i_pre)))
        if strand_sign < 1:
            return L - 1 - i_pre
        return i_pre

    def _build_window(self, data, idx, shift):
        """Build one (WT, CRISPRi) window shifted ``shift`` bp from the TSS-centred start.

        Re-fetched from the genome at ``seq_start + shift`` so the flanks are real
        sequence (not rolled/wrapped). Returns
        ``(x_WT (1, L, 4), x_crispri (N, L, 4), tss_seq_index, seq_start)``.
        """
        tss = int(data["tss_pos"])
        enh_middle = int(data["enh_middle"])
        strand_i = int(data["Strand"])
        seq_start = tss - (self.sequence_length // 2) + int(shift)

        seqs = self.genome.get_seq_start(
            chrom="chr" + str(data["Chromosome/scaffold name"]),
            seq_start=seq_start,
            strand="+",
            ohe=True,
            rev_comp=False,
            pad_seq=True,
        ).swapaxes(1, 2).to(torch.float32)
        seqs_wt = seqs.clone()
        seqs = self._crispri(seqs, enh_middle, seq_start, self.enhancer_bp, pair_idx=idx)

        if strand_i < 1:
            seqs = self._reverse_complement_dna(seqs)
            seqs_wt = self._reverse_complement_dna(seqs_wt)

        tss_seq_index = self._tss_seq_index(
            tss, seq_start, strand_i, data["Chromosome/scaffold name"]
        )
        return seqs_wt, seqs, tss_seq_index, seq_start

    def __getitem__(self, idx):
        # Initialize genome loaders if needed (per worker process)
        self._init_genome_loaders()

        data = self.crispri_data.iloc[idx]
        tss = int(data["tss_pos"])
        enh_middle = int(data["enh_middle"])
        strand_i = int(data["Strand"])

        meta = {
            "y_delta": torch.tensor(data["Diff_expression_test_fold_change"], dtype=torch.float32),
            "gene_id": data["ENSG"],
            "gene_name": data["target_gene_short"],
            "enh_loc": data["enh_loc"],
            "tss": tss,
            "enh_middle": enh_middle,  # genomic midpoint of enhancer (offset from window centre)
            "strand": strand_i,
            "enh_dist": data["enh_dist"].item(),
        }

        if self.tta_shifts == (0,):
            seqs_wt, seqs, tss_seq_index, _ = self._build_window(data, idx, 0)
            meta["x_WT"] = seqs_wt
            meta["x_crispri"] = seqs
            meta["tss_seq_index"] = tss_seq_index
            return meta

        # TTA: one real-flank window per shift, stacked along a leading shift axis.
        wts, crs, tss_idxs, seq_starts = [], [], [], []
        for s in self.tta_shifts:
            wt, cr, tss_idx, ss = self._build_window(data, idx, s)
            wts.append(wt)
            crs.append(cr)
            tss_idxs.append(int(tss_idx))
            seq_starts.append(int(ss))
        meta["x_WT"] = torch.cat(wts, dim=0)             # (S, L, 4)
        meta["x_crispri"] = torch.stack(crs, dim=0)      # (S, N, L, 4)
        meta["tss_seq_index"] = torch.tensor(tss_idxs, dtype=torch.long)
        meta["seq_start"] = torch.tensor(seq_starts, dtype=torch.long)
        meta["tta_shifts"] = torch.tensor(self.tta_shifts, dtype=torch.long)
        return meta


def _fulco_ensembl_lookup_symbols(symbols, genome_build="hg38"):
    """
    Map HGNC symbols to Ensembl gene IDs and strand via Ensembl REST (used when
    ``ziga_additional_columns.tsv`` has ``Gene`` but no ``gene_id``, as in Karollus Zenodo).
    """
    # Ensembl /lookup/symbol rejects some legacy table symbols (HTTP 400); try HGNC-preferred alias.
    symbol_aliases = {
        "H1FX": "H1-10",
    }
    server = (
        "https://rest.ensembl.org"
        if genome_build == "hg38"
        else "https://grch37.rest.ensembl.org"
    )
    rows = []
    for sym in sorted(set(str(s).strip() for s in symbols if str(s).strip())):
        lookup_sym = symbol_aliases.get(sym, sym)
        path = "/lookup/symbol/homo_sapiens/{}".format(urllib.parse.quote(lookup_sym, safe=""))
        url = server + path + "?content-type=application/json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (FulcoDataset)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            gid = str(data.get("id", "")).split(".")[0]
            st = data.get("strand", None)
            if gid and st is not None:
                rows.append(
                    {
                        "resolved_symbol": sym,
                        "Gene stable ID": gid,
                        "Strand_ensembl": int(st),
                    }
                )
            else:
                warnings.warn("Ensembl lookup for {!r} returned no id/strand".format(sym))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                warnings.warn("Ensembl: no symbol match for {!r}".format(sym))
            else:
                warnings.warn("Ensembl lookup failed for {!r}: {}".format(sym, exc))
        except Exception as exc:
            warnings.warn("Ensembl lookup failed for {!r}: {}".format(sym, exc))
        time.sleep(0.07)
    if not rows:
        raise RuntimeError(
            "Could not resolve any gene symbols via Ensembl REST. "
            "Add a ``gene_id`` / ``Gene stable ID`` column to ziga, or check network / symbols."
        )
    return pd.DataFrame(rows)


class FulcoDataset(torch.utils.data.Dataset):
    """
    Fulco et al. (2019) CRISPRi enhancer knockdown screen, aligned to Avsec et al. / Karollus et al.
    sequence windows (``ziga_additional_columns.tsv`` from SequenceModelBenchmark Zenodo).

    Experimental fold-change remaining (Gasperini-comparable): ``expr_ratio = 1 + Fraction_change``,
    where *Fraction change in gene expr* is from ``enhancer_knockdown_effects.tsv``. Benchmark
    scripts use ``y_delta = 1 - expr_ratio`` (positive when RNA decreased). Pass
    ``observed_subset='knockdown_only'`` (via ``test_fulco_* --fulco_corr_observed_subset``) to keep
    only rows with Fulco fraction change ``< 0`` (measured RNA decrease); inference and the result
    CSV are restricted to that subset. ``all`` evaluates every merged pair (same as
    ``plot_fulco_results.py --observed_mode signed``).

    Strand: optional Biomart CSV (``--gene_tss_csv``) merged on ``ENSG``; if a row is missing there,
    strand from Ensembl symbol lookup is used when the Zenodo ziga table has symbols only.
    """

    def __init__(
        self,
        data_path: str = "./metadata/",
        sequence_length: int = 196_608,
        seed: int = 101,
        N: int = 25,
        enhancer_bp: int = 2000,
        genome_build: str = "hg38",
        validated_only: bool = True,
        crispri_perturb_mode: str = "dinucleotide",
        min_enh_dist: int = 990,
        gene_tss_csv: Optional[str] = None,
        fraction_change_col: str = "Fraction change in gene expr",
        observed_subset: str = "all",
        tta_shifts=(0,),
    ):
        super().__init__()
        self.data_path = data_path.rstrip("/") + "/"
        assert sequence_length % 128 == 0
        self.sequence_length = sequence_length
        self.seed = seed
        # Test-time augmentation shifts (bp). Each shift re-fetches a window at
        # ``seq_start + shift`` from the genome, so the flanks are real sequence
        # (not wrapped/rolled). (0,) = no shift augmentation (default).
        self.tta_shifts = tuple(int(s) for s in tta_shifts) or (0,)
        self.N = N
        self.enhancer_bp = enhancer_bp
        assert genome_build in ("hg38", "hg19"), "Genome build must be hg38 or hg19"
        self.genome_build = genome_build
        self.genome = None
        self.genome_ref = None
        self.chr_sizes = None
        self.validated_only = validated_only
        self.min_enh_dist = int(min_enh_dist)
        self.gene_tss_csv = gene_tss_csv
        self.fraction_change_col = fraction_change_col
        _osub = ("all", "knockdown_only")
        if observed_subset not in _osub:
            raise ValueError("observed_subset must be one of {}, got {!r}".format(_osub, observed_subset))
        self.observed_subset = observed_subset

        _valid = ("dinucleotide", "shuffle_bases", "random_permute")
        if crispri_perturb_mode not in _valid:
            raise ValueError(
                "crispri_perturb_mode must be one of {}, got {!r}".format(_valid, crispri_perturb_mode)
            )
        self.crispri_perturb_mode = (
            "shuffle_bases" if crispri_perturb_mode == "random_permute" else crispri_perturb_mode
        )

        self._prepare_fulco_data()
        self._filt_enh_receptive_field()

    @staticmethod
    def _normalize_chrom(c):
        s = str(c).strip()
        if s.lower().startswith("chr"):
            return "chr" + s[3:]
        return "chr" + s

    def _init_genome_loaders(self):
        if self.genome is None:
            self.genome = seq_loader(build=self.genome_build, model_receptive_field=self.sequence_length)
        if self.genome_ref is None:
            self.genome_ref = pysam.Fastafile(get_genome(build=self.genome_build))
            self.chr_sizes = dict(zip(self.genome_ref.references, self.genome_ref.lengths))

    def _prepare_fulco_data(self):
        ziga_path = os.path.join(self.data_path, "ziga_additional_columns.tsv")
        fulco_fx_path = os.path.join(self.data_path, "enhancer_knockdown_effects.tsv")
        if not os.path.exists(ziga_path):
            raise FileNotFoundError(
                "Missing {} (from Zenodo SequenceBenchmark.zip: Data/Fulco_CRISPRi/).".format(ziga_path)
            )
        if not os.path.exists(fulco_fx_path):
            raise FileNotFoundError(
                "Missing {} (from Zenodo SequenceBenchmark.zip: Data/Fulco_CRISPRi/).".format(fulco_fx_path)
            )

        ziga = pd.read_csv(ziga_path, sep="\t")
        if "dataset_name" in ziga.columns:
            ziga = ziga[ziga["dataset_name"].astype(str) == "fulco2019"].copy()
        if self.validated_only and "validated" in ziga.columns:
            vc = ziga["validated"]
            if vc.dtype == object:
                ziga = ziga[vc.astype(str).str.lower().isin(("true", "1"))].copy()
            else:
                ziga = ziga[vc.astype(bool)].copy()

        fx = pd.read_csv(fulco_fx_path, sep="\t")
        if "fulco_key" not in fx.columns and "Gene" in fx.columns and "Element name" in fx.columns:
            fx = fx.copy()
            fx["fulco_key"] = fx["Gene"].astype(str) + "_" + fx["Element name"].astype(str)

        if self.fraction_change_col not in fx.columns:
            raise KeyError(
                "Column {!r} not found in {}; columns include: {}".format(
                    self.fraction_change_col, fulco_fx_path, list(fx.columns)[:30])
            )

        fx_sub = fx[["fulco_key", self.fraction_change_col]].drop_duplicates(subset=["fulco_key"])
        self.crispri_data = ziga.merge(fx_sub, on="fulco_key", how="inner")
        self.crispri_data = self.crispri_data.dropna(subset=[self.fraction_change_col])

        # Ensembl gene id: Zenodo ziga often has ``Gene`` / ``gene`` only (no gene_id).
        if "gene_id" in self.crispri_data.columns:
            self.crispri_data["ENSG"] = (
                self.crispri_data["gene_id"].astype(str).str.split(".").str[0]
            )
        elif "Gene stable ID" in self.crispri_data.columns:
            self.crispri_data["ENSG"] = (
                self.crispri_data["Gene stable ID"].astype(str).str.split(".").str[0]
            )
        elif "ensembl_gene_id" in self.crispri_data.columns:
            self.crispri_data["ENSG"] = (
                self.crispri_data["ensembl_gene_id"].astype(str).str.split(".").str[0]
            )
        else:
            sym_col = "Gene" if "Gene" in self.crispri_data.columns else None
            if sym_col is None and "gene" in self.crispri_data.columns:
                sym_col = "gene"
            if sym_col is None:
                raise KeyError(
                    "ziga_additional_columns.tsv needs gene_id, Gene stable ID, ensembl_gene_id, "
                    "or Gene/gene (symbol) column."
                )
            print(
                "Resolving Ensembl IDs from column {!r} via Ensembl REST (Karollus Zenodo ziga)...".format(
                    sym_col)
            )
            sym_df = _fulco_ensembl_lookup_symbols(
                self.crispri_data[sym_col].dropna().unique().tolist(),
                genome_build=self.genome_build,
            )
            self.crispri_data = self.crispri_data.merge(
                sym_df, left_on=sym_col, right_on="resolved_symbol", how="inner"
            )
            self.crispri_data.drop(
                columns=[c for c in ("resolved_symbol",) if c in self.crispri_data.columns],
                inplace=True,
            )
            self.crispri_data["ENSG"] = self.crispri_data["Gene stable ID"].astype(str).str.split(".").str[0]
            self.crispri_data["Strand_ensembl"] = self.crispri_data["Strand_ensembl"].astype(int)
            self.crispri_data.drop(columns=["Gene stable ID"], inplace=True, errors="ignore")

        # Expression remaining (parallel to Gasperini Diff_expression_test_fold_change)
        fc = pd.to_numeric(self.crispri_data[self.fraction_change_col], errors="coerce")
        self.crispri_data["Diff_expression_test_fold_change"] = 1.0 + fc
        # Benchmark y_delta = 1 - expr_ratio = -fraction_change → > 0 iff measured knockdown.
        if self.observed_subset == "knockdown_only":
            n_fc = len(self.crispri_data)
            kd = fc < 0
            self.crispri_data = self.crispri_data.loc[kd].copy().reset_index(drop=True)
            print(
                "fulco observed_subset=knockdown_only: kept {}/{} rows (Fraction change in gene expr < 0).".format(
                    len(self.crispri_data), n_fc
                )
            )

        # Gene symbol for outputs
        if "Gene" in self.crispri_data.columns:
            self.crispri_data["target_gene_short"] = self.crispri_data["Gene"].astype(str)
        else:
            self.crispri_data["target_gene_short"] = self.crispri_data["ENSG"]

        # TSS from Avsec/Karollus windows (not Biomart) so coordinates match their pipeline
        if "main_tss_start" not in self.crispri_data.columns or "main_tss_end" not in self.crispri_data.columns:
            raise KeyError("ziga table must contain main_tss_start and main_tss_end.")
        self.crispri_data["tss_pos"] = (
            (self.crispri_data["main_tss_start"].astype(int) + self.crispri_data["main_tss_end"].astype(int)) // 2
        )

        # Enhancer interval (narrow CRE coordinates used in Karollus et al. overlap analyses)
        if "enhancer_start" in self.crispri_data.columns and "enhancer_end" in self.crispri_data.columns:
            es = self.crispri_data["enhancer_start"].astype(int)
            ee = self.crispri_data["enhancer_end"].astype(int)
        elif "fix_enhancer_wide_start" in self.crispri_data.columns:
            es = self.crispri_data["fix_enhancer_wide_start"].astype(int)
            ee = self.crispri_data["fix_enhancer_wide_end"].astype(int)
        else:
            raise KeyError(
                "Need enhancer_start/enhancer_end or fix_enhancer_wide_start/fix_enhancer_wide_end in ziga table."
            )

        chrom_series = self.crispri_data["chromosome"].map(self._normalize_chrom)
        self.crispri_data["Chromosome/scaffold name"] = chrom_series.str.replace("chr", "", regex=False)
        self.crispri_data["enh_loc"] = (
            chrom_series + ":" + es.astype(str) + "-" + ee.astype(str)
        )

        # Strand: prefer Biomart CSV when provided; else use strand from Ensembl symbol lookup.
        tss_csv = self.gene_tss_csv
        if tss_csv is None:
            cand = os.path.join(self.data_path, "Fulco_gene_tss_hg38_biomart_GRCh38_p14.csv")
            tss_csv = cand if os.path.exists(cand) else None

        if tss_csv is not None and os.path.isfile(tss_csv):
            gene_meta = pd.read_csv(tss_csv)
            if "Gene stable ID" not in gene_meta.columns or "Strand" not in gene_meta.columns:
                raise KeyError("gene_tss_csv must include 'Gene stable ID' and 'Strand' columns.")
            gm = gene_meta[["Gene stable ID", "Strand"]].drop_duplicates()
            gm["Gene stable ID"] = gm["Gene stable ID"].astype(str).str.split(".").str[0]
            self.crispri_data = self.crispri_data.merge(
                gm.rename(columns={"Strand": "Strand_biomart"}),
                left_on="ENSG",
                right_on="Gene stable ID",
                how="left",
            )
            if "Strand_ensembl" in self.crispri_data.columns:
                self.crispri_data["Strand"] = self.crispri_data["Strand_biomart"].where(
                    self.crispri_data["Strand_biomart"].notna(),
                    self.crispri_data["Strand_ensembl"],
                )
                drop_cols = ["Strand_biomart", "Strand_ensembl", "Gene stable ID"]
            else:
                self.crispri_data["Strand"] = self.crispri_data["Strand_biomart"]
                drop_cols = ["Strand_biomart", "Gene stable ID"]
            self.crispri_data.drop(
                columns=[c for c in drop_cols if c in self.crispri_data.columns],
                inplace=True,
                errors="ignore",
            )
        elif "Strand_ensembl" in self.crispri_data.columns:
            self.crispri_data["Strand"] = self.crispri_data["Strand_ensembl"]
            self.crispri_data.drop(columns=["Strand_ensembl"], inplace=True, errors="ignore")
        else:
            raise FileNotFoundError(
                "No gene_tss_csv for strand, and ziga has no gene_id (Ensembl symbol lookup was not run). "
                "Pass --gene_tss_csv with Gene stable ID + Strand, or use a ziga file that includes gene_id."
            )

        self.crispri_data = self.crispri_data.dropna(subset=["Strand"])
        self.crispri_data["Strand"] = self.crispri_data["Strand"].astype(int)
        self.crispri_data.reset_index(drop=True, inplace=True)
        print("Number of Fulco enhancer-gene pairs after merge:", self.crispri_data.shape[0])

    def __len__(self):
        return self.crispri_data.shape[0]

    def _filt_enh_receptive_field(self):
        enh_parts = self.crispri_data["enh_loc"].str.split(":", expand=True)
        enh_coords = enh_parts[1].str.split("-", expand=True)
        enh_start = enh_coords[0].astype(int)
        enh_end = enh_coords[1].astype(int)
        self.crispri_data["enh_len"] = (enh_end - enh_start).abs()
        self.crispri_data["enh_middle"] = (enh_start + enh_end) // 2
        self.crispri_data["enh_dist"] = abs(self.crispri_data["tss_pos"] - self.crispri_data["enh_middle"])

        self.crispri_data = self.crispri_data[
            self.crispri_data["enh_dist"] <= self.sequence_length // 2
        ]
        if self.min_enh_dist > 0:
            n_before = self.crispri_data.shape[0]
            self.crispri_data = self.crispri_data[
                self.crispri_data["enh_dist"] > self.min_enh_dist
            ]
            print(
                "min_enh_dist={} bp: removed {} near-promoter pairs, {} remaining.".format(
                    self.min_enh_dist, n_before - self.crispri_data.shape[0], self.crispri_data.shape[0]
                )
            )
        assert self.crispri_data.shape[0] > 0, "No Fulco pairs within the model receptive field"
        print(
            "Number of Fulco enhancer-gene pairs within receptive field ({} bp): {}".format(
                self.sequence_length, self.crispri_data.shape[0]
            )
        )

    def _reverse_complement_dna(self, seq):
        # seq shape: (N, seq_len, 4)  dim1=positions, dim2=channels [ACGT]
        # Reverse complement = flip positions (dim 1) AND complement bases (dim 2, ACGT→TGCA)
        return torch.flip(seq, dims=[1, 2])

    def _crispri(self, seq, cCRE_i, seq_window_start_genomic, crispri_effect_size_i, pair_idx=0):
        seq = seq.swapaxes(1, 2)
        start_pos = cCRE_i - crispri_effect_size_i // 2
        rel_pos = max(0, start_pos - seq_window_start_genomic)
        silenced_seq = seq[:, :, rel_pos : rel_pos + crispri_effect_size_i]
        L = silenced_seq.shape[-1]
        all_shuffled = []

        if self.crispri_perturb_mode == "dinucleotide":
            silenced_seq = dinucleotide_shuffle(silenced_seq, allow_N=True, n=self.N)
            for n_i in range(self.N):
                silenced_seq_i = silenced_seq[:, n_i, :, :]
                seq_crispri_i = torch.concat(
                    [seq[:, :, 0:rel_pos], silenced_seq_i, seq[:, :, rel_pos + crispri_effect_size_i :]],
                    axis=2,
                )
                seq_crispri_i = seq_crispri_i.swapaxes(1, 2)
                all_shuffled.append(seq_crispri_i)
        else:
            dev = silenced_seq.device
            for n_i in range(self.N):
                g = torch.Generator()
                g.manual_seed(int(self.seed) + int(pair_idx) * 1_000_003 + n_i)
                perm = torch.randperm(L, generator=g).to(device=dev, dtype=torch.long)
                silenced_seq_i = silenced_seq[:, :, perm]
                seq_crispri_i = torch.concat(
                    [seq[:, :, 0:rel_pos], silenced_seq_i, seq[:, :, rel_pos + crispri_effect_size_i :]],
                    axis=2,
                )
                seq_crispri_i = seq_crispri_i.swapaxes(1, 2)
                all_shuffled.append(seq_crispri_i)

        return torch.cat(all_shuffled, axis=0)

    def _tss_seq_index(self, tss_genomic, seq_start, strand_sign, chrom_scaffold_name):
        L = self.sequence_length
        pad_N_strt = max(seq_start * -1, 0)
        start = max(0, seq_start)
        chrom = "chr" + str(chrom_scaffold_name)
        chrom_len = self.genome.genome_dat.get_reference_length(chrom)
        end = min(seq_start + self.sequence_length + self.genome.mod, chrom_len)

        if tss_genomic < start:
            i_pre = 0
        elif tss_genomic >= end:
            i_pre = L - 1
        else:
            i_pre = pad_N_strt + (tss_genomic - start)
        i_pre = int(max(0, min(L - 1, i_pre)))
        if strand_sign < 1:
            return L - 1 - i_pre
        return i_pre

    def _build_window(self, data, idx, shift):
        """Build one (WT, CRISPRi) window shifted ``shift`` bp from the TSS-centred start.

        The window is re-fetched from the genome at ``seq_start + shift``, so the
        flanking bases are real sequence (not rolled/wrapped). Returns
        ``(x_WT (1, L, 4), x_crispri (N, L, 4), tss_seq_index, seq_start)``.
        """
        tss = int(data["tss_pos"])
        enh_middle = int(data["enh_middle"])
        strand_i = int(data["Strand"])
        seq_start = tss - (self.sequence_length // 2) + int(shift)
        chrom = "chr" + str(data["Chromosome/scaffold name"])

        seqs = self.genome.get_seq_start(
            chrom=chrom,
            seq_start=seq_start,
            strand="+",
            ohe=True,
            rev_comp=False,
            pad_seq=True,
        ).swapaxes(1, 2).to(torch.float32)
        seqs_wt = seqs.clone()
        # _crispri locates the enhancer via (enh_middle - seq_start), so it tracks the shift.
        seqs = self._crispri(seqs, enh_middle, seq_start, self.enhancer_bp, pair_idx=idx)

        if strand_i < 1:
            seqs = self._reverse_complement_dna(seqs)
            seqs_wt = self._reverse_complement_dna(seqs_wt)

        tss_seq_index = self._tss_seq_index(
            tss, seq_start, strand_i, data["Chromosome/scaffold name"]
        )
        return seqs_wt, seqs, tss_seq_index, seq_start

    def __getitem__(self, idx):
        self._init_genome_loaders()
        data = self.crispri_data.iloc[idx]
        tss = int(data["tss_pos"])
        enh_middle = int(data["enh_middle"])
        strand_i = int(data["Strand"])

        meta = {
            "y_delta": torch.tensor(data["Diff_expression_test_fold_change"], dtype=torch.float32),
            "gene_id": data["ENSG"],
            "gene_name": data["target_gene_short"],
            "enh_loc": data["enh_loc"],
            "tss": tss,
            "enh_middle": enh_middle,
            "strand": strand_i,
            "enh_dist": float(data["enh_dist"]),
        }

        if self.tta_shifts == (0,):
            seqs_wt, seqs, tss_seq_index, _ = self._build_window(data, idx, 0)
            meta["x_WT"] = seqs_wt
            meta["x_crispri"] = seqs
            meta["tss_seq_index"] = tss_seq_index
            return meta

        # TTA: one real-flank window per shift, stacked along a leading shift axis.
        wts, crs, tss_idxs, seq_starts = [], [], [], []
        for s in self.tta_shifts:
            wt, cr, tss_idx, ss = self._build_window(data, idx, s)
            wts.append(wt)         # (1, L, 4)
            crs.append(cr)         # (N, L, 4)
            tss_idxs.append(int(tss_idx))
            seq_starts.append(int(ss))
        meta["x_WT"] = torch.cat(wts, dim=0)             # (S, L, 4)
        meta["x_crispri"] = torch.stack(crs, dim=0)      # (S, N, L, 4)
        meta["tss_seq_index"] = torch.tensor(tss_idxs, dtype=torch.long)   # (S,)
        meta["seq_start"] = torch.tensor(seq_starts, dtype=torch.long)     # (S,)
        meta["tta_shifts"] = torch.tensor(self.tta_shifts, dtype=torch.long)
        return meta


