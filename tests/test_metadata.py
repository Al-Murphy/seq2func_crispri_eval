"""
Smoke tests for the shipped metadata files under ``metadata/``.

These guard against accidental commits of empty / corrupted reference tables
and document the expected column schemas.
"""

import pandas as pd
import pytest


def test_borzoi_targets_loads(metadata_dir):
    path = metadata_dir / "borzoi_human_targets.txt"
    assert path.exists()
    df = pd.read_csv(path, sep="\t")
    assert "description" in df.columns
    assert "file" in df.columns
    # Borzoi's human head has 7611 tracks
    assert len(df) == 7611


def test_borzoi_k562_rna_subset_nonempty(metadata_dir):
    """At least a handful of K562 RNA-Seq tracks should be present."""
    df = pd.read_csv(metadata_dir / "borzoi_human_targets.txt", sep="\t")
    k562_rna = df[
        df["description"].astype(str).str.startswith("RNA:")
        & df["description"].astype(str).str.contains("K562", case=False, na=False)
    ]
    assert len(k562_rna) > 50, f"expected ~94 K562 RNA tracks, got {len(k562_rna)}"


def test_alphagenome_track_metadata_loads(metadata_dir):
    path = metadata_dir / "track_metadata.parquet"
    assert path.exists()
    df = pd.read_parquet(path)
    for col in ("organism", "output_type"):
        assert col in df.columns, f"missing column: {col}"
    # Human RNA-Seq head should have 768 tracks; CAGE head 640
    human = df[df["organism"].str.lower() == "human"]
    rna = human[human["output_type"].str.lower() == "rna_seq"]
    cage = human[human["output_type"].str.lower() == "cage"]
    assert len(rna) == 768, f"expected 768 human RNA-Seq tracks, got {len(rna)}"
    assert len(cage) == 640, f"expected 640 human CAGE tracks, got {len(cage)}"


def test_alphagenome_k562_nonempty(metadata_dir):
    df = pd.read_parquet(metadata_dir / "track_metadata.parquet")
    human_rna = df[
        (df["organism"].str.lower() == "human")
        & (df["output_type"].str.lower() == "rna_seq")
    ]
    search_col = "biosample_name" if "biosample_name" in human_rna.columns else "track_name"
    k562 = human_rna[human_rna[search_col].str.contains("K562", case=False, na=False)]
    assert len(k562) > 0


def test_gasperini_tss_csv_loads(metadata_dir):
    df = pd.read_csv(metadata_dir / "Gasperini_gene_tss_hg38_biomart_GRCh38_p14.csv")
    # Biomart export: Gene stable ID + Strand columns
    assert any("Gene stable ID" in c or "ENSG" in c for c in df.columns)
    assert "Strand" in df.columns or "strand" in df.columns
    assert len(df) > 50


def test_gasperini_sig_pairs_loads(metadata_dir):
    df = pd.read_csv(metadata_dir / "Gasperini_sign_enhancer_gene_pairs.csv")
    assert len(df) > 100, f"expected hundreds of significant pairs, got {len(df)}"


def test_gasperini_liftover_bed_loads(metadata_dir):
    """UCSC liftover output here is a single-column ``chrN:start-end`` list."""
    import re
    path = metadata_dir / "Gasperini_enh_hg38_ucscLiftOver.bed"
    assert path.exists()
    df = pd.read_csv(path, sep="\t", header=None, names=["locus"])
    assert len(df) > 100
    # Every line should look like 'chr<N>:<start>-<end>'
    pat = re.compile(r"^chr[\w]+:\d+-\d+$")
    assert df["locus"].astype(str).str.match(pat).all()


def test_fulco_ziga_loads(metadata_dir):
    df = pd.read_csv(metadata_dir / "fulco" / "ziga_additional_columns.tsv", sep="\t")
    assert len(df) > 100


def test_fulco_knockdown_loads(metadata_dir):
    df = pd.read_csv(metadata_dir / "fulco" / "enhancer_knockdown_effects.tsv", sep="\t")
    assert len(df) > 100
    assert "Fraction change in gene expr" in df.columns


@pytest.mark.parametrize("name", ["Fulco_gene_exons_hg38.csv", "Gasperini_gene_exons_hg38.csv"])
def test_exon_csv_schema(metadata_dir, name):
    df = pd.read_csv(metadata_dir / name)
    for col in ("gene_id", "chrom", "start", "end"):
        assert col in df.columns, f"{name} missing column: {col}"
    # At least one row per gene; each gene has multiple exons
    n_genes = df["gene_id"].nunique()
    assert n_genes >= 5
    assert len(df) > n_genes  # multiple exons per gene
