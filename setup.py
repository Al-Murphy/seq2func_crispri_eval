from setuptools import find_packages, setup

setup(
    name="crispri_eval",
    version="0.1.0",
    description=(
        "Zero-shot CRISPRi enhancer-perturbation evaluation for genomic "
        "deep-learning models (Enformer, Borzoi, NTv3, AlphaGenome) on the "
        "Fulco and Gasperini screens."
    ),
    author="Alan Murphy",
    license="MIT",
    packages=find_packages(include=["crispri_eval", "crispri_eval.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "tqdm",
        "pysam",
        "tangermeme",
    ],
    extras_require={
        "plotting": ["matplotlib", "seaborn"],
        "enformer": ["enformer-pytorch"],
        "borzoi": ["borzoi-pytorch"],
        "ntv3": ["transformers"],
        "alphagenome": ["alphagenome-pytorch", "clgenomics"],
    },
)
