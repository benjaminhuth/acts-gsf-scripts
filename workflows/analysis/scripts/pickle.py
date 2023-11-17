import pickle
from gsfanalysis.pandas_import import *

summary, _ = uproot_to_pandas(
    uproot.open(f"{snakemake.input[0]}:tracksummary"),
    uproot.open(f"{snakemake.input[1]}:trackstates"),
)

with open(snakemake.output[0],"wb") as f:
    pickle.dump(summary, f)
