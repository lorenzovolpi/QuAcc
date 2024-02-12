from quacc.evaluation.report import DatasetReport
import pandas as pd

dr = DatasetReport.unpickle("output/main/imdb/imdb.pickle")

_data = dr.data(
    metric="acc", estimators=["bin_sld_lr_mc", "bin_sld_lr_ne", "bin_sld_lr_c"]
)
d1 = _data.loc[((0.9, 0.1), (1.0, 0.0), slice(None)), :]
d2 = _data.loc[((0.1, 0.9), (0.0, 1.0), slice(None)), :]
dd = pd.concat([d1, d2], axis=0)

print(d1.to_numpy(), "\n", d1.mean(), "\n")
print(d2.to_numpy(), "\n", d2.mean(), "\n")
print(dd.to_numpy(), "\n", dd.mean(), "\n")
