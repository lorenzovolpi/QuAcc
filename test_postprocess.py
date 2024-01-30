from quacc.evaluation.report import DatasetReport

dr = DatasetReport.unpickle("output/main/imdb/imdb.pickle")
_estimators = ["sld_lr_gs", "bin_sld_lr_gs", "mul_sld_lr_gs", "m3w_sld_lr_gs"]
_data = dr.data(metric="acc", estimators=_estimators)
for idx, cr in zip(_data.index.unique(0), dr.crs[::-1]):
    print(cr.train_prev)
    print({k: v for k, v in cr.fit_scores.items() if k in _estimators})
    print(_data.loc[(idx, slice(None), slice(None)), :])
