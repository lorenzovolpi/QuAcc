import numpy as np

from quacc.evaluation.report import DatasetReport

datasets = [
    "imdb/imdb.pickle",
    "rcv1_CCAT/rcv1_CCAT.pickle",
    "rcv1_GCAT/rcv1_GCAT.pickle",
    "rcv1_MCAT/rcv1_MCAT.pickle",
]

gs = {
    "sld_lr_gs": [
        "bin_sld_lr_gs",
        "mul_sld_lr_gs",
        "m3w_sld_lr_gs",
    ],
    "kde_lr_gs": [
        "bin_kde_lr_gs",
        "mul_kde_lr_gs",
        "m3w_kde_lr_gs",
    ],
}

for dst in datasets:
    dr = DatasetReport.unpickle("output/main/" + dst)
    print(f"{dst}\n")
    for name, methods in gs.items():
        print(f"{name}")
        sel_methods = [
            {k: v for k, v in cr.fit_scores.items() if k in methods} for cr in dr.crs
        ]

        best_methods = [
            list(ms.keys())[np.argmin(list(ms.values()))] for ms in sel_methods
        ]
        m_cnt = []
        for m in methods:
            m_cnt.append((np.array(best_methods) == m).nonzero()[0].shape[0])
        m_cnt = np.array(m_cnt)
        m_freq = m_cnt / len(best_methods)

        for n in methods:
            print(n, end="\t")
        print()
        for v in m_freq:
            print(f"{v*100:.2f}", end="\t")
        print("\n\n")
