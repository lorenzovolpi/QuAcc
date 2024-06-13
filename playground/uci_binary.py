from quapy.data.datasets import UCI_BINARY_DATASETS

from quacc.dataset import DatasetProvider as DP

if __name__ == "__main__":
    for dn in UCI_BINARY_DATASETS:
        L, V, U = DP.uci_binary(dn)
        print(f"{dn}: L={len(L)}; V={len(V)}; U={len(U)}")
