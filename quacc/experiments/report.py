import os

from genericpath import isfile

from quacc.utils.commons import get_results_path, load_json_file, save_json_file


class TestReport:
    def __init__(
        self,
        basedir,
        cls_name,
        acc_name,
        dataset_name,
        method_name,
    ):
        self.basedir = basedir
        self.cls_name = cls_name
        self.acc_name = acc_name
        self.dataset_name = dataset_name
        self.method_name = method_name

    @property
    def path(self):
        return get_results_path(
            self.basedir,
            self.cls_name,
            self.acc_name,
            self.dataset_name,
            self.method_name,
        )

    def add_result(self, test_prevs, true_accs, estim_accs, t_train, t_test_ave):
        self.test_prevs = test_prevs
        self.true_accs = true_accs
        self.estim_accs = estim_accs
        self.t_train = t_train
        self.t_test_ave = t_test_ave
        return self

    def save_json(self, basedir):
        if not all([hasattr(self, _attr) for _attr in ["true_accs", "estim_accs"]]):
            raise AttributeError("Incomplete report cannot be dumped")

        result = {
            "basedir": self.basedir,
            "cls_name": self.cls_name,
            "acc_name": self.acc_name,
            "dataset_name": self.dataset_name,
            "method_name": self.method_name,
            "test_prevs": self.test_prevs,
            "true_accs": self.true_accs,
            "estim_accs": self.estim_accs,
            "t_train": self.t_train,
            "t_test_ave": self.t_test_ave,
        }

        save_json_file(self.path, result)

    @classmethod
    def load_json(cls, path) -> "TestReport":
        def _test_report_hook(_dict):
            return TestReport(
                basedir=_dict["basedir"],
                cls_name=_dict["cls_name"],
                acc_name=_dict["acc_name"],
                dataset_name=_dict["dataset_name"],
                method_name=_dict["method_name"],
            ).add_result(
                test_prevs=_dict["test_prevs"],
                true_accs=_dict["true_accs"],
                estim_accs=_dict["estim_accs"],
                t_train=_dict["t_train"],
                t_test_ave=_dict["t_test_ave"],
            )

        return load_json_file(path, object_hook=_test_report_hook)


class Report:
    def __init__(self, results: list[TestReport]):
        self.results = results

    @classmethod
    def load_results(cls, basedir):
        def walk_results(path):
            results = []
            if not os.path.exists(path):
                return results

            for f in os.listdir(path):
                n_path = os.path.join(path, f)
                if os.path.isdir(n_path):
                    results += walk_results(n_path)
                if os.path.isfile(n_path) and n_path.endswith(".json"):
                    results.append(TestReport.load_json(n_path))

            return results

        _path = os.path.join("results", basedir)
        _results = walk_results(_path)
        return Report(results=_results)

    def _filter_by_dataset(self):
        pass

    def _filer_by_acc(self):
        pass

    def _filter_by_methods(self):
        pass

    def train_table(self):
        pass

    def test_table(self):
        pass

    def shift_table(self):
        pass
