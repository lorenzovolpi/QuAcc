import os

from quacc.experiments.util import getpath
from quacc.utils.commons import load_json_file, save_json_file


class TestReport:
    def __init__(
        self,
        cls_name,
        acc_name,
        dataset_name,
        method_name,
    ):
        self.cls_name = cls_name
        self.acc_name = acc_name
        self.dataset_name = dataset_name
        self.method_name = method_name

    def path(self, basedir):
        return getpath(
            basedir, self.cls_name, self.acc_name, self.dataset_name, self.method_name
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
            "cls_name": self.cls_name,
            "acc_name": self.acc_name,
            "dataset_name": self.dataset_name,
            "method_name": self.method_name,
            "t_train": self.t_train,
            "t_test_ave": self.t_test,
            "true_accs": self.true_accs,
            "estim_accs": self.estim_accs,
        }

        result_path = self.path(basedir)
        save_json_file(result_path, result)

    @classmethod
    def load_json(cls, path) -> "TestReport":
        def _test_report_hook(_dict):
            return TestReport(
                cls_name=_dict["cls_name"],
                acc_name=_dict["acc_name"],
                dataset_name=_dict["dataset_name"],
                method_name=_dict["method_name"],
            ).add_result(
                true_accs=_dict["true_accs"],
                estim_accs=_dict["estim_accs"],
                t_train=_dict["t_train"],
                t_test_ave=_dict["t_test_ave"],
            )

        return load_json_file(path, object_hook=_test_report_hook)


class Report:
    def __init__(self, tests: list[TestReport]):
        self.tests = tests

    @classmethod
    def load_tests(cls, path):
        if not os.path.isdir(path):
            raise ValueError("Cannot load test results: invalid directory")

        _tests = []
        for f in os.listdir(path):
            if f.endswith(".json"):
                _tests.append(TestReport.load_json(f))

        return Report(_tests)

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
