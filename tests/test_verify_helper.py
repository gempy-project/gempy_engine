import pytest

from approvaltests.reporters import GenericDiffReporter, PythonNativeReporter

from tests.verify_helper import get_approval_reporter


@pytest.mark.parametrize("ci_variable", [None, "CI", "TEAMCITY_VERSION"])
@pytest.mark.parametrize("interactive", [None, "False", "True"])
def test_approval_reporter_requires_explicit_interactive_opt_in(monkeypatch, ci_variable, interactive):
    for name in ("CI", "TEAMCITY_VERSION", "GEMPY_TEST_PLOTS"):
        monkeypatch.delenv(name, raising=False)
    if ci_variable:
        monkeypatch.setenv(ci_variable, "1")
    if interactive is not None:
        monkeypatch.setenv("GEMPY_TEST_PLOTS", interactive)

    reporter = get_approval_reporter()

    expected_type = GenericDiffReporter if interactive == "True" and ci_variable is None else PythonNativeReporter
    assert isinstance(reporter, expected_type)
