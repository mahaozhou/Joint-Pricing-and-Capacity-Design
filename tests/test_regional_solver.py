import json

from model import Model, evaluate
from regional_solver import is_regional_baseline, optimize_regional
from paths import MAIN


def test_scope_is_limited_to_the_continuous_staged_baseline():
    assert is_regional_baseline(Model())
    assert not is_regional_baseline(Model(servers=18))
    assert not is_regional_baseline(Model(recurrent=True))
    assert not is_regional_baseline(Model(elasticity=0.01))
    assert not is_regional_baseline(Model(copay=0.3))


def test_f1_boundary_solution_is_reconstructed_by_original_equilibrium():
    model = Model(demand=572, premium=30, d2=0.1)
    result = optimize_regional(model, "online", scalar_maxfun=700,
                               inner_maxfun=100, outer_maxfun=180)
    checked = evaluate(model, (result["mu_on"], result["p_on"], result["p_off"]))
    assert checked is not None
    assert abs(checked["objective"] - 28058.8216) < 2e-3
    assert checked["region"] == "V"
    assert checked["residual"] < 1e-5


def test_full_factorial_cross_validation_tolerances():
    payload = json.loads(
        (MAIN / "regional_factorial_validation.json").read_text()
    )
    summary = payload["summary"]
    assert summary["count"] == 84
    assert summary["max_absolute_objective_gap"] < 3e-4
    assert summary["max_relative_objective_gap"] < 1e-8
    assert all(row["regional"]["residual"] < 1e-5 for row in payload["rows"])
