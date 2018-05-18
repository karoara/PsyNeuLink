import psyneulink as pnl
import pytest


# (ancestor, child, should_override)
ancestor_child_data = [
    (pnl.Component, pnl.TransferMechanism, False),
    (pnl.Component, pnl.OutputState, False),
    (pnl.Component, pnl.InputState, True),
    (pnl.Component, pnl.SimpleIntegrator, False),
    (pnl.Function_Base, pnl.SimpleIntegrator, True),
    (pnl.TransferMechanism, pnl.RecurrentTransferMechanism, False)
]


@pytest.fixture(scope='function')
def reset_variable(*args):
    yield
    # pytest cannot provide the exact parametrized arguments to fixtures
    # so just reset all of the possibilities
    # this must be used when altering class level defaults
    for item in ancestor_child_data:
        item[0].parameters.variable.reset()
        item[1].parameters.variable.reset()


def test_parameter_assignment_by_value(reset_variable):
    old_var = pnl.Component.parameters.variable
    pnl.Component.parameters.variable = 2

    assert pnl.Component.parameters.variable.default_value == 2
    assert pnl.Component.parameters.variable.default_value != old_var.default_value


def test_parameter_value_reset(reset_variable):
    old_var = pnl.Component.parameters.variable
    pnl.Component.parameters.variable = 2

    pnl.Component.parameters.variable.reset()

    assert pnl.Component.parameters.variable.default_value == old_var.default_value


@pytest.mark.parametrize('ancestor, child', [(item[0], item[1]) for item in ancestor_child_data])
def test_parameter_propagation(ancestor, child):
    for param in ancestor.parameters:
        child_params = child.parameters.values(show_all=True)

        assert param.name in child_params


@pytest.mark.parametrize('ancestor, child, should_override', ancestor_child_data)
def test_parameter_values_overriding(ancestor, child, should_override, reset_variable):
    original_child_variable = child.parameters.variable.default_value

    # ancestor updates
    ancestor.parameters.variable = -1
    assert ancestor.parameters.variable.default_value == -1

    if should_override:
        assert child.parameters.variable.default_value == -1
    else:
        assert child.parameters.variable.default_value == original_child_variable

    # child updates and ancestor does not update
    child.parameters.variable = -2
    assert child.parameters.variable.default_value == -2
    assert ancestor.parameters.variable.default_value == -1

    # child should not get overridden because it is explicitly specified
    ancestor.parameters.variable = -3
    assert child.parameters.variable.default_value == -2

    # revert to original behavior
    child.parameters.variable.reset()
    if should_override:
        assert child.parameters.variable.default_value == -3
    else:
        assert child.parameters.variable.default_value == original_child_variable
