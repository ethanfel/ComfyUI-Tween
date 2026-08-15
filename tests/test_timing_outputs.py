import ast
from pathlib import Path


NODE_SOURCE = Path(__file__).resolve().parents[1] / "nodes.py"
TIMED_CLASSES = {
    "BIMVFIInterpolate",
    "BIMVFISegmentInterpolate",
    "EMAVFIInterpolate",
    "EMAVFISegmentInterpolate",
    "SGMVFIInterpolate",
    "SGMVFISegmentInterpolate",
    "LDFVFIInterpolate",
    "SPEEDVFIInterpolate",
    "SPEEDVFISegmentInterpolate",
    "GIMMVFIInterpolate",
    "GIMMVFISegmentInterpolate",
}
DIRECTLY_DECORATED = TIMED_CLASSES - {
    "SPEEDVFIInterpolate",
    "SPEEDVFISegmentInterpolate",
}


def _class_definitions():
    tree = ast.parse(NODE_SOURCE.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in TIMED_CLASSES
    }


def _literal_assignment(class_node, name):
    for statement in class_node.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
        ):
            return ast.literal_eval(statement.value)
    raise AssertionError(f"{class_node.name} does not define {name}")


def test_all_interpolation_nodes_expose_elapsed_seconds_last():
    classes = _class_definitions()
    assert classes.keys() == TIMED_CLASSES

    for class_node in classes.values():
        assert _literal_assignment(class_node, "RETURN_TYPES")[-1] == "FLOAT"
        assert _literal_assignment(class_node, "RETURN_NAMES")[-1] == "elapsed_seconds"


def test_direct_interpolation_methods_append_timing_output():
    classes = _class_definitions()
    for class_name in DIRECTLY_DECORATED:
        interpolate = next(
            statement
            for statement in classes[class_name].body
            if isinstance(statement, ast.FunctionDef)
            and statement.name == "interpolate"
        )
        assert any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "_with_elapsed_seconds"
            for decorator in interpolate.decorator_list
        )
