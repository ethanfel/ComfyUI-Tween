import json
from pathlib import Path


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "example_workflows"
    / "tween_speed_bim_model_lab.json"
)


def _workflow():
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_speed_bim_demo_is_clean_and_current():
    workflow = _workflow()
    node_types = [node["type"] for node in workflow["nodes"]]

    assert "Note" not in node_types
    assert not any("LDF" in node_type for node_type in node_types)
    assert node_types.count("SPEEDVFIInterpolate") == 1
    assert node_types.count("BIMVFIInterpolate") == 1
    assert [group["title"] for group in workflow["groups"]] == [
        "INPUT · 24 FPS / 25 FRAMES",
        "SPEED",
        "BIM-VFI",
    ]


def test_speed_bim_demo_links_are_internally_consistent():
    workflow = _workflow()
    nodes = {node["id"]: node for node in workflow["nodes"]}
    node_ids = set(nodes)
    links = {link[0]: link for link in workflow["links"]}

    assert workflow["last_node_id"] == max(node_ids)
    assert workflow["last_link_id"] == max(links)

    for link_id, source_id, source_slot, target_id, target_slot, _ in workflow["links"]:
        assert link_id in links
        assert source_id in node_ids
        assert target_id in node_ids
        assert link_id in nodes[source_id]["outputs"][source_slot]["links"]
        assert nodes[target_id]["inputs"][target_slot]["link"] == link_id

    for node in workflow["nodes"]:
        for input_slot in node.get("inputs", []):
            if input_slot.get("link") is not None:
                assert input_slot["link"] in links
        for output_slot in node.get("outputs", []):
            for link_id in output_slot.get("links") or []:
                assert link_id in links
