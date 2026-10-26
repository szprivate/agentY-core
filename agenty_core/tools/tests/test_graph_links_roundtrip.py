"""UI-format graphs must keep their wires: the frontend reads links as arrays at version 0.4."""
import json
import unittest
import uuid
from unittest import mock

from agenty_core.tools import comfyui as C

OBJ_GRAPH = {
    "id": "not-a-uuid", "version": 0.4, "last_node_id": 1, "last_link_id": 0,
    "nodes": [
        {"id": 10, "type": "A", "inputs": [], "outputs": [{"name": "o", "type": "IMAGE", "links": []}]},
        {"id": 11, "type": "B", "inputs": [{"name": "image", "type": "IMAGE", "link": None}],
         "outputs": [{"name": "o", "type": "IMAGE", "links": None}]},
        {"id": 12, "type": "C", "inputs": [{"name": "image", "type": "IMAGE", "link": None}], "outputs": []},
    ],
    "links": [
        {"id": 5, "origin_id": 10, "origin_slot": 0, "target_id": 11, "target_slot": 0, "type": "IMAGE"},
        {"id": 6, "origin_id": 11, "origin_slot": 0, "target_id": 12, "target_slot": 0, "type": "IMAGE"},
    ],
}


def frontend_load(graph):
    """What LGraph.configure does at version 0.4: LLink.createFromArray(link) for each."""
    out = {}
    for lk in graph["links"]:
        assert isinstance(lk, list) and len(lk) == 6, lk
        out[lk[0]] = (lk[1], lk[2], lk[3], lk[4], lk[5])
    return out


class GraphLinks(unittest.TestCase):
    def test_object_links_become_arrays_and_consistent(self):
        g = C._normalize_graph_links(json.loads(json.dumps(OBJ_GRAPH)))
        self.assertEqual(frontend_load(g), {5: (10, 0, 11, 0, "IMAGE"), 6: (11, 0, 12, 0, "IMAGE")})
        n = {x["id"]: x for x in g["nodes"]}
        self.assertEqual(n[10]["outputs"][0]["links"], [5])
        self.assertEqual(n[11]["inputs"][0]["link"], 5)
        self.assertEqual(n[11]["outputs"][0]["links"], [6])
        self.assertEqual(n[12]["inputs"][0]["link"], 6)
        self.assertEqual((g["last_link_id"], g["last_node_id"]), (6, 12))
        uuid.UUID(g["id"])

    def test_object_links_survive_conversion_to_api(self):
        with mock.patch.object(C, "_get_object_info", return_value={}):
            api = C._convert_graph_to_api(json.loads(json.dumps(OBJ_GRAPH)))
        self.assertEqual(api["11"]["inputs"]["image"], ["10", 0])
        self.assertEqual(api["12"]["inputs"]["image"], ["11", 0])

    def test_api_to_ui_roundtrip_keeps_every_link(self):
        api = {"1": {"class_type": "A", "inputs": {}},
               "2": {"class_type": "B", "inputs": {"image": ["1", 0]}},
               "3": {"class_type": "C", "inputs": {"image": ["2", 0], "other": ["1", 0]}}}
        with mock.patch.object(C, "_get_object_info", return_value={}):
            g = C._normalize_graph_links(C._api_to_graph(api))
            self.assertEqual(len(frontend_load(g)), 3)
            back = C._convert_graph_to_api(g)
        self.assertEqual(back["3"]["inputs"]["image"], ["2", 0])
        self.assertEqual(back["3"]["inputs"]["other"], ["1", 0])
        ids = [l[0] for l in g["links"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_save_workflow_normalizes_ui_file(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d, mock.patch.object(C, "_workflows_dir", return_value=pathlib.Path(d)):
            p = C._save_workflow(json.loads(json.dumps(OBJ_GRAPH)), name="x")
            frontend_load(json.loads(pathlib.Path(p).read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
