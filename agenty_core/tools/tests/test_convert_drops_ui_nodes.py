"""graph -> API leaves out what the frontend's graphToPrompt leaves out: notes and Reroutes.

LTX-2.5's VFX example workflows (ComfyUI-LTXVideo, example_workflows/2.5) carry
MarkdownNotes, a Note and a Reroute inside and beside their subgraphs; registered
as templates they kept all nine and failed validation as unknown class_types.
"""
import unittest
from unittest import mock

from agenty_core.tools import comfyui as C

GRAPH = {
    "last_node_id": 5, "last_link_id": 2,
    "nodes": [
        {"id": 1, "type": "LoadVideo", "inputs": [], "widgets_values": ["clip.mp4", "image"],
         "outputs": [{"name": "VIDEO", "type": "VIDEO", "links": [1]}]},
        {"id": 2, "type": "Reroute", "inputs": [{"name": "", "type": "*", "link": 1}],
         "outputs": [{"name": "", "type": "VIDEO", "links": [2]}]},
        {"id": 3, "type": "SaveVideo", "inputs": [{"name": "video", "type": "VIDEO", "link": 2}],
         "widgets_values": ["out", "auto", "auto"], "outputs": []},
        {"id": 4, "type": "MarkdownNote", "inputs": [], "outputs": [], "widgets_values": ["## Inputs"]},
        {"id": 5, "type": "Note", "inputs": [], "outputs": [], "widgets_values": ["read me"]},
    ],
    "links": [[1, 1, 0, 2, 0, "VIDEO"], [2, 2, 0, 3, 0, "VIDEO"]],
}


class ConvertDropsUiNodes(unittest.TestCase):
    def test_notes_and_reroutes_are_left_out(self):
        with mock.patch.object(C, "_get_object_info", return_value={}):
            api = C._convert_graph_to_api(GRAPH)
        self.assertEqual(sorted(api), ["1", "3"])
        # the Reroute's consumer is wired to the Reroute's own source
        self.assertEqual(api["3"]["inputs"]["video"], ["1", 0])


if __name__ == "__main__":
    unittest.main()
