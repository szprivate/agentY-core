"""A loader's media file that ComfyUI does not list yet is never swapped for another.

update_workflow(add_nodes=[LoadVideo file="…camF_wide_from3s.mp4"]) came back "ok"
with file="…camE_follow.mp4": the combo snap took the same-prefix option when the
requested clip had not been uploaded yet. The wrong video would have been rendered.

    python -m unittest agenty_core.tools.tests.test_media_file_never_swapped
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import comfyui as C
from agenty_core.tools.assembly_deterministic import harden_node_inputs

VIDEOS = ["spo_destruction_camE_follow.mp4", "shots\\other.mov"]
REQ = {"file": [VIDEOS, {"video_upload": True}]}
IMAGES = ["agent/a.png", "b.png"]
IMG_REQ = {"image": [IMAGES, {"image_upload": True}]}


class MediaFileNeverSwapped(unittest.TestCase):

    def test_unknown_video_kept_and_reported(self):
        n = {"class_type": "LoadVideo", "inputs": {"file": "spo_destruction_camF_wide_from3s.mp4"}}
        notes, unres = [], []
        harden_node_inputs(n, REQ, [], None, None, notes, unres)
        self.assertEqual(n["inputs"]["file"], "spo_destruction_camF_wide_from3s.mp4")
        self.assertEqual(notes, [])
        self.assertEqual(len(unres), 1)
        self.assertIn("camF_wide_from3s", unres[0])

    def test_unknown_image_kept(self):
        n = {"class_type": "LoadImage", "inputs": {"image": "agent/c.png"}}
        unres = []
        harden_node_inputs(n, IMG_REQ, [], None, None, [], unres)
        self.assertEqual(n["inputs"]["image"], "agent/c.png")
        self.assertEqual(len(unres), 1)

    def test_same_file_in_another_folder_is_found(self):
        n = {"class_type": "LoadVideo", "inputs": {"file": "other.mov"}}
        unres = []
        harden_node_inputs(n, REQ, [], None, None, [], unres)
        self.assertEqual(n["inputs"]["file"], "shots\\other.mov")
        self.assertEqual(unres, [])

    def test_exact_option_untouched(self):
        n = {"class_type": "LoadVideo", "inputs": {"file": VIDEOS[0]}}
        unres = []
        harden_node_inputs(n, REQ, [], None, None, [], unres)
        self.assertEqual(n["inputs"]["file"], VIDEOS[0])
        self.assertEqual(unres, [])

    def test_without_collector_still_not_swapped(self):
        n = {"class_type": "LoadVideo", "inputs": {"file": "new.mp4"}}
        harden_node_inputs(n, REQ, [], None, None, [])
        self.assertEqual(n["inputs"]["file"], "new.mp4")

    def test_plain_enum_swap_is_still_reported(self):
        req = {"mode": [["fast", "slow"]]}
        n = {"class_type": "X", "inputs": {"mode": "fats"}}
        notes = []
        harden_node_inputs(n, req, [], None, None, notes, [])
        self.assertEqual(n["inputs"]["mode"], "fast")
        self.assertEqual(len(notes), 1)


class ThroughUpdateWorkflow(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        info = {"LoadVideo": {"input": {"required": REQ}}}
        for target, value in (("_get_object_info", info), ("_server_validate", {}),
                              ("_workflows_dir", self.tmp / "agentY"),
                              ("_corpus_root", self.tmp / "corpus")):
            patcher = mock.patch.object(C, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_added_loader_keeps_its_file_and_warns(self):
        path = self.tmp / "work" / "w.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
        add = [{"node_id": "41", "class_type": "LoadVideo",
                "inputs": {"file": "spo_destruction_camF_wide_from3s.mp4"}}]
        out = json.loads(C.update_workflow(str(path), add_nodes=json.dumps(add)))
        saved = json.loads(Path(out["workflow_path"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["41"]["inputs"]["file"], "spo_destruction_camF_wide_from3s.mp4")
        self.assertTrue(any("not in ComfyUI" in w and "camF_wide" in w for w in out["warnings"]), out)


if __name__ == "__main__":
    unittest.main()
