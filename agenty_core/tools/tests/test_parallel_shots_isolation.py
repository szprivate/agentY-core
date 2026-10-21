"""Parallel shots: no shared graph file, no staged-name collisions (issue #69)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import comfyui, image_io
from agenty_core.utils import turn_scope


class StageImageCollisionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.input = root / "input"
        self.input.mkdir()
        self.a = root / "sh01"; self.a.mkdir()
        self.b = root / "sh02"; self.b.mkdir()
        (self.a / "first.png").write_bytes(b"AAAA-frame-one")
        (self.b / "first.png").write_bytes(b"BBBB-frame-two")  # same size, other bytes
        self.uploads = []

        def fake_post(path, data=None, files=None):
            name = files["image"][0]
            (self.input / name).write_bytes(files["image"][1].read())
            self.uploads.append(name)
            return {"name": name, "subfolder": "", "type": "input"}

        client = mock.Mock(); client.post.side_effect = fake_post
        self._p = [mock.patch.object(image_io, "get_client", return_value=client),
                   mock.patch.object(image_io, "comfy_input_dir", return_value=str(self.input))]
        for p in self._p:
            p.start()

    def tearDown(self):
        for p in self._p:
            p.stop()
        self.tmp.cleanup()

    def test_same_named_frames_from_two_shots_get_distinct_names(self):
        r1 = image_io.stage_image(str(self.a / "first.png"))
        r2 = image_io.stage_image(str(self.b / "first.png"))
        self.assertEqual(r1["name"], "first.png")
        self.assertNotEqual(r2["name"], "first.png")
        self.assertEqual((self.input / r1["name"]).read_bytes(), b"AAAA-frame-one")
        self.assertEqual((self.input / r2["name"]).read_bytes(), b"BBBB-frame-two")

    def test_overwrite_does_not_clobber_another_shots_frame(self):
        image_io.stage_image(str(self.a / "first.png"))
        r2 = image_io.stage_image(str(self.b / "first.png"), overwrite=True)
        self.assertEqual((self.input / "first.png").read_bytes(), b"AAAA-frame-one")
        self.assertEqual((self.input / r2["name"]).read_bytes(), b"BBBB-frame-two")

    def test_restaging_the_same_file_is_still_a_no_op(self):
        image_io.stage_image(str(self.a / "first.png"))
        r = image_io.stage_image(str(self.a / "first.png"))
        self.assertEqual(r["name"], "first.png")
        self.assertIn("skipped", r["note"])
        r2 = image_io.stage_image(str(self.b / "first.png"))
        r3 = image_io.stage_image(str(self.b / "first.png"))
        self.assertEqual(r3["name"], r2["name"])
        self.assertEqual(len(self.uploads), 2)


class TemplateCopyPerConversationTest(unittest.TestCase):
    def test_conversation_tag_differs_per_thread(self):
        tok = turn_scope.enter(turn_scope.Scope("r1", "thread-aaaa1111"))
        try:
            a = comfyui._conversation_tag()
        finally:
            turn_scope.leave(tok)
        tok = turn_scope.enter(turn_scope.Scope("r2", "thread-bbbb2222"))
        try:
            b = comfyui._conversation_tag()
        finally:
            turn_scope.leave(tok)
        self.assertTrue(a and b and a != b)
        self.assertEqual(comfyui._conversation_tag(), "")

    def test_each_conversation_gets_its_own_graph_file(self):
        with tempfile.TemporaryDirectory() as d:
            tpl = {"1": {"class_type": "LoadImage", "inputs": {"image": "x.png"}}}
            comfyui._tool_template_results.clear()
            paths = []
            with mock.patch.object(comfyui, "_workflows_dir", return_value=Path(d)), \
                 mock.patch.object(comfyui, "_force_build", return_value=False), \
                 mock.patch.object(comfyui, "_custom_templates_dir", return_value=Path(d) / "none"), \
                 mock.patch.object(comfyui, "_fetch_template", return_value=tpl), \
                 mock.patch.object(comfyui, "_load_index", return_value=[]):
                for tid in ("thread-aaaa1111", "thread-bbbb2222", "thread-aaaa1111"):
                    tok = turn_scope.enter(turn_scope.Scope("r", tid))
                    try:
                        paths.append(json.loads(comfyui._get_workflow_template_one("t"))["workflow_path"])
                    finally:
                        turn_scope.leave(tok)
            self.assertNotEqual(paths[0], paths[1])
            self.assertEqual(paths[0], paths[2])  # same conversation: cached
            comfyui._tool_template_results.clear()


class InputRefTest(unittest.TestCase):
    def test_a_real_subfolder_reference_keeps_its_path(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "shots" / "sh01").mkdir(parents=True)
            (Path(d) / "shots" / "sh01" / "first.png").write_bytes(b"x")
            with mock.patch.object(image_io, "comfy_input_dir", return_value=d):
                self.assertEqual(comfyui._agent_input_ref("shots/sh01/first.png"),
                                 "shots/sh01/first.png")
                self.assertEqual(comfyui._agent_input_ref("agent/missing.png"), "missing.png")


if __name__ == "__main__":
    unittest.main()
