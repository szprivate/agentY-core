"""Where models and node packs go when ComfyUI is not on D: (the Mac, 2026-11-15).

    python -m unittest agenty_core.tools.tests.test_mac_comfyui_paths
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import comfyui, huggingface as hf


class ModelsBaseDir(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "models" / "checkpoints").mkdir(parents=True)
        (self.root / "custom_nodes").mkdir()
        self.env = mock.patch.object(hf, "get_secret", lambda k: None)
        self.app = mock.patch.object(hf, "_app_root", lambda: self.root / "nowhere")
        self.env.start(); self.app.start()

    def tearDown(self):
        self.env.stop(); self.app.stop(); self.tmp.cleanup()

    def test_the_running_comfyuis_models_folder(self):
        fp = {"checkpoints": [str(self.root / "models" / "checkpoints")]}
        with mock.patch.object(hf, "_folder_paths", lambda: fp):
            self.assertEqual(hf._models_base_dir(), self.root / "models")

    def test_never_a_windows_drive_off_windows(self):
        with mock.patch.object(hf, "_folder_paths", lambda: {}), \
             mock.patch.object(comfyui, "_comfyui_root", lambda: None), \
             mock.patch.object(hf.os, "name", "posix"):
            got = hf._models_base_dir()
        self.assertNotIn("D:", str(got))
        self.assertEqual(got, Path.home() / "ComfyUI" / "models")

    def test_custom_nodes_from_comfyuis_folder_list(self):
        fp = {"custom_nodes": [str(self.root / "custom_nodes")]}
        with mock.patch.object(hf, "_folder_paths", lambda: fp), \
             mock.patch.object(comfyui, "_comfyui_root", lambda: None):
            self.assertEqual(comfyui._custom_nodes_dir(), self.root / "custom_nodes")


class ComfyuiRoot(unittest.TestCase):

    def test_a_relative_main_py_is_not_our_own_folder(self):
        client = mock.Mock()
        client.get.return_value = {"system": {"argv": ["main.py", "--port", "8188"]}}
        with mock.patch.object(comfyui, "get_client", lambda: client), \
             mock.patch.object(comfyui, "_dirs_from_comfyui", lambda: {}):
            self.assertIsNone(comfyui._comfyui_root())

    def test_the_base_folder_from_comfyuis_user_dir(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "custom_nodes").mkdir()
            client = mock.Mock()
            client.get.return_value = {"system": {"argv": ["main.py"]}}
            with mock.patch.object(comfyui, "get_client", lambda: client), \
                 mock.patch.object(comfyui, "_dirs_from_comfyui", lambda: {"user_dir": os.path.join(d, "user")}):
                self.assertEqual(comfyui._comfyui_root(), Path(d))

    def test_an_absolute_main_py(self):
        with tempfile.TemporaryDirectory() as d:
            client = mock.Mock()
            client.get.return_value = {"system": {"argv": [os.path.join(d, "main.py")]}}
            with mock.patch.object(comfyui, "get_client", lambda: client):
                self.assertEqual(comfyui._comfyui_root(), Path(d).resolve())


if __name__ == "__main__":
    unittest.main()
