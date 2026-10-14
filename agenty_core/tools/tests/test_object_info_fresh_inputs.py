"""A file staged into the input dir is a valid LoadImage value at once.

The cached ``/object_info`` lists the input dir as it was when it was fetched, so
every update_workflow after an upload_image "snapped" the new file back to an old
one, and the agent kept putting it back with a script. The cache now follows the
input dir, and a staging upload drops it outright.

    python -m unittest agenty_core.tools.tests.test_object_info_fresh_inputs
"""

import os
import tempfile
import time
import unittest
from unittest import mock

from agenty_core.tools import comfyui, image_io


class _Client:
    def __init__(self):
        self.gets = 0
        self.posts = 0

    def get(self, path):
        self.gets += 1
        return {"LoadImage": {"n": self.gets}}

    def post(self, path, data=None, files=None):
        self.posts += 1
        return {"name": files["image"][0], "subfolder": "", "type": "input"}


class TheCacheFollowsTheInputDir(unittest.TestCase):

    def setUp(self):
        comfyui.reset_object_info_cache()
        self.client = _Client()
        self.dir = tempfile.mkdtemp()
        patches = [mock.patch.object(comfyui, "get_client", return_value=self.client),
                   mock.patch.object(image_io, "comfy_input_dir", return_value=self.dir)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(comfyui.reset_object_info_cache)

    def test_unchanged_dir_is_one_fetch(self):
        comfyui._get_object_info()
        comfyui._get_object_info()
        self.assertEqual(self.client.gets, 1)

    def test_a_new_file_fetches_again(self):
        comfyui._get_object_info()
        later = time.time() + 5
        with open(os.path.join(self.dir, "new.png"), "wb") as fh:
            fh.write(b"x")
        os.utime(self.dir, (later, later))
        self.assertEqual(comfyui._get_object_info()["LoadImage"]["n"], 2)

    def test_a_canvas_copy_in_input_agentY_fetches_again(self):
        sub = os.path.join(self.dir, "agentY")
        os.mkdir(sub)
        comfyui._get_object_info()
        later = time.time() + 5
        os.utime(sub, (later, later))
        comfyui._get_object_info()
        self.assertEqual(self.client.gets, 2)

    def test_no_local_dir_keeps_the_cache(self):
        with mock.patch.object(image_io, "comfy_input_dir", return_value=None):
            comfyui._get_object_info()
            comfyui._get_object_info()
        self.assertEqual(self.client.gets, 1)


class AnUploadDropsTheCache(unittest.TestCase):

    def test_after_stage_image_the_next_read_is_fresh(self):
        client = _Client()
        src = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        src.write(b"png")
        src.close()
        self.addCleanup(os.unlink, src.name)
        with mock.patch.object(comfyui, "get_client", return_value=client), \
                mock.patch.object(image_io, "get_client", return_value=client), \
                mock.patch.object(image_io, "comfy_input_dir", return_value=None):
            comfyui.reset_object_info_cache()
            comfyui._get_object_info()
            image_io.stage_image(src.name)
            comfyui._get_object_info()
        comfyui.reset_object_info_cache()
        self.assertEqual(client.posts, 1)
        self.assertEqual(client.gets, 2)


if __name__ == "__main__":
    unittest.main()
