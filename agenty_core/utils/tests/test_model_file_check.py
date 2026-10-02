"""Text-encoder files a native loader lists but cannot read are named before the run."""

import json
import os
import struct
import tempfile
import unittest
from unittest import mock

from agenty_core.utils import model_file_check as mfc


def _write(path, keys):
    header = {k: {"dtype": "F32", "shape": [1], "data_offsets": [i * 4, i * 4 + 4]}
              for i, k in enumerate(keys)}
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header).encode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob + b"\0" * 4 * len(keys))


SHARD = ["lm_head.weight", "model.layers.26.mlp.down_proj.weight",
         "model.layers.27.self_attn.q_proj.weight", "model.norm.weight"]
WRAPPER_T5 = ["blocks.0.attn.k.weight", "blocks.0.ffn.fc1.weight", "blocks.23.attn.q.weight",
              "norm.weight", "token_embedding.weight"]
NATIVE_T5 = ["encoder.block.0.layer.0.SelfAttention.k.weight",
             "encoder.block.23.layer.1.DenseReluDense.wo.weight", "shared.weight"]
NATIVE_QWEN = ["model.layers.0.mlp.down_proj.weight", "model.layers.27.mlp.down_proj.weight",
               "visual.merger.mlp.0.weight"]


class KeysProblem(unittest.TestCase):
    def test_shard_and_wrapper_layout_are_named(self):
        self.assertIn("layers 26-27", mfc.keys_problem(SHARD))
        self.assertIn("WanVideoWrapper", mfc.keys_problem(WRAPPER_T5))

    def test_whole_native_files_pass(self):
        self.assertIsNone(mfc.keys_problem(NATIVE_T5))
        self.assertIsNone(mfc.keys_problem(NATIVE_QWEN))
        self.assertIsNone(mfc.keys_problem([]))


class WorkflowErrors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = self.tmp.name
        _write(os.path.join(d, "UMT5", "wrapper.safetensors"), WRAPPER_T5)
        _write(os.path.join(d, "UMT5", "native.safetensors"), NATIVE_T5)
        _write(os.path.join(d, "QWEN", "shard.safetensors"), SHARD)
        with open(os.path.join(d, "broken.safetensors"), "wb") as f:
            f.write(b"\x01")
        p = mock.patch.object(mfc, "_text_encoder_dirs", return_value=[d])
        p.start()
        self.addCleanup(p.stop)
        mfc._CACHE.clear()

    def test_loader_with_unreadable_file_is_an_error_naming_a_readable_one(self):
        wf = {"1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "UMT5\\wrapper.safetensors", "type": "wan"}},
              "2": {"class_type": "DualCLIPLoader",
                    "inputs": {"clip_name1": "UMT5\\native.safetensors", "clip_name2": "QWEN\\shard.safetensors"}}}
        errs = mfc.unloadable_encoder_errors(
            wf, ["UMT5\\wrapper.safetensors", "UMT5\\a_lora_elsewhere.safetensors",
                 "UMT5\\native.safetensors", "QWEN\\shard.safetensors"])
        self.assertNotIn("a_lora_elsewhere", errs[0])
        self.assertEqual(len(errs), 2)
        self.assertIn("Node 1 (CLIPLoader)", errs[0])
        self.assertIn("UMT5\\native.safetensors", errs[0])
        self.assertIn("clip_name2", errs[1])

    def test_other_loaders_wires_and_unknown_files_are_left_alone(self):
        wf = {"1": {"class_type": "LoadWanVideoT5TextEncoder", "inputs": {"model_name": "UMT5\\wrapper.safetensors"}},
              "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": ["9", 0]}},
              "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "nowhere.safetensors"}},
              "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "broken.safetensors"}},
              "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": "UMT5\\native.safetensors"}}}
        self.assertEqual(mfc.unloadable_encoder_errors(wf, []), [])


if __name__ == "__main__":
    unittest.main()
