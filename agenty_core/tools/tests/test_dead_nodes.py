"""The nodes a workflow builds and ComfyUI never runs.

From three real from-scratch builds: two came back carrying nodes whose output
nothing read, and both validated clean — locally and server-side — because
ComfyUI executes a graph backwards from its output nodes, so such a node fails
nothing. It simply does not happen:

* a `GetImageSize` added to derive the resolution, while the latent kept hardcoded
  dimensions — the input image's aspect ratio silently ignored;
* a `VAEDecodeAudio` whose audio never reached `CreateVideo.audio` (an *optional*
  input, so nothing complained) — a silent video out of an audio model;
* a duplicate `VAEDecodeTiled` decoding the same latent as the wired decoder.

The other half of these tests is the false positive that would make the check
worthless: a graph of ten independent load→preview pairs is ten islands and not
one dead node, and a class ComfyUI cannot be asked about is never accused.

Runs under pytest or directly (``python test_dead_nodes.py``).
"""
import unittest

from agenty_core.tools.assembly_deterministic import dead_node_warnings, dead_nodes

# Minimal /object_info: only what tells an output node from an ordinary one.
INFO = {
    "LoadImage": {"output": ["IMAGE", "MASK"]},
    "CheckpointLoaderSimple": {"output": ["MODEL", "CLIP", "VAE"]},
    "KSampler": {"output": ["LATENT"]},
    "VAEDecode": {"output": ["IMAGE"]},
    "VAEDecodeTiled": {"output": ["IMAGE"]},
    "VAEDecodeAudio": {"output": ["AUDIO"]},
    "GetImageSize": {"output": ["INT", "INT"]},
    "LTXVImgToVideoInplace": {"output": ["LATENT"]},
    "CreateVideo": {"output": ["VIDEO"]},
    "LTXVAddGuide": {"output": ["CONDITIONING", "CONDITIONING", "LATENT"]},
    "SaveImage": {"output": [], "output_node": True},
    "PreviewImage": {"output": [], "output_node": True},
    "SaveVideo": {"output": [], "output_node": True},
}


def node(cls, **inputs):
    return {"class_type": cls, "inputs": dict(inputs)}


class ACleanGraph(unittest.TestCase):

    def test_a_straight_chain_has_nothing_dead(self):
        wf = {"1": node("CheckpointLoaderSimple", ckpt_name="x.safetensors"),
              "2": node("KSampler", model=["1", 0], seed=1),
              "3": node("VAEDecode", samples=["2", 0], vae=["1", 2]),
              "4": node("SaveImage", images=["3", 0])}
        self.assertEqual(dead_nodes(wf, INFO), [])

    def test_ten_islands_that_each_end_in_an_output_are_fine(self):
        # A real graph: "show me these ten images". Multiple components is not a
        # defect — being unable to reach ANY output is.
        wf = {}
        for i in range(1, 11):
            wf[str(i)] = node("LoadImage", image=f"{i}.png")
            wf[str(100 + i)] = node("PreviewImage", images=[str(i), 0])
        self.assertEqual(dead_nodes(wf, INFO), [])

    def test_an_output_node_reading_nothing_is_not_called_dead(self):
        # It is broken for another reason (validate_workflow reports the missing
        # required input); it is not dead, and saying both would be noise.
        wf = {"1": node("SaveImage")}
        self.assertEqual(dead_nodes(wf, INFO), [])


class TheThreeRealDefects(unittest.TestCase):

    def test_a_size_node_whose_dimensions_were_hardcoded(self):
        # As built: the image reaches the sampler, and the node that was supposed
        # to make the latent match its shape hangs off the side.
        wf = {"1": node("LoadImage", image="a.png"),
              "2": node("GetImageSize", image=["1", 0]),
              "3": node("LTXVImgToVideoInplace", image=["1", 0], width=1280, height=704),
              "4": node("VAEDecode", samples=["3", 0]),
              "5": node("SaveImage", images=["4", 0])}
        found = dead_nodes(wf, INFO)
        self.assertEqual([d["node_id"] for d in found], ["2"])
        self.assertIn("never", found[0]["problem"])

    def test_audio_that_never_reaches_the_video(self):
        wf = {"1": node("KSampler", seed=1),
              "2": node("VAEDecode", samples=["1", 0]),
              "3": node("VAEDecodeAudio", samples=["1", 0]),
              "4": node("CreateVideo", images=["2", 0], fps=24),  # no `audio`
              "5": node("SaveVideo", video=["4", 0])}
        self.assertEqual([d["node_id"] for d in dead_nodes(wf, INFO)], ["3"])

    def test_a_duplicate_decoder(self):
        wf = {"1": node("KSampler", seed=1),
              "2": node("VAEDecode", samples=["1", 0]),
              "3": node("VAEDecodeTiled", samples=["1", 0]),
              "4": node("SaveImage", images=["2", 0])}
        self.assertEqual([d["node_id"] for d in dead_nodes(wf, INFO)], ["3"])


class AChainOfDeadNodes(unittest.TestCase):

    def test_the_whole_branch_is_reported(self):
        # 7 feeds 8 feeds nothing. Only 8 has no consumer to begin with, so a
        # single pass would report half the branch and the next build would find
        # the rest.
        wf = {"1": node("KSampler", seed=1),
              "2": node("VAEDecode", samples=["1", 0]),
              "3": node("SaveImage", images=["2", 0]),
              "7": node("LoadImage", image="ref.png"),
              "8": node("GetImageSize", image=["7", 0])}
        self.assertEqual(sorted(d["node_id"] for d in dead_nodes(wf, INFO)), ["7", "8"])

    def test_the_root_of_the_branch_is_named_as_the_one_to_fix(self):
        wf = {"1": node("KSampler", seed=1),
              "2": node("VAEDecode", samples=["1", 0]),
              "3": node("SaveImage", images=["2", 0]),
              "7": node("LoadImage", image="ref.png"),
              "8": node("GetImageSize", image=["7", 0])}
        by_id = {d["node_id"]: d for d in dead_nodes(wf, INFO)}
        self.assertIn("8", by_id["7"]["problem"])       # 7 feeds the dead 8
        self.assertNotIn("dead too", by_id["8"]["problem"])


class WhenItCannotKnow(unittest.TestCase):

    def test_no_object_info_means_no_opinion(self):
        wf = {"1": node("LoadImage", image="a.png"),
              "2": node("GetImageSize", image=["1", 0])}
        self.assertEqual(dead_nodes(wf, {}), [])
        self.assertEqual(dead_nodes(wf, None), [])

    def test_an_unknown_class_counts_as_an_output(self):
        # A custom saver this ComfyUI does not report must never be accused of
        # being dead: a check that fires on a working graph gets ignored.
        wf = {"1": node("KSampler", seed=1),
              "2": node("SomeoneElsesWriterNode", samples=["1", 0])}
        self.assertEqual(dead_nodes(wf, INFO), [])

    def test_a_name_that_looks_like_a_saver_counts_as_an_output(self):
        info = dict(INFO, MyCustomSaveThing={"output": []})   # no output_node flag
        wf = {"1": node("KSampler", seed=1),
              "2": node("MyCustomSaveThing", samples=["1", 0])}
        self.assertEqual(dead_nodes(wf, info), [])

    def test_an_empty_or_odd_workflow(self):
        self.assertEqual(dead_nodes({}, INFO), [])
        self.assertEqual(dead_nodes(None, INFO), [])
        self.assertEqual(dead_nodes({"1": "not a node"}, INFO), [])


class TheWarningLines(unittest.TestCase):

    def test_each_line_names_the_node_its_class_and_its_title(self):
        wf = {"1": node("KSampler", seed=1),
              "2": node("VAEDecode", samples=["1", 0]),
              "3": node("SaveImage", images=["2", 0]),
              "9": node("VAEDecodeTiled", samples=["1", 0])}
        wf["9"]["_meta"] = {"title": "VAE Decode (tiled)"}
        found, lines = dead_node_warnings(wf, INFO)
        self.assertEqual(len(found), 1)
        self.assertEqual(len(lines), 1)
        self.assertIn("Node 9", lines[0])
        self.assertIn("VAEDecodeTiled", lines[0])
        self.assertIn("VAE Decode (tiled)", lines[0])

    def test_a_clean_graph_produces_no_lines(self):
        wf = {"1": node("KSampler", seed=1), "2": node("SaveImage", images=["1", 0])}
        self.assertEqual(dead_node_warnings(wf, INFO), ([], []))


if __name__ == "__main__":
    unittest.main()
