"""get_workflow_template(request=...) says whether the template really fits the request."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import comfyui as C

REQUEST = ("animate each starting frame with Seedance 2.5 first-frame, 480p, no audio, "
           "cut into one film")


class FitInTemplateSummary(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        for target, val in (("_workflows_dir", Path(self.d.name)), ("_get_object_info", {})):
            p = mock.patch.object(C, target, return_value=val)
            p.start()
            self.addCleanup(p.stop)
        C.clear_tool_caches()

    def get(self, name, request=""):
        return json.loads(C.get_workflow_template(name, request=request))

    def test_the_storyboard_template_is_flagged_for_the_regression_request(self):
        out = self.get("template_seedance2_storyboard_to_video", REQUEST)
        if out.get("error"):
            self.skipTest("template not in this corpus")
        self.assertEqual(out["fit"]["verdict"], "reject")
        self.assertIn("storyboard", " ".join(out["fit"]["reasons"]))

    def test_no_request_no_fit_key(self):
        out = self.get("template_seedance2_storyboard_to_video")
        if out.get("error"):
            self.skipTest("template not in this corpus")
        self.assertNotIn("fit", out)

    def test_the_right_template_is_not_a_reject(self):
        out = self.get("api_seedance2_5_flf2v", REQUEST)
        if out.get("error"):
            self.skipTest("template not in this corpus")
        self.assertNotEqual(out["fit"]["verdict"], "reject")


if __name__ == "__main__":
    unittest.main()
