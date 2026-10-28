"""The prompt hooks: what a host runs before ComfyUI is handed a graph.

``requests.post`` is stubbed; nothing talks to a server.
"""

import unittest
from unittest import mock

from agenty_core.utils import comfyui_client as cc


class _Resp:
    status_code = 200
    text = "{}"

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        pass


class PromptHooks(unittest.TestCase):

    def setUp(self):
        self._saved = list(cc._prompt_hooks)
        cc._prompt_hooks.clear()
        cc.forget_down()
        self.client = cc.ComfyUIClient(base_url="http://127.0.0.1:1", api_key="")
        self.order = []

    def tearDown(self):
        cc._prompt_hooks[:] = self._saved

    def _post(self, *args, **kwargs):
        self.order.append("post")
        return _Resp({"prompt_id": "p1"})

    def test_hook_runs_before_a_prompt_and_its_follow_up_after(self):
        seen = []

        def hook(prompt):
            seen.append(prompt)
            self.order.append("hook")
            return lambda: self.order.append("after")

        cc.add_prompt_hook(hook)
        cc.add_prompt_hook(hook)                     # idempotent
        with mock.patch.object(cc.requests, "post", side_effect=self._post):
            out = self.client.post("/prompt", json_data={"prompt": {"1": {"class_type": "KSampler"}}})
        self.assertEqual(out, {"prompt_id": "p1"})
        self.assertEqual(seen, [{"1": {"class_type": "KSampler"}}])
        self.assertEqual(self.order, ["hook", "post", "after"])

    def test_other_posts_do_not_run_the_hooks(self):
        cc.add_prompt_hook(lambda prompt: self.order.append("hook"))
        with mock.patch.object(cc.requests, "post", side_effect=self._post):
            self.client.post("/free", json_data={"unload_models": True})
        self.assertEqual(self.order, ["post"])

    def test_suspended_block_skips_the_hooks(self):
        cc.add_prompt_hook(lambda prompt: self.order.append("hook"))
        with mock.patch.object(cc.requests, "post", side_effect=self._post):
            with cc.prompt_hooks_suspended():
                self.client.post("/prompt", json_data={"prompt": {}})
            self.client.post("/prompt", json_data={"prompt": {}})
        self.assertEqual(self.order, ["post", "hook", "post"])

    def test_a_failing_hook_never_costs_the_render(self):
        def boom(prompt):
            raise RuntimeError("no")

        cc.add_prompt_hook(boom)
        with mock.patch.object(cc.requests, "post", side_effect=self._post):
            self.assertEqual(self.client.post("/prompt", json_data={"prompt": {}}), {"prompt_id": "p1"})

    def test_follow_up_runs_when_the_post_fails(self):
        cc.add_prompt_hook(lambda prompt: (lambda: self.order.append("after")))
        with mock.patch.object(cc.requests, "post", side_effect=cc.requests.ConnectionError("down")):
            with self.assertRaises(cc.requests.ConnectionError):
                self.client.post("/prompt", json_data={"prompt": {}})
        self.assertEqual(self.order, ["after"])
        cc.forget_down()


if __name__ == "__main__":
    unittest.main()
