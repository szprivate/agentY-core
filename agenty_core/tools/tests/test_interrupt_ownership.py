"""Stop in one conversation leaves another conversation's render alone.

Several agentY conversations share one ComfyUI. interrupt_execution and
queue(clear_running) used to stop whatever was rendering; now a job the ledger
says another conversation queued is left running unless asked for explicitly.
"""

import json
import unittest
from unittest import mock

from agenty_core import queue_ledger
from agenty_core.tools import comfyui
from agenty_core.utils import turn_scope


class _Client:
    def __init__(self, running_pid):
        self.running = running_pid
        self.posts = []

    def get(self, path, params=None):
        assert path == "/queue"
        return {"queue_running": [[1, self.running, {}, {}, []]] if self.running else [],
                "queue_pending": []}

    def post(self, path, json_data=None):
        self.posts.append((path, json_data))
        return {}


def _call(tool, **kw):
    fn = getattr(tool, "_tool_func", None) or getattr(tool, "__wrapped__", None) or tool
    return json.loads(fn(**kw) or "{}")


class InterruptOwnership(unittest.TestCase):

    def setUp(self):
        queue_ledger.clear()
        self.addCleanup(queue_ledger.clear)
        token = turn_scope.enter(turn_scope.Scope("r1", "chat-b"))
        self.addCleanup(turn_scope.leave, token)

    def run_with(self, client, tool, **kw):
        with mock.patch.object(comfyui, "get_client", return_value=client):
            return _call(tool, **kw)

    def test_another_conversations_job_is_left_running(self):
        queue_ledger.remember("p-a", owner="chat-a")
        client = _Client("p-a")
        out = self.run_with(client, comfyui.interrupt_execution)
        self.assertEqual(out["status"], "not_interrupted")
        self.assertEqual(client.posts, [])

    def test_unless_asked_for_explicitly(self):
        queue_ledger.remember("p-a", owner="chat-a")
        client = _Client("p-a")
        self.run_with(client, comfyui.interrupt_execution, include_other_conversations=True)
        self.assertEqual(client.posts, [("/interrupt", {})])

    def test_its_own_job_or_the_users_is_stopped(self):
        queue_ledger.remember("p-b", owner="chat-b")
        for pid in ("p-b", "p-user"):
            client = _Client(pid)
            self.run_with(client, comfyui.interrupt_execution)
            self.assertEqual(client.posts, [("/interrupt", {})], pid)

    def test_clear_running_follows_the_same_rule(self):
        queue_ledger.remember("p-a", owner="chat-a")
        client = _Client("p-a")
        out = self.run_with(client, comfyui.queue, action="clear_running")
        self.assertEqual(out["status"], "not_interrupted")
        self.assertEqual(client.posts, [])


if __name__ == "__main__":
    unittest.main()
