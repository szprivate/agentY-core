"""install_custom_node clones only what it should, where it should, and installs
the pack's requirements into the Python ComfyUI runs on.

The folder name used to come unchecked from the URL's last segment and any host
(and ssh / http) was cloned from. Nothing here runs git or pip: subprocess is
mocked, and every test checks the command that would have run.
"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import comfyui as C


class Validation(unittest.TestCase):

    def test_accepts_a_repository_url(self):
        for src in ("https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer",
                    "https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer.git",
                    "https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer/",
                    "  https://GitHub.com/Blueforcer/ComfyUI-DLSS5-Enhancer.git "):
            self.assertEqual(C.validate_node_pack_source(src),
                             ("https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer.git",
                              "ComfyUI-DLSS5-Enhancer"), src)
        self.assertEqual(C.validate_node_pack_source("https://gitlab.com/group/sub/my_pack"),
                         ("https://gitlab.com/group/sub/my_pack.git", "my_pack"))

    def test_refuses_other_schemes_and_hosts(self):
        for src in ("git@github.com:owner/pack.git", "http://github.com/owner/pack",
                    "ssh://git@github.com/owner/pack", "file:///C:/pack", "ext::sh -c touch% /tmp/x",
                    "https://evil.example.com/owner/pack", "https://github.com.evil.io/owner/pack",
                    "https://user:token@github.com/owner/pack", "https://github.com:8443/owner/pack",
                    "https://github.com/owner/pack?ref=x", "https://github.com/owner/pack#main",
                    "", "github.com/owner/pack"):
            with self.assertRaises(ValueError, msg=src):
                C.validate_node_pack_source(src)

    def test_refuses_unsafe_folder_names(self):
        for src in ("https://github.com/owner/..", "https://github.com/owner/.hidden",
                    "https://github.com/owner/-rf", "https://github.com/owner/a%2F..",
                    "https://github.com/owner/a..b", "https://github.com/owner/CON",
                    "https://github.com/owner/nul.txt", "https://github.com/owner/a\\..\\b",
                    "https://github.com/owner/pack/tree/main", "https://github.com/owner",
                    "https://github.com/owner/" + "a" * 101):
            with self.assertRaises(ValueError, msg=src):
                C.validate_node_pack_source(src)

    def test_more_hosts_by_environment(self):
        with mock.patch.dict("os.environ", {"AGENTY_CUSTOM_NODE_HOSTS": "git.studio.local"}):
            self.assertEqual(C.validate_node_pack_source("https://git.studio.local/td/pack")[1], "pack")


class Install(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cn = self.root / "custom_nodes"
        self.cn.mkdir()
        self.calls = []

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, cmd, **kw):
        self.calls.append((list(cmd), kw))
        if cmd[:1] == ["git"]:
            dest = Path(cmd[-1])
            dest.mkdir()
            (dest / "requirements.txt").write_text("numpy\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def _install(self, source, py=None, **kw):
        with mock.patch.object(C, "_custom_nodes_dir", return_value=self.cn), \
                mock.patch.object(C, "_resolve_comfyui_python", return_value=py), \
                mock.patch("shutil.which", side_effect=lambda n: f"C:/bin/{n}.exe"), \
                mock.patch("subprocess.run", side_effect=self._run):
            return json.loads(C.install_custom_node(source, **kw))

    def test_refused_url_runs_nothing(self):
        out = self._install("https://evil.example.com/owner/pack")
        self.assertEqual(out["status"], "error")
        self.assertEqual(self.calls, [])
        self.assertEqual(list(self.cn.iterdir()), [])

    def test_clones_the_canonical_url_into_custom_nodes(self):
        py = self.root / "python_embeded" / "python.exe"
        out = self._install("https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer", py=py)
        self.assertEqual(out["status"], "ok", out)
        git = self.calls[0][0]
        self.assertEqual(git[0], "git")
        self.assertIn("protocol.allow=never", git)
        i = git.index("--")
        self.assertEqual(git[i + 1:], ["https://github.com/Blueforcer/ComfyUI-DLSS5-Enhancer.git",
                                       str((self.cn / "ComfyUI-DLSS5-Enhancer").resolve())])
        self.assertEqual(self.calls[0][1]["env"]["GIT_TERMINAL_PROMPT"], "0")
        self.assertTrue(out["needs_restart"])
        self.assertIn("Restart ComfyUI", out["message"])

    def test_requirements_go_into_comfyuis_embedded_python(self):
        py = self.root / "python_embeded" / "python.exe"
        out = self._install("https://github.com/owner/pack", py=py)
        self.assertEqual(out["pip"], "installed", out)
        pip = self.calls[-1][0]
        self.assertEqual(pip[:4], [str(py), "-s", "-m", "pip"])
        self.assertEqual(pip[-2:], ["-r", str((self.cn / "pack").resolve() / "requirements.txt")])
        self.assertEqual(out["python"], str(py))

    def test_venv_without_pip_uses_uv(self):
        py = self.root / ".venv" / "Scripts" / "python.exe"

        def run(cmd, **kw):
            if cmd[1:4] == ["-m", "pip", "--version"]:
                self.calls.append((list(cmd), kw))
                return subprocess.CompletedProcess(cmd, 1, "", "No module named pip")
            return self._run(cmd, **kw)

        with mock.patch.object(C, "_custom_nodes_dir", return_value=self.cn), \
                mock.patch.object(C, "_resolve_comfyui_python", return_value=py), \
                mock.patch("shutil.which", side_effect=lambda n: f"C:/bin/{n}.exe"), \
                mock.patch("subprocess.run", side_effect=run):
            out = json.loads(C.install_custom_node("https://github.com/owner/pack"))
        self.assertEqual(out["pip"], "installed", out)
        self.assertEqual(self.calls[-1][0][:5], ["C:/bin/uv.exe", "pip", "install", "--python", str(py)])

    def test_no_python_found_says_so_and_still_asks_for_restart(self):
        out = self._install("https://github.com/owner/pack", py=None)
        self.assertEqual(out["pip"], "unresolved-python")
        self.assertTrue(out["needs_restart"])
        self.assertIn("Restart ComfyUI", out["message"])


class ComfyPython(unittest.TestCase):

    def test_skips_a_venv_that_is_not_the_one_comfyui_runs_on(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "ComfyUI"
            scripts = "Scripts" if C.os.name == "nt" else "bin"
            exe = "python.exe" if C.os.name == "nt" else "python"
            wrong = root / "venv" / scripts / exe
            right = root.parent / ".venv" / scripts / exe
            for p in (wrong, right):
                p.parent.mkdir(parents=True)
                p.write_text("")
            versions = {str(wrong): "3.10", str(right): "3.12"}
            with mock.patch.object(C, "_comfyui_root", return_value=root), \
                    mock.patch.object(C, "_comfyui_python_version", return_value="3.12"), \
                    mock.patch.object(C, "_python_version_of", side_effect=lambda p: versions[str(p)]):
                self.assertEqual(C._resolve_comfyui_python(), right)


if __name__ == "__main__":
    unittest.main()
