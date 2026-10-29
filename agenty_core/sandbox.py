"""What ``run_script`` is allowed to run, and where it is allowed to point.

``run_script`` used to be ``subprocess.run(command, shell=True)`` with no checks
at all. That is a general-purpose remote shell attached to a language model, and
it is reached by anything that can put a sentence in front of that model — a web
page talking to the host, a prompt injected through a downloaded file, or simply
the model deciding, wrongly, that deleting something would help.

**What this can and cannot promise.** A path filter over a *shell string* is
theatre: you cannot decide what ``foo | bash -c "$(echo …)"`` will touch without
writing a shell. So the string stops being a shell string. ``shlex`` splits it,
``shell=False`` runs it, and the whole class of quoting, chaining and substitution
attacks is gone — not mitigated, absent, because there is no shell left to attack.

What remains is one program with literal arguments, and that is small enough to
reason about:

* the program must be one we named (:data:`DEFAULT_EXECUTABLES`). ``ffmpeg`` and
  ``grep`` are here because real skills use them; ``bash``, ``curl`` and
  ``osascript`` are not, because a general interpreter or a network client makes
  the rest of the list pointless.
* every argument that names a path must resolve inside a root we allow. This one
  is a genuine limit rather than a guarantee — a program handed a relative path
  can still write relative to its working directory, which is why the working
  directory is set explicitly instead of inherited.

The remaining hole is deliberate and documented: an allowed program can be asked
to do an unwanted thing inside an allowed root. ``src.utils.tool_permissions`` is
what covers that, by asking a person.
"""

from __future__ import annotations

import os
import shlex
import sys
import tempfile
from pathlib import Path

# Programs run_script may launch, by basename. Windows extensions are stripped
# before the check, so "python.exe" matches "python".
#
# Chosen from what the skills and prompts actually ask for: python for skill
# scripts, ffmpeg/ffprobe for the video work, the listing and searching tools the
# coder prompt uses to read a cloned repository, and git because
# custom-node-from-github clones one.
#
# What is missing is the point. There is no shell (bash, sh, zsh, cmd,
# powershell), no downloader (curl, wget), no scripting host (osascript, node,
# perl, ruby), and no package manager — each of those would make every other
# entry on this list decorative.
DEFAULT_EXECUTABLES = (
    "python", "python3", "py",
    "ffmpeg", "ffprobe",
    "ls", "dir", "cat", "type", "head", "tail", "wc",
    "grep", "findstr", "find", "where", "which",
    "git",
)

# Tokens that mean something to a shell and nothing to us. They cannot DO
# anything now that shell=False, but a command containing one was written
# expecting a shell, so it would not have done what its author meant either.
# Saying so is more useful than running a mangled version of it.
_SHELL_OPERATORS = frozenset({";", "|", "||", "&", "&&", ">", ">>", "<", "<<", "2>", "&>"})

_extra_executables: tuple[str, ...] = ()
_extra_roots: tuple[str, ...] = ()


def configure(*, executables=(), roots=()) -> None:
    """Widen the policy from the host application's settings.

    agentY calls this at startup with ComfyUI's input/output/user directories,
    which it knows and this package deliberately does not: resolving them from
    here would mean an HTTP call to ComfyUI on every command.
    """
    global _extra_executables, _extra_roots
    _extra_executables = tuple(str(e).strip() for e in (executables or ()) if str(e).strip())
    _extra_roots = tuple(str(r) for r in (roots or ()) if str(r))


def allowed_executables() -> frozenset[str]:
    return frozenset(DEFAULT_EXECUTABLES) | frozenset(
        norm_exe(e) for e in _extra_executables)


def allowed_roots() -> list[Path]:
    """Directories a path argument may point inside.

    The checkout, the system temp directory (where every intermediate render
    goes), and whatever the host application added.
    """
    roots: list[Path] = []
    try:
        from agenty_core.paths import project_root
        roots.append(Path(project_root()))
    except Exception:  # noqa: BLE001
        pass
    roots.append(Path(tempfile.gettempdir()))
    # /tmp as well as gettempdir(). On a Mac those are different places —
    # gettempdir() returns the per-user /var/folders/… directory — and a model
    # writing an intermediate render says /tmp, because everything does. It is
    # world-writable already, so allowing it grants nothing that was not granted.
    if os.name != "nt":
        roots.append(Path("/tmp"))
    roots.extend(Path(r) for r in _extra_roots)

    out: list[Path] = []
    for r in roots:
        try:
            resolved = r.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved not in out:
            out.append(resolved)
    return out


def norm_exe(token: str) -> str:
    """The comparable name of argv[0]: basename, lowercased, no .exe/.bat/.cmd.

    A full path is normal and fine — sys.executable is one — so the check is on
    the name, not on where it lives.
    """
    name = Path(str(token or "")).name.lower()
    for ext in (".exe", ".bat", ".cmd", ".com"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def looks_like_path(token: str) -> bool:
    """Is this argument naming a location on disk?

    Conservative in the direction that matters: an option value that merely looks
    like a path gets checked (harmless — it will be inside a root or the command
    was wrong anyway), while a bare word like ``-vcodec`` or ``libx264`` does not,
    so ordinary ffmpeg invocations are not rejected for being ffmpeg invocations.
    """
    t = str(token or "")
    if not t or t.startswith("-"):
        return False
    # A URL is not a path, and it contains slashes. `git clone https://host/repo`
    # was refused for pointing outside the project, which is both wrong and
    # incomprehensible — the argument names nothing on this disk at all.
    if "://" in t:
        return False
    if t.startswith("~"):
        return True
    if os.path.isabs(t):
        return True
    return "/" in t or "\\" in t


def _inside(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return False
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def split_command(command: str, *, windows: bool | None = None) -> list[str]:
    """Split *command* into the argv a program should receive.

    POSIX rules strip quotes but treat backslashes as escapes, which is wrong on
    Windows, where every path is full of them. Non-POSIX ``shlex`` keeps the
    backslashes but also KEEPS THE QUOTES: ``python -c "print('hi')"`` handed
    Python the code ``"print('hi')"`` quotes included — a string literal, which it
    evaluated, printed nothing for, and exited 0. Every inline ``-c`` the agent ran
    on Windows came back empty with no error, and it concluded, reasonably, that
    the shell was broken. So on Windows the outer pair of matching quotes comes off
    each token, which is what the program would have received from a real command
    line.
    """
    windows = (os.name == "nt") if windows is None else windows
    if not windows:
        return shlex.split(command, posix=True)
    return [_unquote(token) for token in shlex.split(command, posix=False)]


def _unquote(token: str) -> str:
    """*token* without one enclosing pair of matching quotes, if it has one."""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


def check_command(command: str, *, cwd: Path | None = None
                  ) -> tuple[list[str] | None, str]:
    """Vet *command*. Returns ``(argv, "")`` to run it, or ``(None, why)``.

    Pure apart from reading the configured policy, so every refusal below can be
    tested without launching anything.
    """
    raw = str(command or "").strip()
    if not raw:
        return None, "no command given."

    if "\n" in raw or "\r" in raw:
        return None, ("run_script takes a single-line command. Write a multi-line "
                      "script to a file with write_text_file and run that file.")

    try:
        argv = split_command(raw)
    except ValueError as exc:
        return None, f"could not parse the command ({exc}). Check the quoting."
    if not argv:
        return None, "no command given."

    hit = next((t for t in argv if t in _SHELL_OPERATORS), "")
    if hit:
        return None, (
            f"'{hit}' needs a shell, and run_script no longer uses one — it runs a "
            "single program directly, so pipes, redirects and chained commands are "
            "not available. Run one command per call, or put the sequence in a "
            "Python script and run that.")

    exe = norm_exe(argv[0])
    permitted = allowed_executables()
    if exe not in permitted:
        return None, (
            f"'{argv[0]}' is not a program run_script may launch. Allowed: "
            f"{', '.join(sorted(permitted))}. This is deliberate — a shell or a "
            "downloader here would undo the restriction on all the others. Add to "
            "security.shell_allowed_commands if you need another one.")

    roots = allowed_roots()
    base = Path(cwd) if cwd else (roots[0] if roots else Path.cwd())
    from agenty_core.utils.path_guard import is_secret
    for token in argv[1:]:
        # `type .env` / `cat .env` carry no slash, so the root check below never
        # saw them, and .env is inside the project. Not a wall (python -c can
        # spell the name any way it likes), but the plain way is refused.
        if not token.startswith("-") and "://" not in token:
            named = Path(token).expanduser()
            if is_secret(named if named.is_absolute() else base / named):
                return None, (f"'{token}' holds credentials; run_script does not read it.")
        if not looks_like_path(token):
            continue
        candidate = Path(token).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        if not _inside(candidate, roots):
            return None, (
                f"'{token}' points outside the folders run_script may touch "
                f"({', '.join(str(r) for r in roots)}). Work inside the project, "
                "ComfyUI's directories, or a temp file.")

    return argv, ""


def working_directory() -> Path:
    """Where a vetted command runs.

    Set rather than inherited: a relative path in an argument is resolved against
    it, so leaving it to whatever the process happened to start in makes the path
    checks above describe a different directory than the one the program uses.
    That exact mismatch — a relative path resolved against the wrong caller's cwd
    — is what split the project-memory store in two.
    """
    roots = allowed_roots()
    return roots[0] if roots else Path.cwd()


def python_executable() -> str:
    """The interpreter a bare ``python`` should mean: this one.

    A skill script needs the environment agentY was installed into. ``python`` on
    PATH is whatever the shell would have found, which on a Mac with a venv
    active is right by accident and wrong as soon as it is not.
    """
    return sys.executable or "python"
