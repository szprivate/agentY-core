"""Find the creative apps installed on this machine, and which version is the newest.

One answer for "is Blender / Houdini / Maya / Nuke / After Effects / Premiere / Resolve / Cavalry / Unity
here, where, and which version", used by

* the MCP preset setup recipes (agentY ``scripts/mcp_setup.py``: ``find_app`` step), and
* the ``find_app_installs`` tool, so an agent can ask before it builds (``dcc_build`` wants a Blender).

Where it looks, in this order: an environment variable that names the executable, the ``PATH``, the
Windows uninstall registry (read-only), the usual install folders. Every hit becomes one record
``{app, version, path, root, source}``; the list is sorted newest first. ``pick`` takes the newest one
that meets a minimum version and reports the others, so a machine with Blender 3.5 and 5.2 is not a coin toss.

Nothing here starts an app, writes anything or needs the network. A version is read from the folder name,
the registry, the macOS bundle's ``Contents/Info.plist`` or the Windows exe's file version, and only when
none of those says (a portable copy in a folder called ``blender``) is the app asked for ``--version`` (Blender
only). That last call can fail on a Mac where the app was never opened (Gatekeeper), so it is the fallback, and a
version that stays unknown is "unknown", never "too old": ``pick`` still offers the install.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
from typing import Iterable, Optional

from agenty_core._compat import tool

# app -> what finds it. ``env``: variables that hold the executable; ``path``: names on PATH; ``globs``:
# executables in the usual folders; ``registry``: a regex over the uninstall DisplayName.
_APPS: dict[str, dict] = {
    "blender": {
        "env": ["BLENDER_PATH"], "path": ["blender"],
        "globs": ["C:/Program Files/Blender Foundation/Blender */blender.exe",
                  "C:/Program Files/Blender Foundation/Blender/blender.exe",
                  "/Applications/Blender.app/Contents/MacOS/Blender",
                  "/Applications/Blender */Blender.app/Contents/MacOS/Blender",
                  "/Applications/Blender*.app/Contents/MacOS/Blender",
                  "~/Applications/Blender*.app/Contents/MacOS/Blender",
                  "/usr/bin/blender", "/snap/bin/blender"],
        "registry": r"^Blender( |$)", "name": r"[Bb]lender[ -]?(\d+(?:\.\d+){0,2})"},
    "houdini": {
        "env": ["HYTHON_PATH"], "path": ["houdini", "hython"],
        "globs": ["C:/Program Files/Side Effects Software/Houdini */bin/houdini.exe",
                  "/opt/hfs*/bin/houdini",
                  "/Applications/Houdini/Houdini*/Frameworks/Houdini.framework/Versions/Current/Resources/bin/houdini"],
        "registry": r"^Houdini \d", "name": r"[Hh]oudini[ -]?(\d+(?:\.\d+){0,2})|hfs(\d+(?:\.\d+){0,2})"},
    "maya": {
        "env": ["MAYA_LOCATION"], "path": ["maya", "mayapy"],
        "globs": ["C:/Program Files/Autodesk/Maya20*/bin/maya.exe",
                  "/Applications/Autodesk/maya20*/Maya.app/Contents/bin/maya", "/usr/autodesk/maya20*/bin/maya"],
        "registry": r"^Autodesk Maya \d{4}$", "name": r"[Mm]aya[ -]?(\d{4}(?:\.\d+)?)"},
    "nuke": {
        "env": ["NUKE_EXE"], "path": ["nuke"],
        "globs": ["C:/Program Files/Nuke*/Nuke[0-9]*.exe", "/Applications/Nuke*/Nuke*.app/Contents/MacOS/Nuke*",
                  "/usr/local/Nuke*/Nuke[0-9]*"],
        "registry": r"^Nuke \d", "name": r"Nuke[ -]?(\d+(?:\.\d+)?)(?:v(\d+))?"},
    "after-effects": {
        "env": ["AE_MCP_EXE"], "path": [],
        "globs": ["C:/Program Files/Adobe/Adobe After Effects */Support Files/AfterFX.exe",
                  "/Applications/Adobe After Effects */Adobe After Effects */Contents/MacOS/After Effects"],
        "registry": r"^Adobe After Effects", "name": r"After Effects[ -]?(\d{4})"},
    "premiere-pro": {
        "env": [], "path": [],
        "globs": ["C:/Program Files/Adobe/Adobe Premiere Pro */Adobe Premiere Pro.exe",
                  "/Applications/Adobe Premiere Pro */Adobe Premiere Pro */Contents/MacOS/Adobe Premiere Pro"],
        "registry": r"^Adobe Premiere Pro", "name": r"Premiere Pro[ -]?(\d{4})"},
    "davinci-resolve": {
        "env": [], "path": [],
        "globs": ["C:/Program Files/Blackmagic Design/DaVinci Resolve/Resolve.exe",
                  "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/MacOS/Resolve",
                  "/opt/resolve/bin/resolve"],
        "registry": r"^DaVinci Resolve( Studio)?$", "name": r"(\d+(?:\.\d+){0,2})"},
    "cavalry": {
        "env": ["CAVALRY_PATH"], "path": [],
        "globs": ["C:/Program Files/Scene Group/Cavalry*/Cavalry.exe", "C:/Program Files/Cavalry*/Cavalry.exe",
                  "C:/Program Files/Cavalry/Cavalry.exe", "/Applications/Cavalry.app/Contents/MacOS/Cavalry"],
        "registry": r"^Cavalry", "name": r"Cavalry[ -]?(\d+(?:\.\d+){0,2})"},
    "unity": {
        "env": ["UNITY_PATH"], "path": [],
        "globs": ["C:/Program Files/Unity/Hub/Editor/*/Editor/Unity.exe", "C:/Program Files/Unity*/Editor/Unity.exe",
                  "/Applications/Unity/Hub/Editor/*/Unity.app/Contents/MacOS/Unity"],
        "registry": r"^Unity \d", "name": r"(\d+\.\d+(?:\.\d+)?|\d{4}\.\d+(?:\.\d+)?)"},
}
APPS = tuple(_APPS)
_ALIASES = {"ae": "after-effects", "aftereffects": "after-effects", "after_effects": "after-effects",
            "premiere": "premiere-pro", "premiere_pro": "premiere-pro", "resolve": "davinci-resolve",
            "davinci": "davinci-resolve", "davinci_resolve": "davinci-resolve", "hython": "houdini",
            "mayapy": "maya"}


def canonical(app: str) -> str:
    a = str(app or "").strip().lower().replace(" ", "-")
    return _ALIASES.get(a, a)


def version_tuple(text: str) -> tuple:
    """``'5.2.2'`` -> ``(5, 2, 2)``; ``'6000.0.23f1'`` -> ``(6000, 0, 23)``; ``''`` -> ``()``."""
    m = re.search(r"\d+(?:\.\d+)*", str(text or ""))
    return tuple(int(n) for n in m.group(0).split(".")) if m else ()


def _version_from_name(app: str, text: str) -> str:
    m = re.search(_APPS[app]["name"], text.replace("\\", "/"))
    if not m:
        return ""
    parts = [g for g in m.groups() if g]
    if app == "nuke" and len(parts) == 2:        # Nuke15.0v4 -> 15.0.4
        return f"{parts[0]}.{parts[1]}"
    return parts[0] if parts else ""


def _app_bundle(path: str) -> str:
    """The ``X.app`` folder a macOS executable lives in, or ''."""
    cur = os.path.abspath(path)
    while True:
        parent = os.path.dirname(cur)
        if cur.endswith(".app") and os.path.isdir(cur):
            return cur
        if parent == cur:
            return ""
        cur = parent


def plist_version(plist_path: str) -> str:
    """``CFBundleShortVersionString`` (else ``CFBundleVersion``) of an Info.plist, '' when unreadable.

    Reads the file, never runs the app: this works on a Blender that was downloaded and never opened.
    """
    import plistlib
    try:
        with open(plist_path, "rb") as fh:
            data = plistlib.load(fh)
    except (OSError, ValueError, plistlib.InvalidFileException):
        return ""
    for key in ("CFBundleShortVersionString", "CFBundleVersion"):
        v = str(data.get(key) or "").strip()
        if re.search(r"\d", v):
            return v
    return ""


def _exe_file_version(path: str) -> str:
    """The version resource of a Windows ``.exe`` ('5.1.0.0' -> '5.1.0'), '' elsewhere or when absent."""
    if not sys.platform.startswith("win") or not path.lower().endswith(".exe"):
        return ""
    try:
        import ctypes
        from ctypes import wintypes
        ver = ctypes.WinDLL("version.dll")
        size = ver.GetFileVersionInfoSizeW(path, None)
        if not size:
            return ""
        buf = ctypes.create_string_buffer(size)
        if not ver.GetFileVersionInfoW(path, 0, size, buf):
            return ""
        ptr, length = ctypes.c_void_p(), wintypes.UINT()
        if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)) or length.value < 52:
            return ""
        info = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32 * 13)).contents
        ms, ls = info[4], info[5]
        parts = [ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF]
        while len(parts) > 2 and parts[-1] == 0:
            parts.pop()
        return ".".join(str(p) for p in parts)
    except (OSError, AttributeError, ValueError):
        return ""


def version_without_running(app: str, path: str) -> str:
    """The version of the program at *path* from its own files: bundle plist (macOS), file version (Windows)."""
    bundle = _app_bundle(path)
    if bundle:
        v = plist_version(os.path.join(bundle, "Contents", "Info.plist"))
        if v:
            return v
    return _exe_file_version(path)


def _blender_asks(path: str) -> str:
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
                             stdin=subprocess.DEVNULL).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    m = re.search(r"Blender (\d+(?:\.\d+)*)", out)
    return m.group(1) if m else ""


def _registry_installs(app: str) -> list[dict]:
    """Uninstall-registry entries whose DisplayName matches the app (Windows, read-only)."""
    if not sys.platform.startswith("win"):
        return []
    try:
        import winreg
    except ImportError:   # pragma: no cover
        return []
    pattern = re.compile(_APPS[app]["registry"])
    out = []
    keys = [(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")]
    for hive, sub in keys:
        try:
            root = winreg.OpenKey(hive, sub)
        except OSError:
            continue
        with root:
            i = 0
            while True:
                try:
                    name = winreg.EnumKey(root, i)
                except OSError:
                    break
                i += 1
                try:
                    with winreg.OpenKey(root, name) as k:
                        def val(n):
                            try:
                                return str(winreg.QueryValueEx(k, n)[0])
                            except OSError:
                                return ""
                        display = val("DisplayName")
                        if display and pattern.search(display):
                            out.append({"display": display, "version": val("DisplayVersion"),
                                        "location": val("InstallLocation")})
                except OSError:
                    continue
    return out


def _record(app: str, path: str, source: str, version: str = "", root: str = "") -> dict:
    path = os.path.normpath(path)
    return {"app": app, "version": version, "path": path,
            "root": os.path.normpath(root) if root else os.path.dirname(path), "source": source}


def find_installs(app: str, *, env: Optional[dict] = None, extra_globs: Iterable[str] = (),
                  system: bool = True) -> list[dict]:
    """Every install of *app* this machine shows, newest first.

    ``env``: the environment to read (default ``os.environ``). ``extra_globs``: more executable patterns
    (a portable copy, a test tree). ``system=False`` reads ONLY the environment variable and
    ``extra_globs``: no PATH, no registry, no standard folders, which is what tests want.
    """
    app = canonical(app)
    if app not in _APPS:
        raise KeyError(f"unknown app '{app}' (known: {', '.join(APPS)})")
    spec = _APPS[app]
    env = os.environ if env is None else env
    found: dict[str, dict] = {}

    def add(rec: dict) -> None:
        key = os.path.normcase(rec["path"])
        if key not in found:
            found[key] = rec

    def versioned(path: str, registry_version: str = "") -> str:
        v = _version_from_name(app, path) or registry_version
        if not v and os.path.isfile(path):
            v = version_without_running(app, path)       # reads files; the app is not started
        if not v and app == "blender" and os.path.isfile(path):
            v = _blender_asks(path)                         # last resort; may fail on a never-opened Mac app
        return v

    for var in spec["env"]:
        raw = str(env.get(var, "") or "").strip().strip('"')
        if raw:
            if os.path.isdir(raw):    # MAYA_LOCATION is a folder; the globs find the program inside it
                for pat in spec["globs"]:
                    tail = pat.rsplit("/bin/", 1)[-1] if "/bin/" in pat else ""
                    cand = os.path.join(raw, "bin", tail) if tail else ""
                    if cand and os.path.isfile(cand):
                        add(_record(app, cand, f"env:{var}", versioned(cand)))
            elif os.path.isfile(raw):
                add(_record(app, raw, f"env:{var}", versioned(raw)))
    for pat in extra_globs:
        for p in glob.glob(pat):
            if os.path.isfile(p):
                add(_record(app, p, "extra", versioned(p)))
    if system:
        for name in spec["path"]:
            hit = shutil.which(name)
            if hit:
                add(_record(app, hit, "path", versioned(hit)))
        for pat in spec["globs"]:
            for p in glob.glob(os.path.expanduser(pat)):
                if os.path.isfile(p):
                    add(_record(app, p, "folder", versioned(p)))
        for reg in _registry_installs(app):
            loc = reg["location"]
            if not loc:
                continue
            exes = [p for pat in spec["globs"] for p in glob.glob(pat)
                    if os.path.isfile(p) and os.path.normcase(p).startswith(os.path.normcase(os.path.normpath(loc)))]
            if exes:
                add(_record(app, exes[0], "registry", reg["version"] or versioned(exes[0]), loc))
    # A registry hit and a folder hit for the same exe are one install; a version a name could not give
    # is filled from the registry record (matched by folder).
    if system:
        regs = _registry_installs(app)
        for reg in regs:
            loc = os.path.normcase(os.path.normpath(reg["location"])) if reg["location"] else ""
            for rec in found.values():
                if not rec["version"] and loc and os.path.normcase(rec["path"]).startswith(loc):
                    rec["version"] = reg["version"]
        # One program on disk and registry records without a folder (Resolve): the newest record is it.
        blank = [r for r in found.values() if not r["version"]]
        newest = max((r["version"] for r in regs if r["version"]), key=version_tuple, default="")
        if len(found) == 1 and blank and newest:
            blank[0]["version"] = newest
    return sorted(found.values(), key=lambda r: version_tuple(r["version"]), reverse=True)


def pick(installs: list[dict], min_version: str = "") -> tuple[Optional[dict], list[dict], str]:
    """``(chosen, others, why)``. The newest install that meets *min_version*.

    An install whose version could not be read is *unknown*, not too old: it is chosen when no install
    is known to meet the minimum ("version unknown, will try"). ``chosen`` is None only when nothing is
    installed or every version that could be read is below the minimum; ``why`` says what was found.
    """
    need = version_tuple(min_version)
    ok = [r for r in installs if not need or (r["version"] and version_tuple(r["version"]) >= need)]
    if not ok:
        unknown = [r for r in installs if not r["version"]]
        if unknown:
            chosen = unknown[0]
            why = f"version unknown (not readable without starting it), will try; {min_version} or newer is needed"
            return chosen, [r for r in installs if r is not chosen], why
        if installs:
            have = ", ".join(f"{r['version']} ({r['path']})" for r in installs)
            return None, installs, f"found {have}, but version {min_version} or newer is needed"
        return None, [], "not installed"
    chosen = ok[0]
    others = [r for r in installs if r is not chosen]
    why = f"newest of {len(installs)} install(s)" if others else "the only install"
    return chosen, others, why


@tool
def find_app_installs(apps: list[str] | None = None, min_version: str = "") -> dict:
    """Find which creative apps are installed here and which version is the newest.

    Looks in the usual places (environment variables, PATH, the Windows uninstall registry, the
    standard install folders) and never starts an app. Use it before building in or setting up an
    app (Blender, Houdini, Maya, Nuke, After Effects, Premiere Pro, DaVinci Resolve, Cavalry, Unity).

    Args:
        apps: Which apps to look for, e.g. ["blender", "houdini"]. Default: all nine.
        min_version: Optional minimum version, e.g. "5.1" for Blender. The answer names the newest
            install that meets it.

    Returns:
        {app: {"installed": bool, "newest": {version, path, root, source} | None, "chosen": ...,
        "others": [...], "why": str}} for every app asked about.
    """
    out = {}
    for name in (apps or list(APPS)):
        app = canonical(name)
        if app not in _APPS:
            out[str(name)] = {"installed": False, "error": f"unknown app (known: {', '.join(APPS)})"}
            continue
        installs = find_installs(app)
        chosen, others, why = pick(installs, min_version)
        out[app] = {"installed": bool(installs), "newest": installs[0] if installs else None,
                    "chosen": chosen, "others": others, "why": why,
                    "all": installs}
    return out
