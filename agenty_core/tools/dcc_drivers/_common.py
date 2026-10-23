# Shared by every app driver: runs inside the DCC app's Python (3.10+). No third-party imports.
import contextlib
import io
import json
import os
import traceback


def exec_build_script(path):
    """Run the whole build script file. Returns (ok, stdout, error, traceback_text)."""
    buf = io.StringIO()
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
        code = compile(src, path, "exec")
        g = {"__name__": "__main__", "__file__": path}
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, g)
        return True, buf.getvalue(), "", ""
    except BaseException as e:  # noqa: BLE001 - SystemExit and friends must not kill the host app
        frames = traceback.extract_tb(e.__traceback__)
        mine = [fr for fr in frames if fr.filename == path]
        where = ""
        if isinstance(e, SyntaxError) and e.lineno:
            where = " (script line %s: %s)" % (e.lineno, (e.text or "").strip())
        elif mine:
            where = " (script line %s: %s)" % (mine[-1].lineno, (mine[-1].line or "").strip())
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        tb = "\n".join(tb.strip().splitlines()[-12:])
        return False, buf.getvalue(), "%s: %s%s" % (type(e).__name__, e, where), tb


def dispatch(job, build, check):
    """Pick the job's action. Any crash becomes an {ok: False, error} answer instead of a dead process."""
    try:
        if job.get("action") == "build":
            return build(job)
        if job.get("action") == "check":
            return check(job)
        return {"ok": False, "error": "unknown action %r" % job.get("action")}
    except BaseException as e:  # noqa: BLE001
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e),
                "traceback": "\n".join(traceback.format_exc().strip().splitlines()[-12:])}


def parse_resolution(text):
    """'1280x720' -> (1280, 720); '' -> None."""
    if not text:
        return None
    a, b = str(text).lower().replace(" ", "").split("x")
    return int(a), int(b)
