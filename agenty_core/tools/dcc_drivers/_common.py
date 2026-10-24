# Shared by every app driver: runs inside the DCC app's Python (3.10+). No third-party imports.
import contextlib
import io
import json
import os
import traceback


def exec_build_script(path, stage="", params=None):
    """Run the whole build script file. Returns (ok, stdout, error, traceback_text).
    ``stage`` (blockout / detail / look) is handed to the script as the global ``DCC_STAGE`` and the environment
    variable ``DCC_STAGE``, so a script can build only what the stage allows.
    ``params`` (a JSON-able dict or None) is handed over the same way: the global ``DCC_PARAMS`` (a dict, {} when none),
    the env var ``DCC_PARAMS`` (JSON text) and ``DCC_PARAMS_FILE`` (a temp .json outside the script folder). The env
    vars are restored and the temp file removed when the script ends, so nothing leaks into a later call."""
    buf = io.StringIO()
    import tempfile
    params = params if isinstance(params, dict) else {}
    ptext = json.dumps(params)
    pfile = ""
    saved_env = {k: os.environ.get(k) for k in ("DCC_PARAMS", "DCC_PARAMS_FILE")}
    try:
        fd, pfile = tempfile.mkstemp(prefix="agenty_dcc_params_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as pf:
            pf.write(ptext)
        os.environ["DCC_PARAMS"] = ptext
        os.environ["DCC_PARAMS_FILE"] = pfile
        return _exec_script(path, stage, params, buf)
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if pfile:
            try:
                os.remove(pfile)
            except OSError:
                pass


def _exec_script(path, stage, params, buf):
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
        code = compile(src, path, "exec")
        g = {"__name__": "__main__", "__file__": path, "DCC_STAGE": stage or "look",
             "DCC_PARAMS": params}
        os.environ["DCC_STAGE"] = stage or "look"
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
