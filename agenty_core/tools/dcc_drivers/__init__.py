"""Source text that runs INSIDE Blender / Maya / Houdini (see ``agenty_core.tools.dcc``).

These files are never imported by agentY: they import ``bpy`` / ``maya`` / ``hou`` and only make sense in
the app's own Python. ``dcc.py`` reads them as text, joins ``_common.py`` + the app's driver + a small
wrapper, and either hands that to a fresh ``blender -b`` / ``mayapy`` / ``hython`` process (batch) or sends it
to the app the user has open (live).
"""
