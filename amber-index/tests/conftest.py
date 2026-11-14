import os
import tempfile

os.environ.setdefault("AMBER_LIVE", "0")
os.environ.setdefault("AMBER_LIVE_CACHE", os.path.join(tempfile.mkdtemp(), "live_events.json"))
