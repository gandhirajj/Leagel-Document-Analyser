import io
from typing import Optional

try:
	import pyttsx3
except Exception:
	pyttsx3 = None


def synthesize_to_wav_bytes(text: str) -> Optional[bytes]:
	if not text:
		return None
	if pyttsx3 is None:
		return None
	try:
		engine = pyttsx3.init()
		buf = io.BytesIO()
		# pyttsx3 saves to file path, so we use a temporary file approach
		import tempfile, os
		with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
			path = tmp.name
		engine.save_to_file(text, path)
		engine.runAndWait()
		with open(path, "rb") as f:
			data = f.read()
		try:
			os.remove(path)
		except OSError:
			pass
		return data
	except Exception:
		return None
