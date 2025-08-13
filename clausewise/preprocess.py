import re
from typing import List

import nltk


def _ensure_nltk():
	try:
		nltk.data.find("tokenizers/punkt")
	except LookupError:
		nltk.download("punkt", quiet=True)
	# Newer NLTK requires punkt_tab; try to get it if missing
	try:
		nltk.data.find("tokenizers/punkt_tab/english/")
	except LookupError:
		try:
			nltk.download("punkt_tab", quiet=True)
		except Exception:
			pass
	try:
		nltk.data.find("corpora/stopwords")
	except LookupError:
		nltk.download("stopwords", quiet=True)


def clean_text(text: str) -> str:
	"""Basic cleanup: normalize whitespace, remove stray numbering artifacts, normalize dashes.
	This is intentionally conservative to avoid altering legal meaning.
	"""
	_ensure_nltk()
	if not text:
		return ""
	# Normalize line endings
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	# Replace multiple spaces/tabs
	text = re.sub(r"[\t\u00A0]+", " ", text)
	# Collapse >2 blank lines
	text = re.sub(r"\n{3,}", "\n\n", text)
	# Normalize bullets like •, ·
	text = re.sub(r"^[\s\-•·◦]+", "", text, flags=re.MULTILINE)
	# Fix spaced numbering like 1 . 1 -> 1.1
	text = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", text)
	# Normalize dashes
	text = text.replace("–", "-").replace("—", "-")
	# Trim
	return text.strip()
