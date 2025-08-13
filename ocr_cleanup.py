import re
from collections import Counter
from typing import List


def _ratio_upper(s: str) -> float:
	alpha = [c for c in s if c.isalpha()]
	if not alpha:
		return 0.0
	upper = sum(1 for c in alpha if c.isupper())
	return upper / len(alpha)


def _remove_common_headers_footers(lines: List[str]) -> List[str]:
	# Remove short, highly repeated lines that are mostly uppercase (stamps/headers)
	counts = Counter([ln.strip() for ln in lines if 3 <= len(ln.strip()) <= 60])
	ban = {t for t, c in counts.items() if c >= 3 and _ratio_upper(t) >= 0.7}
	if not ban:
		return lines
	return [ln for ln in lines if ln.strip() not in ban]


def _dehyphenate(text: str) -> str:
	# word-\nnext -> wordnext
	text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
	return text


def _merge_wrapped_lines(text: str) -> str:
	# Merge lines where a line does not end with sentence punctuation
	lines = text.split("\n")
	merged: List[str] = []
	buf = []
	for ln in lines:
		strip = ln.strip()
		if not strip:
			# flush paragraph
			if buf:
				merged.append(" ".join(buf).strip())
				buf = []
			merged.append("")
			continue
		buf.append(strip)
		if strip.endswith(('.', '?', '!', ':', ';')):
			merged.append(" ".join(buf).strip())
			buf = []
	# flush
	if buf:
		merged.append(" ".join(buf).strip())
	return "\n".join(merged)


def _remove_noise_lines(text: str) -> str:
	lines = text.split("\n")
	# Drop very short all-caps noise lines
	cleaned = []
	for ln in lines:
		strip = ln.strip()
		if not strip:
			cleaned.append("")
			continue
		if len(strip) <= 4 and _ratio_upper(strip) >= 0.8:
			continue
		# Known stamp words
		if re.fullmatch(r"(LEASE\s*DEED|STAMP|DRAFT|TAMILNADU|ANNEXURE|SCHEDULE)\b.*", strip, flags=re.IGNORECASE):
			continue
		cleaned.append(ln)
	# Remove common headers/footers
	cleaned = _remove_common_headers_footers(cleaned)
	return "\n".join(cleaned)


def clean_ocr_text(text: str) -> str:
	if not text:
		return ""
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	text = _dehyphenate(text)
	text = _remove_noise_lines(text)
	text = _merge_wrapped_lines(text)
	# Collapse 3+ newlines to 2
	text = re.sub(r"\n{3,}", "\n\n", text)
	return text.strip()
