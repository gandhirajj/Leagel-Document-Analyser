import re
from typing import List, Dict


CLAUSE_HEADING_KEYWORDS = [
	"definitions",
	"confidentiality",
	"payment",
	"fees",
	"term",
	"termination",
	"liability",
	"limitation of liability",
	"governing law",
	"dispute",
	"intellectual property",
	"ip",
	"warranty",
	"indemnity",
	"assignment",
	"notices",
]


_HEADING_RE = re.compile(
	r"(?im)^(?:\s*(?:(?:section\s+)?\d+(?:\.\d+)*|\([a-zA-Z]\)|[A-Z][A-Z\s]{3,}|" +
	"|".join(re.escape(k) for k in CLAUSE_HEADING_KEYWORDS) +
	"))(?::|\.)?\s*$"
)


def _looks_like_heading(line: str) -> bool:
	return bool(_HEADING_RE.match(line.strip()))


def segment_into_clauses(text: str) -> List[Dict]:
	"""Segment text into clauses using heading heuristics; fallback to paragraphs.
	Returns a list of dicts: {id, title, text}
	"""
	if not text:
		return []

	lines = text.split("\n")
	segments: List[Dict] = []
	title: str = ""
	buffer: List[str] = []

	def flush():
		if buffer:
			seg_text = "\n".join(buffer).strip()
			if seg_text:
				segments.append({
					"id": len(segments) + 1,
					"title": title.strip() if title else None,
					"text": seg_text,
				})

	for line in lines:
		if _looks_like_heading(line):
			# start new segment
			flush()
			title = line.strip()
			buffer = []
		else:
			buffer.append(line)
	# flush last
	flush()

	if not segments:
		# paragraph fallback
		paras = [p.strip() for p in text.split("\n\n") if p.strip()]
		for i, para in enumerate(paras, start=1):
			segments.append({"id": i, "title": None, "text": para})

	return segments
