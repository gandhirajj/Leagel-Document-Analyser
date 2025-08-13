import re
from typing import List, Dict, Optional

import pandas as pd

try:
	import spacy
	try:
		nlp = spacy.load("en_core_web_lg")
	except Exception:
		try:
			nlp = spacy.load("en_core_web_sm")
		except Exception:
			nlp = None
except Exception:
	nlp = None


_MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
_DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_PARTY_RE = re.compile(r"\b(Company|Provider|Client|Customer|Contractor|Consultant|Employer|Employee|Licensor|Licensee|Landlord|Tenant)\b")


def _regex_extract(text: str) -> List[Dict]:
	entities = []
	for pattern, label in [(_MONEY_RE, "MONEY"), (_DATE_RE, "DATE"), (_EMAIL_RE, "EMAIL"), (_URL_RE, "URL"), (_PARTY_RE, "PARTY")]:
		for m in pattern.finditer(text):
			entities.append({
				"text": m.group(0),
				"type": label,
				"start": m.start(),
				"end": m.end(),
			})
	return entities


def _obligation_phrases(text: str) -> List[Dict]:
	entities: List[Dict] = []
	for m in re.finditer(r"\b(shall|must|agrees? to|obligated to)\b[\s\S]{0,120}", text, flags=re.IGNORECASE):
		span = text[m.start():m.end()]
		entities.append({
			"text": span.strip(),
			"type": "OBLIGATION",
			"start": m.start(),
			"end": m.end(),
		})
	return entities


def _spacy_entities(text: str) -> List[Dict]:
	if not nlp:
		return []
	doc = nlp(text)
	ents = []
	for ent in doc.ents:
		ents.append({
			"text": ent.text,
			"type": ent.label_,
			"start": ent.start_char,
			"end": ent.end_char,
		})
	return ents


def extract_entities(text: str, providers: Optional[object] = None) -> pd.DataFrame:
	"""Return entities as a DataFrame with columns: text, type, start, end."""
	if not text:
		return pd.DataFrame(columns=["text", "type", "start", "end"])

	# Prefer IBM Watson NLU if available
	if providers and getattr(providers, "is_watson_ready", False):
		try:
			entities = providers.watson_nlu_entities(text) or []
			if entities:
				return pd.DataFrame(entities)
		except Exception:
			pass

	# Fallback: spaCy + regex
	ents = []
	ents.extend(_spacy_entities(text))
	ents.extend(_regex_extract(text))
	ents.extend(_obligation_phrases(text))

	# dedupe by span/type
	seen = set()
	unique = []
	for e in sorted(ents, key=lambda x: (x["start"], -len(x["text"]))):
		key = (e["start"], e["end"], e["type"].upper())
		if key in seen:
			continue
		seen.add(key)
		unique.append(e)

	return pd.DataFrame(unique) if unique else pd.DataFrame(columns=["text", "type", "start", "end"])
