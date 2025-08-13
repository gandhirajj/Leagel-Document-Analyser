from typing import Optional
import re

try:
	from nltk.tokenize import sent_tokenize
	import nltk
	# Ensure resources for newer NLTK
	try:
		nltk.data.find("tokenizers/punkt")
	except LookupError:
		nltk.download("punkt", quiet=True)
	try:
		nltk.data.find("tokenizers/punkt_tab/english/")
	except LookupError:
		try:
			nltk.download("punkt_tab", quiet=True)
		except Exception:
			pass
except Exception:
	def sent_tokenize(text: str):
		return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


PLAIN_REPLACEMENTS = {
	"herein": "in this agreement",
	"hereof": "of this",
	"hereto": "to this",
	"thereof": "of that",
	"therein": "in that",
	"whereas": "because",
	"prior to": "before",
	"subsequent to": "after",
	"pursuant to": "under",
	"in the event that": "if",
}


def _heuristic_simplify(text: str) -> str:
	if not text:
		return ""
	out = text
	for k, v in PLAIN_REPLACEMENTS.items():
		out = re.sub(rf"\b{k}\b", v, out, flags=re.IGNORECASE)
	# Short sentences when extremely long
	try:
		sents = sent_tokenize(out)
	except Exception:
		sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", out) if s.strip()]
	if len(sents) <= 3:
		return out.strip()
	return " ".join(sents[:3]).strip()


def simplify_clause(text: str, providers: Optional[object] = None) -> str:
	if not text:
		return ""
	if providers and getattr(providers, "is_watsonx_ready", False):
		prompt = (
			"You are a legal assistant. Rewrite the following contract clause in plain language for a non-lawyer. "
			"Keep meaning intact, avoid changing obligations, remove redundancy, and keep key entities.\n\nClause:\n" + text + "\n\nPlain-language rewrite:"
		)
		try:
			resp = providers.watsonx_generate(prompt)
			if resp:
				return resp.strip()
		except Exception:
			pass
	return _heuristic_simplify(text)
