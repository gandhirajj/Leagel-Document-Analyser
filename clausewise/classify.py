from typing import Tuple, Dict, Optional, List
import re

DOC_LABELS = [
	"NDA",
	"Lease",
	"Employment Agreement",
	"Service Agreement",
]


def _heuristic_doc_type(text: str) -> Tuple[str, float, Dict]:
	t = text.lower()
	votes = {}
	def add(label: str, w: float):
		votes[label] = votes.get(label, 0.0) + w
	if any(k in t for k in ["non-disclosure", "nondisclosure", "confidential information", "nda"]):
		add("NDA", 1.0)
	if any(k in t for k in ["tenant", "landlord", "rent", "lease term", "premises"]):
		add("Lease", 1.0)
	if any(k in t for k in ["employee", "employer", "salary", "position", "duties", "benefits"]):
		add("Employment Agreement", 1.0)
	if any(k in t for k in ["services", "statement of work", "sla", "service levels", "provider", "client"]):
		add("Service Agreement", 1.0)
	if not votes:
		return ("Service Agreement", 0.5, {"scores": {l: 0.0 for l in DOC_LABELS}})
	label = max(votes, key=votes.get)
	sum_scores = sum(votes.values())
	details = {"scores": votes}
	return (label, min(0.9, votes[label] / max(1.0, sum_scores) + 0.4), details)


def classify_document(text: str, providers: Optional[object] = None) -> Tuple[str, float, Dict]:
	if not text:
		return ("", 0.0, {})
	# Prefer IBM
	if providers and getattr(providers, "is_watson_ready", False):
		try:
			resp = providers.watson_nlu_classify(text)
			if resp and isinstance(resp, dict) and resp.get("label"):
				return (resp["label"], resp.get("score", 0.7), resp)
		except Exception:
			pass
	if providers and getattr(providers, "is_watsonx_ready", False):
		try:
			label = providers.watsonx_classify_zero_shot(text, DOC_LABELS)
			if label:
				return (label, 0.7, {"scores": {label: 0.7}})
		except Exception:
			pass
	return _heuristic_doc_type(text)


_CLAUSE_RULES: List[Tuple[str, List[str]]] = [
	("Definitions", ["means", "defined terms", "definition", "shall mean"]),
	("Confidentiality", ["confidential", "non-disclosure", "recipient", "discloser"]),
	("Payment Terms", ["payment", "fee", "invoice", "due", "payable"]),
	("Term and Termination", ["term", "termination", "expire", "renewal", "notice of termination"]),
	("Limitation of Liability", ["limitation of liability", "liable", "damages", "consequential"]),
	("Governing Law", ["governing law", "jurisdiction", "venue", "court"]),
	("Dispute Resolution", ["arbitration", "mediati", "dispute", "tribunal"]),
	("Intellectual Property", ["intellectual property", "ip", "license", "licence", "ownership"]),
	("Warranty", ["warranty", "warranties", "as is", "merchantability"]),
	("Indemnity", ["indemnify", "indemnification", "hold harmless"]),
	("Assignment", ["assign", "assignment", "transfer"]),
	("Notices", ["notice", "notify", "address for notice", "delivery"]),
]


def classify_clause_type(text: str, providers: Optional[object] = None) -> Tuple[str, float]:
	t = text.lower()
	best = ("General", 0.4)
	for label, keys in _CLAUSE_RULES:
		score = sum(1 for k in keys if k in t)
		if score > 0 and score + 0.4 > best[1]:
			best = (label, min(0.95, 0.5 + 0.2 * score))
	return best
