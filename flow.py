from typing import Optional, Dict
from dateutil import parser as dateparser
import re

import streamlit as st
from graphviz import Digraph


def _parse_date(s: str) -> Optional[str]:
	try:
		d = dateparser.parse(s, dayfirst=True, fuzzy=True)
		return d.strftime("%d %b %Y") if d else None
	except Exception:
		return None


def _extract_term(text: str) -> Optional[str]:
	m = re.search(r"\b(term|lease term)\b[\s\S]{0,50}\b(\d{1,2})\s*(years?|months?)\b", text, re.IGNORECASE)
	if not m:
		return None
	return f"{m.group(2)} {m.group(3)}"


def render_lease_timeline(text: str):
	start = None
	end = None
	for pat in [r"commencement\s*date\s*[:\-]?\s*(.+)", r"start\s*date\s*[:\-]?\s*(.+)"]:
		m = re.search(pat, text, re.IGNORECASE)
		if m:
			start = _parse_date(m.group(1))
			break
	for pat in [r"expiry|expiration\s*date\s*[:\-]?\s*(.+)"]:
		m = re.search(pat, text, re.IGNORECASE)
		if m:
			end = _parse_date(m.group(1))
			break
	term = _extract_term(text)

	dot = Digraph()
	dot.attr(rankdir="LR", fontsize="12")
	dot.node("lease", "Lease", shape="folder")
	if start:
		dot.node("start", f"Start\n{start}", shape="circle")
		dot.edge("lease", "start")
	if term:
		dot.node("term", f"Term\n{term}", shape="box")
		dot.edge("start" if start else "lease", "term")
	if end:
		dot.node("end", f"End\n{end}", shape="doublecircle")
		dot.edge("term" if term else ("start" if start else "lease"), "end")
	st.graphviz_chart(dot)
