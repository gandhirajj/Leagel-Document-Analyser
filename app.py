import io
import os
from typing import List, Dict, Optional

import streamlit as st
from dotenv import load_dotenv

# Local modules
from clausewise.file_io import extract_text_from_file, extract_text_from_docx_bytes
from clausewise.preprocess import clean_text
from clausewise.segmentation import segment_into_clauses
from clausewise.ner import extract_entities
from clausewise.simplify import simplify_clause
from clausewise.classify import classify_document, classify_clause_type
from clausewise.tts import synthesize_to_wav_bytes
from clausewise.providers import Providers
from clausewise.ui import inject_css
from clausewise.export import generate_docx_bytes, generate_docx_from_clauses
from clausewise.flow import render_lease_timeline
from clausewise.deadlines import extract_deadlines_and_obligations, generate_deadlines_dataframe, get_upcoming_deadlines, get_overdue_deadlines
from clausewise.chatbot import LegalDocumentChatbot


def bootstrap():
	load_dotenv()
	st.set_page_config(page_title="ClauseWise – AI Legal Document Analyzer", layout="wide")
	inject_css()


def init_session_state():
	if "raw_text" not in st.session_state:
		st.session_state.raw_text = ""
	if "clean_text" not in st.session_state:
		st.session_state.clean_text = ""
	if "clauses" not in st.session_state:
		st.session_state.clauses = []
	if "active_clause_idx" not in st.session_state:
		st.session_state.active_clause_idx = 0
	if "deadlines" not in st.session_state:
		st.session_state.deadlines = []
	if "chatbot" not in st.session_state:
		st.session_state.chatbot = None


def render_sidebar(providers: Providers):
	st.sidebar.title("ClauseWise")
	st.sidebar.caption("AI-powered legal document analyzer")

	uploaded = st.sidebar.file_uploader(
		"Upload a document", type=["pdf", "docx", "txt"], accept_multiple_files=False
	)

	use_ibm = st.sidebar.checkbox("Use IBM services if available", value=True)
	st.sidebar.divider()
	st.sidebar.subheader("OCR Settings")
	force_ocr = st.sidebar.checkbox("Force OCR for PDFs", value=False, help="Use OCR even if the PDF has selectable text.")
	min_chars = st.sidebar.slider("Min chars before OCR", min_value=0, max_value=500, value=80, step=10,
		help="If extracted text is shorter than this per page, try OCR.")
	ocr_langs = st.sidebar.multiselect("OCR Languages", ["en", "ta", "hi", "mr", "bn", "te", "kn", "ml"], default=["en"], help="Add languages present in the scan (e.g., Tamil 'ta').")

	if uploaded is not None:
		with st.spinner("Reading document…"):
			text = extract_text_from_file(uploaded, force_ocr=force_ocr, min_ocr_chars=min_chars, ocr_langs=ocr_langs)
			st.session_state.raw_text = text
			st.session_state.clean_text = clean_text(text)
			st.session_state.clauses = segment_into_clauses(st.session_state.clean_text)
			
			# Extract deadlines and obligations
			if text and len(text.strip()) > 0:
				st.session_state.deadlines = extract_deadlines_and_obligations(text)
				
				# Initialize chatbot with document context
				if st.session_state.chatbot is None:
					st.session_state.chatbot = LegalDocumentChatbot(providers if use_ibm else None)
				st.session_state.chatbot.set_document_context(text)
			
			# Debug: Show what was extracted
			st.sidebar.success(f"✅ Extracted {len(text)} characters")
			if len(text) > 0:
				st.sidebar.info(f"First 50 chars: {text[:50]}...")
				if st.session_state.deadlines:
					st.sidebar.info(f"📅 Found {len(st.session_state.deadlines)} deadlines/obligations")
			else:
				st.sidebar.error("❌ No text extracted from file")

	st.sidebar.divider()
	st.sidebar.write("Providers")
	st.sidebar.write(
		f"Watson NLU: {'ready' if providers.is_watson_ready and use_ibm else 'off'} | "
		f"Granite: {'ready' if providers.is_watsonx_ready and use_ibm else 'off'}"
	)

	return uploaded, use_ibm


def main():
	bootstrap()
	init_session_state()

	providers = Providers()

	uploaded, use_ibm = render_sidebar(providers)

	st.title("ClauseWise")
	st.caption("Upload legal documents and analyze clauses, entities, and document type")

	if not uploaded and not st.session_state.clean_text:
		st.info("Upload a PDF, DOCX, or TXT file from the sidebar to begin.")
		return

	# Debug info
	with st.expander("🔧 Debug Info", expanded=False):
		col1, col2 = st.columns(2)
		with col1:
			st.write("**OCR Status:**")
			try:
				import pytesseract
				st.success("✓ Tesseract available")
			except ImportError:
				st.error("✗ Tesseract not installed")
			try:
				import easyocr
				st.success("✓ EasyOCR available")
			except ImportError:
				st.error("✗ EasyOCR not installed")
		with col2:
			st.write("**File Info:**")
			if uploaded:
				st.write(f"Type: {uploaded.type}")
				st.write(f"Size: {uploaded.size} bytes")
				st.write(f"Name: {uploaded.name}")
		
		# Add text debugging
		st.write("**Text Debug:**")
		st.write(f"Raw text length: {len(st.session_state.raw_text)}")
		st.write(f"Clean text length: {len(st.session_state.clean_text)}")
		st.write(f"Clean text empty: {not st.session_state.clean_text.strip()}")
		if st.session_state.clean_text:
			st.write(f"First 100 chars: {repr(st.session_state.clean_text[:100])}")
	
		# Session state debugging
		st.write("**Session State:**")
		st.write(f"Session keys: {list(st.session_state.keys())}")
		st.write(f"Raw text in session: {repr(st.session_state.get('raw_text', 'NOT_FOUND'))[:100]}")
		st.write(f"Clean text in session: {repr(st.session_state.get('clean_text', 'NOT_FOUND'))[:100]}")

	# Diagnostics & export
	col1, col2, col3 = st.columns(3)
	with col1:
		st.metric("Characters", len(st.session_state.clean_text))
	with col2:
		st.metric("Clauses detected", len(st.session_state.clauses))
	with col3:
		st.metric("Lines", st.session_state.clean_text.count("\n") + 1)
	
	# Preview extracted text with better logic
	with st.expander("Preview extracted text", expanded=False):
		if st.session_state.clean_text and len(st.session_state.clean_text.strip()) > 0:
			preview = st.session_state.clean_text[:1500]
			st.code(preview)
			if len(st.session_state.clean_text) > 1500:
				st.caption(f"... and {len(st.session_state.clean_text) - 1500} more characters")
		else:
			st.error("❌ No text extracted!")
			st.write("**Troubleshooting:**")
			st.write("1. Check if 'Force OCR' is enabled in sidebar")
			st.write("2. Lower the 'Min chars before OCR' value")
			st.write("3. Ensure Tesseract is installed on your system")
			st.write("4. Try uploading a DOCX/TXT version instead")
		
		# Export buttons
		dl1, dl2 = st.columns(2)
		with dl1:
			if st.button("Download DOCX (full text)"):
				docx_bytes = generate_docx_bytes(st.session_state.clean_text, title="Extracted Document")
				st.download_button("Save DOCX", data= docx_bytes, file_name="clausewise_extracted.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
		with dl2:
			if st.button("Download DOCX (by clauses)"):
				docx_bytes = generate_docx_from_clauses(st.session_state.clauses, title="Extracted Clauses")
				st.download_button("Save DOCX", data=docx_bytes, file_name="clausewise_clauses.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

	if len(st.session_state.clean_text) < 20:
		st.warning("Very little text extracted. Try toggling Force OCR or increasing the Min chars slider in the sidebar.")

	tabs = st.tabs([
		"Clause Simplification",
		"Named Entity Recognition (NER)",
		"Clause Extraction & Breakdown",
		"Document Type Classification",
		"📅 Deadline & Obligation Tracker",  # New tab
		"🤖 AI Q&A Chatbot",               # New tab
	])

	# Tab 1: Clause Simplification
	with tabs[0]:
		if not st.session_state.clauses:
			st.warning("No clauses detected.")
		else:
			clause_options = [f"{c['id']}: {c['title']}" if c.get('title') else f"Clause {c['id']}" for c in st.session_state.clauses]
			idx = st.selectbox("Choose a clause to simplify", options=list(range(len(clause_options))), format_func=lambda i: clause_options[i], index=min(st.session_state.active_clause_idx, len(clause_options)-1))
			st.session_state.active_clause_idx = idx
			clause = st.session_state.clauses[idx]

			cols = st.columns(2)
			with cols[0]:
				st.subheader("Original Clause")
				st.write(clause["text"])
			with cols[1]:
				st.subheader("Simplified Clause")
				simplified = simplify_clause(clause["text"], providers if use_ibm else None)
				st.write(simplified)
				if st.button("🔊 Play simplified audio"):
					audio_bytes = synthesize_to_wav_bytes(simplified)
					if audio_bytes:
						st.audio(audio_bytes, format="audio/wav")
					else:
						st.warning("TTS engine unavailable on this system.")

	# Tab 2: NER
	with tabs[1]:
		scope = st.radio("Analyze", ["Selected clause", "Entire document"], index=0, horizontal=True)
		text_for_ner = None
		if scope == "Selected clause" and st.session_state.clauses:
			idx2 = st.selectbox(
				"Choose a clause for NER",
				options=list(range(len(st.session_state.clauses))),
				format_func=lambda i: st.session_state.clauses[i].get("title") or f"Clause {st.session_state.clauses[i]['id']}",
				key="ner_clause_select",
			)
			text_for_ner = st.session_state.clauses[idx2]["text"]
		else:
			text_for_ner = st.session_state.clean_text

		if text_for_ner:
			entities = extract_entities(text_for_ner, providers if use_ibm else None)
			if not entities.empty:
				st.dataframe(entities, use_container_width=True)
			else:
				st.info("No entities found. Install spaCy models or enable IBM Watson for richer NER.")

	# Tab 3: Clause Extraction & Breakdown
	with tabs[2]:
		if not st.session_state.clauses:
			st.warning("No clauses detected.")
		else:
			for clause in st.session_state.clauses:
				with st.expander(clause.get("title") or f"Clause {clause['id']}"):
					st.write(clause["text"])
					label, score = classify_clause_type(clause["text"], providers if use_ibm else None)
					st.caption(f"Predicted clause type: {label} (confidence {score:.2f})")

	# Tab 4: Document Type Classification
	with tabs[3]:
		label, score, details = classify_document(st.session_state.clean_text, providers if use_ibm else None)
		st.metric("Predicted document type", label or "(unknown)", delta=f"confidence {score:.2f}")
		if details:
			st.json(details)
		if (label or "").lower().startswith("lease"):
			st.subheader("Lease Timeline")
			render_lease_timeline(st.session_state.clean_text)

	# Tab 5: Deadline & Obligation Tracker
	with tabs[4]:
		st.subheader("📅 Deadline & Obligation Tracker")
		st.caption("Automatically extracted deadlines and obligations from your document")
		
		if not st.session_state.deadlines:
			st.warning("No deadlines or obligations detected. Upload a document to analyze.")
		else:
			# Summary metrics
			col1, col2, col3, col4 = st.columns(4)
			with col1:
				total_deadlines = len(st.session_state.deadlines)
				st.metric("Total Deadlines", total_deadlines)
			with col2:
				overdue = len([d for d in st.session_state.deadlines if d['urgency'] == 'Overdue'])
				st.metric("Overdue", overdue, delta_color="inverse")
			with col3:
				critical = len([d for d in st.session_state.deadlines if d['urgency'] == 'Critical'])
				st.metric("Critical (≤7 days)", critical, delta_color="inverse")
			with col4:
				upcoming = len([d for d in st.session_state.deadlines if d['urgency'] in ['High', 'Medium']])
				st.metric("Upcoming", upcoming)
			
			# Filter options
			st.subheader("Filter & View")
			filter_col1, filter_col2 = st.columns(2)
			with filter_col1:
				urgency_filter = st.multiselect(
					"Filter by Urgency",
					["Overdue", "Critical", "High", "Medium", "Low", "Unknown"],
					default=["Overdue", "Critical", "High"]
				)
			with filter_col2:
				category_filter = st.multiselect(
					"Filter by Category",
					["Financial", "Contract Management", "Compliance", "General"],
					default=["Financial", "Contract Management", "Compliance"]
				)
			
			# Filter deadlines
			filtered_deadlines = [
				d for d in st.session_state.deadlines
				if d['urgency'] in urgency_filter and d['category'] in category_filter
			]
			
			if filtered_deadlines:
				# Display deadlines table
				df = generate_deadlines_dataframe(filtered_deadlines)
				st.dataframe(df, use_container_width=True)
				
				# Export deadlines
				if st.button("📊 Export Deadlines to CSV"):
					csv = df.to_csv(index=False)
					st.download_button(
						"Download CSV",
						data=csv,
						file_name="clausewise_deadlines.csv",
						mime="text/csv"
					)
			else:
				st.info("No deadlines match the selected filters.")
			
			# Upcoming deadlines section
			if st.session_state.deadlines:
				st.subheader("🚨 Critical & Upcoming Deadlines")
				upcoming = get_upcoming_deadlines(st.session_state.deadlines, days_ahead=90)
				
				if upcoming:
					for deadline in upcoming[:5]:  # Show top 5
						days_until = deadline.get('days_until', 0)
						urgency_color = {
							'Critical': '🔴',
							'High': '🟠',
							'Medium': '🟡',
							'Low': '🟢'
						}.get(deadline['urgency'], '⚪')
						
						with st.expander(f"{urgency_color} {deadline['type']} - {deadline['text']}"):
							st.write(f"**Context:** {deadline['context']}")
							st.write(f"**Date:** {deadline['parsed_date']}")
							st.write(f"**Urgency:** {deadline['urgency']}")
							st.write(f"**Category:** {deadline['category']}")
							if days_until >= 0:
								st.write(f"**Days until deadline:** {days_until}")
							else:
								st.write(f"**Days overdue:** {abs(days_until)}")
				else:
					st.success("✅ No critical deadlines in the next 90 days!")

	# Tab 6: AI Q&A Chatbot
	with tabs[5]:
		st.subheader("🤖 AI Q&A Chatbot")
		st.caption("Ask questions about your legal document and get AI-powered answers")
		
		if not st.session_state.chatbot or not st.session_state.clean_text:
			st.warning("Upload a document first to enable the AI chatbot.")
		else:
			# Chat interface
			st.subheader("💬 Ask a Question")
			
			# Question input
			question = st.text_input(
				"Type your question here...",
				placeholder="e.g., What are my termination rights?",
				key="chatbot_question"
			)
			
			if st.button("🚀 Ask AI", type="primary"):
				if question.strip():
					with st.spinner("🤖 AI is thinking..."):
						response = st.session_state.chatbot.ask_question(question)
					
					# Display answer
					st.subheader("🤖 AI Answer")
					st.write(response['answer'])
					
					# Answer metadata
					meta_col1, meta_col2, meta_col3 = st.columns(3)
					with meta_col1:
						st.caption(f"**Confidence:** {response['confidence']:.1%}")
					with meta_col2:
						st.caption(f"**Source:** {response['source']}")
					with meta_col3:
						st.caption(f"**Type:** {response['question_type']}")
				else:
					st.error("Please enter a question.")
			
			# Conversation history
			if st.session_state.chatbot:
				history = st.session_state.chatbot.get_conversation_history()
				if history:
					st.subheader("📚 Conversation History")
					
					# Clear history button
					if st.button("🗑️ Clear History"):
						st.session_state.chatbot.clear_history()
						st.rerun()
					
					# Display history
					for i, conv in enumerate(reversed(history[-10:])):  # Show last 10
						with st.expander(f"Q: {conv['question'][:100]}..."):
							st.write(f"**Question:** {conv['question']}")
							st.write(f"**Answer:** {conv['answer']}")
							st.caption(f"**Confidence:** {conv['confidence']:.1%} | **Source:** {conv.get('source', 'Unknown')}")


if __name__ == "__main__":
	main()
