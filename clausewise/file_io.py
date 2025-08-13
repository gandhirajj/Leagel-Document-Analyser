import io
import os
import tempfile
from typing import List, Optional

import fitz  # PyMuPDF
from docx import Document

try:
	import pytesseract
	from PIL import Image
	import numpy as np
	import cv2
	_HAS_TESS = True
	print("✅ OCR Dependencies: Tesseract, PIL, OpenCV, NumPy loaded successfully")
except Exception as e:
	_HAS_TESS = False
	print(f"❌ OCR Dependencies failed to load: {e}")

try:
	import easyocr  # type: ignore
	_HAS_EASYOCR = True
	print("✅ EasyOCR loaded successfully")
except Exception as e:
	_HAS_EASYOCR = False
	print(f"❌ EasyOCR failed to load: {e}")

from .ocr_cleanup import clean_ocr_text


def test_ocr_engines():
	"""Test if OCR engines are working properly."""
	print("\n🔍 Testing OCR Engines...")
	
	if _HAS_TESS:
		try:
			version = pytesseract.get_tesseract_version()
			print(f"✅ Tesseract version: {version}")
		except Exception as e:
			print(f"❌ Tesseract binary not accessible: {e}")
			print("   Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
	else:
		print("❌ Tesseract not available")
	
	if _HAS_EASYOCR:
		try:
			reader = easyocr.Reader(['en'], gpu=False)
			print("✅ EasyOCR initialized successfully")
		except Exception as e:
			print(f"❌ EasyOCR initialization failed: {e}")
	else:
		print("❌ EasyOCR not available")
	
	print("🔍 OCR Engine Test Complete\n")


# Run OCR test on import
test_ocr_engines()


def _preprocess_for_ocr(pix) -> 'Image.Image':
	# Convert PyMuPDF pixmap to numpy array
	img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
	arr = np.array(img)
	# Grayscale
	gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
	# Remove noise
	gray = cv2.bilateralFilter(gray, 7, 50, 50)
	# Threshold (adaptive helps with uneven lighting)
	thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY, 31, 11)
	# Morph open to remove small seals/stamps noise
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
	opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
	# Return as PIL Image
	return Image.fromarray(opened)


def _ocr_with_easyocr(image, langs: List[str]) -> str:
	# EasyOCR language codes mapping
	lang_mapping = {
		"en": "en", "ta": "ta", "hi": "hi", "mr": "mr", 
		"bn": "bn", "te": "te", "kn": "kn", "ml": "ml"
	}
	# Convert to valid EasyOCR language codes
	valid_langs = [lang_mapping.get(lang, lang) for lang in langs if lang in lang_mapping]
	if not valid_langs:
		valid_langs = ["en"]  # fallback to English
	
	reader = easyocr.Reader(valid_langs, gpu=False)
	# easyocr expects numpy array (BGR or RGB). We pass RGB numpy
	arr = np.array(image)
	results = reader.readtext(arr, detail=1, paragraph=True)
	# Sort by top-left y, then x
	results = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
	texts = [r[1] for r in results if len(r) >= 2 and isinstance(r[1], str)]
	return "\n".join(texts)


def _read_pdf_bytes(data: bytes, force_ocr: bool = False, min_ocr_chars: int = 80, ocr_langs: Optional[List[str]] = None) -> str:
	text_parts = []
	langs = ocr_langs or ["en"]
	
	# Debug: Check OCR availability
	print(f"🔍 OCR Debug: force_ocr={force_ocr}, min_chars={min_ocr_chars}, langs={langs}")
	print(f"🔍 OCR Engines: Tesseract={_HAS_TESS}, EasyOCR={_HAS_EASYOCR}")
	
	# Validate and map language codes for Tesseract
	tess_lang_mapping = {
		"en": "eng", "ta": "tam", "hi": "hin", "mr": "mar",
		"bn": "ben", "te": "tel", "kn": "kan", "ml": "mal"
	}
	tess_langs = [tess_lang_mapping.get(lang, lang) for lang in langs if lang in tess_lang_mapping]
	if not tess_langs:
		tess_langs = ["eng"]  # fallback to English
	
	print(f"🔍 Tesseract languages: {tess_langs}")
	
	with fitz.open(stream=data, filetype="pdf") as doc:
		print(f"🔍 PDF has {len(doc)} pages")
		for page_num, page in enumerate(doc):
			text = page.get_text() or ""
			ocr_text = ""
			
			print(f"🔍 Page {page_num}: Original text length = {len(text.strip())}")
			
			# Check if we need OCR
			needs_ocr = force_ocr or len(text.strip()) < max(0, int(min_ocr_chars))
			print(f"🔍 Page {page_num}: Needs OCR = {needs_ocr} (force_ocr={force_ocr}, text_len={len(text.strip())}, min_chars={min_ocr_chars})")
			
			if needs_ocr:
				print(f"🔍 Page {page_num}: Starting OCR process...")
				pix = page.get_pixmap(dpi=300)
				print(f"🔍 Page {page_num}: Image size = {pix.width}x{pix.height}")
				
				if _HAS_TESS:
					try:
						print(f"🔍 Page {page_num}: Trying Tesseract OCR...")
						proc_img = _preprocess_for_ocr(pix)
						lang_str = "+".join(tess_langs)
						ocr_text = pytesseract.image_to_string(proc_img, config=f"--oem 3 --psm 6 -l {lang_str}")
						print(f"🔍 Page {page_num}: Tesseract raw output length = {len(ocr_text)}")
						if ocr_text and len(ocr_text.strip()) > 10:
							print(f"✓ Tesseract OCR succeeded on page {page_num} with langs: {lang_str}")
							print(f"🔍 Page {page_num}: Tesseract text preview: {ocr_text[:100]}...")
						else:
							print(f"🔍 Page {page_num}: Tesseract output too short or empty")
					except Exception as e:
						ocr_text = ""
						print(f"✗ Tesseract OCR failed on page {page_num}: {e}")
				
				if (not ocr_text or len(ocr_text.strip()) < 10) and _HAS_EASYOCR:
					try:
						print(f"🔍 Page {page_num}: Trying EasyOCR...")
						proc_img = _preprocess_for_ocr(pix)
						ocr_text = _ocr_with_easyocr(proc_img, langs)
						print(f"🔍 Page {page_num}: EasyOCR output length = {len(ocr_text)}")
						if ocr_text and len(ocr_text.strip()) > 10:
							print(f"✓ EasyOCR succeeded on page {page_num} with langs: {langs}")
							print(f"🔍 Page {page_num}: EasyOCR text preview: {ocr_text[:100]}...")
						else:
							print(f"🔍 Page {page_num}: EasyOCR output too short or empty")
					except Exception as e:
						ocr_text = ""
						print(f"✗ EasyOCR failed on page {page_num}: {e}")
				
				if not ocr_text or len(ocr_text.strip()) < 10:
					print(f"🔍 Page {page_num}: Both OCR engines failed or produced insufficient text")
			else:
				print(f"🔍 Page {page_num}: OCR not needed, using original text")
			
			# Use OCR text if available, otherwise fall back to original
			final_text = ocr_text if ocr_text and len(ocr_text.strip()) > len(text.strip()) else text
			print(f"🔍 Page {page_num}: Final text length = {len(final_text.strip())}")
			
			# Always add some text, even if it's minimal
			if final_text and final_text.strip():
				# Clean OCR text if it came from OCR
				if ocr_text and len(ocr_text.strip()) > len(text.strip()):
					final_text = clean_ocr_text(final_text)
					print(f"🔍 Page {page_num}: Cleaned OCR text length = {len(final_text.strip())}")
				text_parts.append(final_text)
			elif text and text.strip():
				# Fallback to original text even if OCR failed
				print(f"🔍 Page {page_num}: Using original text as fallback")
				text_parts.append(text)
			else:
				print(f"🔍 Page {page_num}: No text available from any source")
	
	result = "\n".join(text_parts)
	print(f"🔍 Final result length = {len(result)}")
	
	# Final fallback: if no text was extracted, try to get at least some text
	if not result.strip():
		print("🔍 No text extracted, trying fallback extraction...")
		try:
			with fitz.open(stream=data, filetype="pdf") as doc:
				for page_num, page in enumerate(doc):
					text = page.get_text() or ""
					if text.strip():
						result = text.strip()
						print(f"🔍 Fallback: Using page {page_num} text: {len(result)} chars")
						break
		except Exception as e:
			print(f"🔍 Fallback extraction failed: {e}")
	
	return result


def _read_docx_bytes(data: bytes) -> str:
	with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
		tmp.write(data)
		tmp.flush()
		path = tmp.name
	try:
		doc = Document(path)
		return "\n".join(p.text for p in doc.paragraphs)
	finally:
		try:
			os.remove(path)
		except OSError:
			pass


def _read_txt_bytes(data: bytes, encoding: str = "utf-8") -> str:
	try:
		return data.decode(encoding)
	except UnicodeDecodeError:
		# Fallback to latin-1
		return data.decode("latin-1", errors="ignore")


def extract_text_from_file(uploaded_file, force_ocr: bool = False, min_ocr_chars: int = 80, ocr_langs: Optional[List[str]] = None) -> str:
	"""Read text from a Streamlit UploadedFile supporting PDF, DOCX, TXT.
	If a PDF has no extractable text or too little text, OCR is attempted when available (Tesseract, then EasyOCR).
	Multi-language OCR is supported via ocr_langs (e.g., ["en", "ta"]).
	"""
	filename = uploaded_file.name.lower()
	data = uploaded_file.read()

	if filename.endswith(".pdf"):
		return _read_pdf_bytes(data, force_ocr=force_ocr, min_ocr_chars=min_ocr_chars, ocr_langs=ocr_langs)
	if filename.endswith(".docx"):
		return _read_docx_bytes(data)
	if filename.endswith(".txt"):
		return _read_txt_bytes(data)

	raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
	"""Public helper to read text from raw DOCX bytes."""
	return _read_docx_bytes(docx_bytes)
