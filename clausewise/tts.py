import io
import re
import platform
from typing import Optional

# Try multiple TTS engines for better compatibility
try:
	import pyttsx3
	PYTTSX3_AVAILABLE = True
except ImportError:
	PYTTSX3_AVAILABLE = False

try:
	import gtts
	from gtts import gTTS
	GTTS_AVAILABLE = True
except ImportError:
	GTTS_AVAILABLE = False

try:
	import speech_recognition as sr
	SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
	SPEECH_RECOGNITION_AVAILABLE = False


def clean_legal_text_for_speech(text: str) -> str:
	"""
	Specialized cleaning for legal documents to remove formatting artifacts
	that make speech sound unnatural.
	"""
	if not text:
		return ""
	
	# Remove legal document formatting patterns
	cleaned = text
	
	# Remove underscores used for blank spaces (common in legal forms)
	cleaned = re.sub(r'_{3,}', ' blank space ', cleaned)  # Multiple underscores = blank space
	cleaned = re.sub(r'_{1,2}', ' ', cleaned)  # Single/double underscores = space
	
	# Remove dashes used for blank spaces
	cleaned = re.sub(r'-{3,}', ' blank space ', cleaned)  # Multiple dashes = blank space
	cleaned = re.sub(r'-{1,2}', ' ', cleaned)  # Single/double dashes = space
	
	# Remove "etc." patterns that are common in legal documents
	cleaned = re.sub(r'\betc\.\b', 'and so on', cleaned)
	
	# Remove "Sri." and "Smt." abbreviations (common in Indian legal documents)
	cleaned = re.sub(r'\bSri\.\b', 'Mister', cleaned)
	cleaned = re.sub(r'\bSmt\.\b', 'Missus', cleaned)
	
	# Remove other legal abbreviations
	cleaned = re.sub(r'\bDr\.\b', 'Doctor', cleaned)
	cleaned = re.sub(r'\bMr\.\b', 'Mister', cleaned)
	cleaned = re.sub(r'\bMrs\.\b', 'Missus', cleaned)
	cleaned = re.sub(r'\bMs\.\b', 'Miss', cleaned)
	
	# Remove standalone punctuation and formatting
	cleaned = re.sub(r'\s+[.,!?;:]\s*', ' ', cleaned)
	cleaned = re.sub(r'\s+[_\-\/\\\[\]{}()"\']\s*', ' ', cleaned)
	
	# Remove multiple spaces
	cleaned = re.sub(r'\s+', ' ', cleaned)
	
	# Remove leading/trailing whitespace
	cleaned = cleaned.strip()
	
	return cleaned


def clean_text_for_speech(text: str) -> str:
	"""
	Clean text for speech synthesis by removing special characters and formatting.
	Only keeps words, numbers, and basic punctuation that should be spoken.
	"""
	if not text:
		return ""
	
	# First, apply legal document specific cleaning
	cleaned = clean_legal_text_for_speech(text)
	
	# Remove underscores, dashes, and other unwanted characters
	# Remove: underscores, multiple dashes, brackets, parentheses, quotes, slashes, etc.
	cleaned = re.sub(r'[_\-\/\\\[\]{}()"\']', ' ', cleaned)
	
	# Remove multiple consecutive underscores or dashes
	cleaned = re.sub(r'_{2,}', ' ', cleaned)  # Multiple underscores
	cleaned = re.sub(r'-{2,}', ' ', cleaned)  # Multiple dashes
	
	# Remove standalone underscores and dashes
	cleaned = re.sub(r'\s+_\s+', ' ', cleaned)  # Standalone underscore
	cleaned = re.sub(r'\s+-\s+', ' ', cleaned)  # Standalone dash
	
	# Remove special characters but keep basic punctuation that should be spoken
	# Keep: letters, numbers, spaces, periods, commas, question marks, exclamation marks, colons, semicolons
	# Remove: other special characters
	cleaned = re.sub(r'[^\w\s.,!?;:]', ' ', cleaned)
	
	# Remove extra whitespace
	cleaned = re.sub(r'\s+', ' ', cleaned)
	
	# Remove standalone punctuation marks
	cleaned = re.sub(r'\s+[.,!?;:]\s*', ' ', cleaned)
	
	# Clean up parentheses content (often contains formatting)
	cleaned = re.sub(r'\([^)]*\)', '', cleaned)
	
	# Remove multiple spaces
	cleaned = re.sub(r'\s+', ' ', cleaned)
	
	# Remove leading/trailing whitespace
	cleaned = cleaned.strip()
	
	# If text becomes too short after cleaning, return original
	if len(cleaned) < 10:
		return text
	
	return cleaned


def get_available_tts_engines():
	"""Get list of available TTS engines on the system."""
	engines = []
	
	if PYTTSX3_AVAILABLE:
		try:
			engine = pyttsx3.init()
			voices = engine.getProperty('voices')
			if voices:
				engines.append("pyttsx3 (Local)")
		except Exception:
			pass
	
	if GTTS_AVAILABLE:
		engines.append("gTTS (Google)")
	
	return engines


def synthesize_to_wav_bytes(text: str) -> Optional[bytes]:
	"""
	Generate speech from text using available TTS engines.
	Tries multiple engines in order of preference.
	"""
	if not text:
		return None
	
	# Clean text for speech
	clean_text = clean_text_for_speech(text)
	
	# Try pyttsx3 first (local, fast)
	if PYTTSX3_AVAILABLE:
		try:
			return _synthesize_with_pyttsx3(clean_text)
		except Exception as e:
			print(f"pyttsx3 failed: {e}")
	
	# Try gTTS as fallback (online, requires internet)
	if GTTS_AVAILABLE:
		try:
			return _synthesize_with_gtts(clean_text)
		except Exception as e:
			print(f"gTTS failed: {e}")
	
	# If all engines fail, return None
	return None


def _synthesize_with_pyttsx3(text: str) -> Optional[bytes]:
	"""Generate speech using pyttsx3 (local engine)."""
	try:
		engine = pyttsx3.init()
		
		# Configure engine properties for better quality
		engine.setProperty('rate', 150)  # Speed of speech
		engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
		
		# Try to set a good voice
		voices = engine.getProperty('voices')
		if voices:
			# Prefer female voices if available
			female_voices = [v for v in voices if 'female' in v.name.lower() or 'zira' in v.name.lower()]
			if female_voices:
				engine.setProperty('voice', female_voices[0].id)
			else:
				engine.setProperty('voice', voices[0].id)
		
		# Save to temporary file
		import tempfile
		import os
		with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
			path = tmp.name
		
		engine.save_to_file(text, path)
		engine.runAndWait()
		
		# Read the generated audio
		with open(path, "rb") as f:
			data = f.read()
		
		# Clean up temporary file
		try:
			os.remove(path)
		except OSError:
			pass
		
		return data
		
	except Exception as e:
		print(f"pyttsx3 synthesis failed: {e}")
		return None


def _synthesize_with_gtts(text: str) -> Optional[bytes]:
	"""Generate speech using gTTS (Google Text-to-Speech)."""
	try:
		# Limit text length for gTTS (max ~5000 characters)
		if len(text) > 4000:
			text = text[:4000] + "..."
		
		# Create gTTS object
		tts = gTTS(text=text, lang='en', slow=False)
		
		# Save to temporary file
		import tempfile
		import os
		with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
			path = tmp.name
		
		tts.save(path)
		
		# Read the generated audio
		with open(path, "rb") as f:
			data = f.read()
		
		# Clean up temporary file
		try:
			os.remove(path)
		except OSError:
			pass
		
		return data
		
	except Exception as e:
		print(f"gTTS synthesis failed: {e}")
		return None


def get_tts_status():
	"""Get detailed status of TTS engines."""
	status = {
		"pyttsx3": {
			"available": PYTTSX3_AVAILABLE,
			"type": "Local",
			"description": "Fast local TTS engine"
		},
		"gTTS": {
			"available": GTTS_AVAILABLE,
			"type": "Online",
			"description": "Google Text-to-Speech (requires internet)"
		}
	}
	
	# Check if pyttsx3 can actually initialize
	if PYTTSX3_AVAILABLE:
		try:
			engine = pyttsx3.init()
			voices = engine.getProperty('voices')
			status["pyttsx3"]["working"] = len(voices) > 0
			status["pyttsx3"]["voice_count"] = len(voices)
		except Exception:
			status["pyttsx3"]["working"] = False
			status["pyttsx3"]["error"] = "Failed to initialize"
	
	return status


def test_tts_engines():
	"""Test TTS engines and return detailed status."""
	test_text = "Hello, this is a test of the text to speech system."
	results = {}
	
	# Test pyttsx3
	if PYTTSX3_AVAILABLE:
		try:
			audio = _synthesize_with_pyttsx3(test_text)
			results["pyttsx3"] = {
				"status": "Working",
				"audio_size": len(audio) if audio else 0,
				"error": None
			}
		except Exception as e:
			results["pyttsx3"] = {
				"status": "Failed",
				"audio_size": 0,
				"error": str(e)
			}
	else:
		results["pyttsx3"] = {
			"status": "Not Available",
			"audio_size": 0,
			"error": "Package not installed"
		}
	
	# Test gTTS
	if GTTS_AVAILABLE:
		try:
			audio = _synthesize_with_gtts(test_text)
			results["gTTS"] = {
				"status": "Working",
				"audio_size": len(audio) if audio else 0,
				"error": None
			}
		except Exception as e:
			results["gTTS"] = {
				"status": "Failed",
				"audio_size": 0,
				"error": str(e)
			}
	else:
		results["gTTS"] = {
			"status": "Not Available",
			"audio_size": 0,
			"error": "Package not installed"
		}
	
	return results
