# Leagel-Document-Analyser

Live Deploy Link: https://leagel-document-analyser-19.streamlit.app/




# ClauseWise - AI Legal Document Analyzer

## 🚀 **COMPLETED FEATURES**

### 1. **Multi-Format Document Upload**
- **Supported**: PDF, DOCX, TXT
- **Flow**: File → Text Extraction → Processing Pipeline
- **Status**: ✅ Complete

### 2. **Document Text Extraction**
- **PDF**: Selectable text + OCR fallback
- **DOCX**: Direct paragraph extraction
- **TXT**: UTF-8/Latin-1 encoding support
- **Flow**: Upload → Extract → Clean → Store
- **Status**: ✅ Complete

### 3. **Clause Segmentation & Breakdown**
- **Method**: Regex patterns + heading detection
- **Fallback**: Paragraph-based segmentation
- **Output**: {id, title, text} for each clause
- **Flow**: Clean text → Pattern matching → Clause list
- **Status**: ✅ Complete

### 4. **Named Entity Recognition (NER)**
- **Entities**: Parties, Dates, Money, Obligations, Emails, URLs
- **Methods**: IBM Watson NLU (primary) + spaCy (fallback)
- **Flow**: Text → Entity extraction → DataFrame display
- **Status**: ✅ Complete

### 5. **Clause Simplification**
- **Methods**: IBM Granite (primary) + Heuristic rules (fallback)
- **Rules**: Legal → Plain language conversion
- **Flow**: Clause text → AI processing → Simplified output
- **Status**: ✅ Complete

### 6. **Document Type Classification**
- **Types**: NDA, Lease, Employment Agreement, Service Agreement
- **Methods**: IBM Watson + Granite + Heuristic rules
- **Flow**: Document text → Classification → Confidence score
- **Status**: ✅ Complete

### 7. **Lease Timeline Visualization**
- **Features**: Start date, Term, End date extraction
- **Visual**: Graphviz flowchart
- **Flow**: Lease detection → Date parsing → Timeline diagram
- **Status**: ✅ Complete

### 8. **Text-to-Speech (TTS)**
- **Engine**: pyttsx3 (local)
- **Usage**: Play simplified clauses
- **Flow**: Text → Audio synthesis → WAV playback
- **Status**: ✅ Complete

### 9. **DOCX Export & Reprocessing**
- **Export**: Full text + Clause-structured
- **Reprocess**: Upload exported DOCX → Re-analyze
- **Flow**: Extract → Export → Upload → Reprocess
- **Status**: ✅ Complete

### 10. **OCR Cleanup Pipeline**
- **Features**: De-hyphenation, Line merging, Noise removal
- **Methods**: Regex + Pattern matching
- **Flow**: Raw OCR → Cleanup → Structured text
- **Status**: ✅ Complete

### 11. **IBM AI Integration**
- **Watson NLU**: Entity extraction + Classification
- **Granite**: Text generation + Zero-shot classification
- **Fallback**: Local models when IBM unavailable
- **Status**: ✅ Complete

## 12. **TECHNICAL FLOW**

```
1. UPLOAD → File validation & type detection
2. EXTRACT → Text extraction (native + OCR)
3. CLEAN → Text preprocessing & normalization
4. SEGMENT → Clause detection & breakdown
5. ANALYZE → NER, classification, simplification
6. VISUALIZE → Timeline diagrams (lease docs)
7. EXPORT → DOCX generation & reprocessing
```

## 📋 **USAGE INSTRUCTIONS**

### **For Scanned PDFs:**
1. Upload PDF in sidebar
2. Enable "Force OCR" checkbox
3. Select languages (e.g., English + Tamil)
4. Lower "Min chars" to 20-50
5. Wait for OCR processing
6. Check "Preview extracted text"

### **For Text-based PDFs:**
1. Upload PDF
2. Keep OCR settings default
3. Text extracted automatically

## 🚨 **TROUBLESHOOTING**

### **"No text extracted" Error:**
1. Check "🔧 Debug Info" expander
2. Verify OCR engines are available
3. Enable "Force OCR" in sidebar
4. Lower "Min chars" threshold
5. Install missing dependencies

### **Install Dependencies:**
```bash
pip install pytesseract easyocr opencv-python-headless pillow
# + Install Tesseract system binary
```

### **Language Issues:**
- Ensure language packs installed
- Check language code mapping
- Verify OCR engine support

## 🎯 **NEXT STEPS (Optional)**

1. **Fine-tune OCR** for specific document types
2. **Add more languages** as needed
3. **Custom clause patterns** for your domain
4. **Batch processing** for multiple documents
5. **API endpoints** for integration

## 📊 **FEATURE STATUS SUMMARY**

| Feature | Status | Notes |
|---------|--------|-------|
| Multi-format upload | ✅ Complete | PDF, DOCX, TXT |
| Clause extraction | ✅ Complete | Smart segmentation |
| NER analysis | ✅ Complete | Multiple engines |
| Text simplification | ✅ Complete | AI + rules |
| Document classification | ✅ Complete | 4 types supported |
| Timeline visualization | ✅ Complete | Lease documents |
| TTS playback | ✅ Complete | Local engine |
| Export/Reprocess | ✅ Complete | DOCX workflow |
| OCR cleanup | ✅ Complete | Post-processing |
| IBM integration | ✅ Complete | Primary + fallback |

**All core features are complete and working!** 
