import re
from typing import List, Dict, Optional, Tuple
import pandas as pd
from .providers import Providers


class LegalDocumentChatbot:
    """
    AI-powered chatbot for answering questions about legal documents.
    Uses IBM Granite or local models for intelligent responses.
    """
    
    def __init__(self, providers: Optional[Providers] = None):
        self.providers = providers
        self.document_context = ""
        self.conversation_history = []
        
        # Common legal question patterns
        self.question_patterns = {
            'termination': [
                'termination', 'terminate', 'cancel', 'cancellation', 'end contract',
                'how to end', 'how to cancel', 'notice period', 'termination rights'
            ],
            'payment': [
                'payment', 'pay', 'rent', 'fee', 'amount', 'due date', 'deadline',
                'how much', 'when to pay', 'payment terms', 'late fees'
            ],
            'renewal': [
                'renew', 'renewal', 'extend', 'extension', 'auto-renew',
                'how to renew', 'renewal terms', 'renewal notice'
            ],
            'obligations': [
                'obligation', 'duty', 'responsibility', 'must do', 'required',
                'what must I do', 'my responsibilities', 'compliance'
            ],
            'rights': [
                'rights', 'entitled', 'can I', 'am I allowed', 'permission',
                'what can I do', 'my rights', 'entitlements'
            ],
            'penalties': [
                'penalty', 'fine', 'breach', 'violation', 'consequence',
                'what happens if', 'penalty for', 'breach consequences'
            ],
            'liability': [
                'liability', 'responsible for', 'damages', 'indemnify',
                'who is responsible', 'liability for', 'damage claims'
            ]
        }
    
    def set_document_context(self, document_text: str):
        """Set the document text as context for the chatbot."""
        self.document_context = document_text
        print(f"📄 Document context set: {len(document_text)} characters")
    
    def ask_question(self, question: str) -> Dict:
        """
        Ask a question about the legal document.
        Returns a dictionary with answer and metadata.
        """
        if not self.document_context:
            return {
                'answer': 'No document loaded. Please upload a document first.',
                'confidence': 0.0,
                'source': 'system',
                'question_type': 'error'
            }
        
        # Add to conversation history
        self.conversation_history.append({
            'question': question,
            'timestamp': pd.Timestamp.now()
        })
        
        # Classify question type
        question_type = self._classify_question(question)
        
        # Generate answer using AI
        answer = self._generate_ai_answer(question, question_type)
        
        # Add answer to history
        self.conversation_history[-1]['answer'] = answer['answer']
        self.conversation_history[-1]['confidence'] = answer['confidence']
        
        return answer
    
    def _classify_question(self, question: str) -> str:
        """Classify the type of question being asked."""
        question_lower = question.lower()
        
        for category, patterns in self.question_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                return category
        
        return 'general'
    
    def _generate_ai_answer(self, question: str, question_type: str) -> Dict:
        """Generate AI-powered answer using IBM Granite or fallback methods."""
        
        # Try IBM Granite first
        if self.providers and self.providers.is_watsonx_ready:
            try:
                return self._generate_granite_answer(question, question_type)
            except Exception as e:
                print(f"Granite answer generation failed: {e}")
        
        # Fallback to local methods
        return self._generate_local_answer(question, question_type)
    
    def _generate_granite_answer(self, question: str, question_type: str) -> Dict:
        """Generate answer using IBM Granite."""
        
        # Create context-aware prompt
        prompt = self._create_granite_prompt(question, question_type)
        
        try:
            response = self.providers.watsonx_generate(prompt)
            
            return {
                'answer': response,
                'confidence': 0.85,  # High confidence for Granite
                'source': 'IBM Granite',
                'question_type': question_type
            }
        except Exception as e:
            print(f"Granite generation error: {e}")
            return self._generate_local_answer(question, question_type)
    
    def _create_granite_prompt(self, question: str, question_type: str) -> str:
        """Create a context-aware prompt for Granite."""
        
        prompt = f"""You are a legal document assistant. Answer the following question based on the provided legal document context.

Document Context (first 2000 characters):
{self.document_context[:2000]}...

Question: {question}
Question Type: {question_type}

Instructions:
1. Answer based ONLY on the document context provided
2. If the information is not in the document, say "This information is not specified in the document"
3. Be clear, concise, and legally accurate
4. Use simple language when possible
5. If quoting the document, use quotation marks

Answer:"""
        
        return prompt
    
    def _generate_local_answer(self, question: str, question_type: str) -> Dict:
        """Generate answer using local pattern matching and heuristics."""
        
        # Extract relevant information from document
        relevant_info = self._extract_relevant_info(question, question_type)
        
        if relevant_info:
            answer = self._format_local_answer(question, relevant_info, question_type)
            confidence = 0.6  # Medium confidence for local methods
        else:
            answer = "I couldn't find specific information about that in the document. Please check the document content or rephrase your question."
            confidence = 0.3
        
        return {
            'answer': answer,
            'confidence': confidence,
            'source': 'Local Analysis',
            'question_type': question_type
        }
    
    def _extract_relevant_info(self, question: str, question_type: str) -> List[str]:
        """Extract relevant information from document based on question."""
        relevant_sections = []
        question_lower = question.lower()
        
        # Split document into paragraphs
        paragraphs = [p.strip() for p in self.document_context.split('\n') if p.strip()]
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            
            # Check if paragraph contains relevant keywords
            relevance_score = 0
            
            # Question-specific keywords
            if question_type == 'termination':
                if any(word in paragraph_lower for word in ['terminate', 'termination', 'cancel', 'notice']):
                    relevance_score += 3
            elif question_type == 'payment':
                if any(word in paragraph_lower for word in ['pay', 'payment', 'rent', 'fee', 'due']):
                    relevance_score += 3
            elif question_type == 'renewal':
                if any(word in paragraph_lower for word in ['renew', 'renewal', 'extend', 'extension']):
                    relevance_score += 3
            elif question_type == 'obligations':
                if any(word in paragraph_lower for word in ['shall', 'must', 'required', 'obligation']):
                    relevance_score += 2
            elif question_type == 'rights':
                if any(word in paragraph_lower for word in ['right', 'entitled', 'may', 'can']):
                    relevance_score += 2
            
            # General relevance based on question words
            question_words = question_lower.split()
            for word in question_words:
                if len(word) > 3 and word in paragraph_lower:
                    relevance_score += 1
            
            if relevance_score >= 2:
                relevant_sections.append(paragraph)
        
        return relevant_sections[:3]  # Return top 3 most relevant sections
    
    def _format_local_answer(self, question: str, relevant_info: List[str], question_type: str) -> str:
        """Format the extracted information into a coherent answer."""
        
        if not relevant_info:
            return "No relevant information found in the document."
        
        answer = f"Based on the document, here's what I found regarding your question about {question_type}:\n\n"
        
        for i, info in enumerate(relevant_info, 1):
            # Clean up the text
            clean_info = re.sub(r'\s+', ' ', info).strip()
            answer += f"{i}. {clean_info}\n\n"
        
        answer += "Note: This is an automated analysis. For legal advice, please consult with a qualified attorney."
        
        return answer
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions based on document content."""
        suggestions = []
        
        if not self.document_context:
            return suggestions
        
        # Generate suggestions based on document content
        doc_lower = self.document_context.lower()
        
        if any(word in doc_lower for word in ['terminate', 'termination', 'cancel']):
            suggestions.append("What are the termination rights and notice periods?")
        
        if any(word in doc_lower for word in ['pay', 'payment', 'rent', 'fee']):
            suggestions.append("What are the payment terms and due dates?")
        
        if any(word in doc_lower for word in ['renew', 'renewal', 'extend']):
            suggestions.append("How does the renewal process work?")
        
        if any(word in doc_lower for word in ['obligation', 'duty', 'responsibility']):
            suggestions.append("What are my main obligations under this contract?")
        
        if any(word in doc_lower for word in ['right', 'entitled', 'may']):
            suggestions.append("What rights am I entitled to under this contract?")
        
        if any(word in doc_lower for word in ['penalty', 'breach', 'violation']):
            suggestions.append("What are the penalties for breaching this contract?")
        
        if any(word in doc_lower for word in ['liability', 'responsible', 'damages']):
            suggestions.append("What are the liability and damage provisions?")
        
        # Add general questions
        suggestions.extend([
            "What is the main purpose of this document?",
            "Who are the parties involved?",
            "What is the duration of this agreement?",
            "Are there any special conditions or exceptions?"
        ])
        
        return suggestions[:8]  # Return top 8 suggestions
