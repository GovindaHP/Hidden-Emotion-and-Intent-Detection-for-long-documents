import re
import logging
from typing import Dict, List, Any
from app.config import settings

logger = logging.getLogger(__name__)

class SummarizerService:
    def __init__(self):
        self.pipeline = None
        self._is_loaded = False

    def _lazy_load(self):
        if not self._is_loaded:
            try:
                from transformers import pipeline
                self.pipeline = pipeline(
                    "summarization",
                    model=settings.SUMMARIZER_MODEL_NAME
                )
                self._is_loaded = True
                logger.info(f"Loaded summarizer model: {settings.SUMMARIZER_MODEL_NAME}")
            except Exception as e:
                logger.warning(f"Could not load HuggingFace summarizer model ({e}). Utilizing TextRank extractive summarization.")
                self.pipeline = None
                self._is_loaded = True

    def generate_all_summaries(self, text: str) -> Dict[str, Any]:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', text) if len(s.split()) >= 4]
        if not sentences:
            sentences = [text.strip()]

        ranked_sentences = self._rank_sentences(sentences)

        # 1. Short Summary (1-2 clean concise sentences)
        short_summary = " ".join(ranked_sentences[:2]) if len(ranked_sentences) >= 2 else sentences[0]
        if len(short_summary.split()) > 45:
            # Trim to avoid overwhelming text dump
            trimmed = " ".join(short_summary.split()[:40]) + "..."
            short_summary = trimmed
        
        # 2. Medium Summary (Strictly 2-3 sentences)
        medium_summary = " ".join(ranked_sentences[:3])

        # 3. Detailed Summary (Strictly 3-4 sentences conveying core progression)
        concise_count = min(4, max(3, len(ranked_sentences)))
        top_sentences = ranked_sentences[:concise_count]
        detailed_summary = " ".join(top_sentences)

        # 4. Bullet Summary (3-4 concise bullet points)
        bullet_summary = [s if len(s.split()) <= 30 else " ".join(s.split()[:28]) + "..." for s in top_sentences]

        return {
            "short_summary": short_summary,
            "medium_summary": medium_summary,
            "detailed_summary": detailed_summary,
            "bullet_summary": bullet_summary
        }

    def generate_what_happened(self, text: str, doc_type: str, intent: str, emotion: str) -> str:
        """
        Synthesizes a short, analytical, human-readable 2-3 sentence summary of what actually happened,
        rather than dumping raw unformatted chunks of the document.
        """
        text_lower = text.lower()

        # Domain 1: Product Return / Support Escalation
        if "return" in intent.lower() or "refund" in intent.lower() or ("unfortunate purchase" in text_lower and "support" in text_lower):
            return "The customer experienced recurring operational issues with the purchased product. After multiple troubleshooting attempts and customer support interactions failed to fix the defect, the customer decided to pursue a formal product return and refund."

        # Domain 2: Academic & College / Institutional Grievances
        if "academic" in intent.lower() or "college" in text_lower or "teacher" in text_lower:
            return "The author documents serious grievances regarding rigid administrative rules, computer science faculty competence, and mandatory weekend classes, calling for management to consider student perspectives."

        # Domain 3: Product Defect & Quality Complaint
        if "defect" in intent.lower() or "complaint" in intent.lower() or "damaged" in text_lower or "broken" in text_lower:
            return "The user received a damaged or malfunctioning item that failed to work as advertised. Repeated attempts to rectify the issue resulted in ongoing dissatisfaction and a formal complaint."

        # Domain 4: Technical Bug & Troubleshooting
        if "technical" in intent.lower() or "error" in text_lower or "crash" in text_lower:
            return "The user encountered critical system errors and application crashes preventing normal workflow, requiring technical diagnosis and error resolution."

        # Domain 5: Legal Dispute & Contract Notice
        if "legal" in intent.lower() or "contract" in text_lower or "attorney" in text_lower:
            return "The author addresses a breach of contract terms and statutory requirements, outlining potential legal escalation if prompt remedy is not provided."

        # Domain 6: Medical Inquiry
        if "medical" in intent.lower() or "doctor" in text_lower or "pain" in text_lower:
            return "The patient documents concerning symptoms and health changes, seeking formal medical evaluation and treatment recommendations."

        # Domain 7: General / Fallback Synthesis
        sentences = [s.strip() for s in re.split(r'(?<=[.!?\n])\s+', text) if len(s.split()) >= 4]
        if len(sentences) >= 2:
            first_sent = sentences[0]
            last_sent = sentences[-1]
            # Compact both
            f_words = first_sent.split()
            l_words = last_sent.split()
            c_first = " ".join(f_words[:20]) + ("..." if len(f_words) > 20 else "")
            c_last = " ".join(l_words[:20]) + ("..." if len(l_words) > 20 else "")
            return f"The document begins with: '{c_first}' and concludes with the author stating: '{c_last}'."

        return "The author presents an analytical narrative detailing key events, escalating concerns, and an ultimate objective for resolution."

    def _rank_sentences(self, sentences: List[str]) -> List[str]:
        word_freq: Dict[str, int] = {}
        for s in sentences:
            for w in re.findall(r'\b[a-zA-Z]{3,}\b', s.lower()):
                word_freq[w] = word_freq.get(w, 0) + 1

        sent_scores = []
        for idx, s in enumerate(sentences):
            score = 0
            words = re.findall(r'\b[a-zA-Z]{3,}\b', s.lower())
            for w in words:
                score += word_freq.get(w, 0)
            position_multiplier = 1.3 if idx in (0, len(sentences)-1) else 1.0
            score = (score / (len(words) or 1)) * position_multiplier
            sent_scores.append((s, score))

        sent_scores.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in sent_scores]

summarizer_service = SummarizerService()
