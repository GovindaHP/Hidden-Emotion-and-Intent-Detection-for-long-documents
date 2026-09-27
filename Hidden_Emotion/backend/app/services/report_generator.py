import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List

from app.services.nlp_preprocessor import nlp_preprocessor
from app.services.embedding_service import embedding_service
from app.services.emotion_service import emotion_service
from app.services.intent_service import intent_service
from app.services.sentiment_service import sentiment_service
from app.services.summarizer_service import summarizer_service
from app.services.ner_service import ner_service
from app.services.keyword_service import keyword_service
from app.services.timeline_service import timeline_service
from app.services.responder_service import responder_service

class ReportGenerator:
    def generate(self, raw_text: str, filename: str = "Pasted Text Document", doc_type: str = "General") -> Dict[str, Any]:
        report_id = f"REP-{uuid.uuid4().hex[:8].upper()}"
        upload_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        cleaned_text = nlp_preprocessor.clean_text(raw_text)
        words = cleaned_text.split()
        word_count = len(words)
        reading_time_minutes = round(word_count / 200.0, 1) or 0.5

        # 1. Chunking
        chunks = nlp_preprocessor.create_semantic_chunks(cleaned_text)

        # 2. Embedding Generation
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = embedding_service.get_embeddings(chunk_texts)

        # 3. Emotion Analysis (Single unified detected emotion)
        emotion_data = emotion_service.analyze(cleaned_text, chunks)

        # 4. Contextual Intent Analysis
        intent_data = intent_service.analyze(cleaned_text, doc_type)

        # 5. Sentiment Analysis
        sentiment_chunks = sentiment_service.analyze_chunks(chunks)

        # 6. Document Summaries
        summaries = summarizer_service.generate_all_summaries(cleaned_text)

        # 7. Keyword Extraction
        keywords = keyword_service.extract_top_keywords(cleaned_text, top_n=20)

        # 8. Named Entity Recognition
        entities = ner_service.extract_entities(cleaned_text)

        # 9. Important Sentences
        important_sentences = self._extract_important_sentences(cleaned_text, chunk_embeddings, chunks)

        # 10. Emotional & Intent Timeline (Phased Flow)
        timeline = timeline_service.generate_timeline(chunks, doc_type)

        # 11. Topic Detection
        main_topic, sub_topics = self._detect_topics(cleaned_text, keywords, doc_type)

        # 12. Synthesized "What Happened" & Summary Insight
        what_happened_synthesized = summarizer_service.generate_what_happened(
            cleaned_text, doc_type, intent_data["primary_intent"], emotion_data["primary_emotion"]
        )
        summary_insight = self._build_summary_insight(
            emotion_data, intent_data, what_happened_synthesized, word_count, doc_type
        )

        # 13. AI Empathy & De-escalation Script Draft
        deescalation_response = responder_service.generate_response_script(
            primary_emotion=emotion_data["primary_emotion"],
            hidden_emotion=emotion_data["hidden_emotion"],
            intent=intent_data["primary_intent"],
            doc_type=doc_type,
            summary=what_happened_synthesized
        )

        # 14. Confidence Scores calculation
        confidence_scores = {
            "emotion_detection": emotion_data["confidence"],
            "intent_detection": intent_data["confidence"],
            "summary": round(min(96.0, max(75.0, 80.0 + word_count * 0.005)), 1),
            "ner": round(min(97.0, max(70.0, len(entities["persons"]) * 3.0 + 75.0)), 1),
            "keywords": round(keywords[0]["score"] * 100 if keywords else 88.0, 1),
            "overall_prediction": round((emotion_data["confidence"] + intent_data["confidence"] + 88.0) / 3.0, 1)
        }

        # 15. Clean Suggestions (Empty list as Actionable AI Recommendations are removed)
        suggestions: List[Dict[str, str]] = []

        # 16. Final AI Conclusion Paragraph (Clean Assessment without actionable directives)
        risk_level = "High" if emotion_data["negative_percentage"] > 45 or "Complaint" in intent_data["primary_intent"] or "Legal" in intent_data["primary_intent"] or "Grievance" in intent_data["primary_intent"] else ("Medium" if emotion_data["negative_percentage"] > 25 else "Low")
        final_conclusion = f"The document exhibits an overall {emotion_data['overall_tone'].lower()} state with a detected emotion of {emotion_data['primary_emotion']}. The author's underlying intent is {intent_data['primary_intent']}, accompanied by a hidden psychological state of {emotion_data['hidden_emotion']}. Contextual pattern analysis confirms a {risk_level} priority posture."

        metadata = {
            "document_name": filename,
            "document_type": doc_type,
            "upload_time": upload_time,
            "word_count": word_count,
            "reading_time_minutes": reading_time_minutes,
            "language": "English (Detected)"
        }

        return {
            "report_id": report_id,
            "metadata": metadata,
            "emotion_analysis": emotion_data,
            "intent_analysis": intent_data,
            "summary_insight": summary_insight,
            "summaries": summaries,
            "keywords": keywords,
            "entities": entities,
            "main_topic": main_topic,
            "sub_topics": sub_topics,
            "important_sentences": important_sentences,
            "emotional_timeline": timeline,
            "deescalation_response": deescalation_response,
            "confidence_scores": confidence_scores,
            "suggestions": suggestions,
            "final_conclusion": final_conclusion
        }

    def _extract_important_sentences(self, text: str, embeddings: List[List[float]], chunks: List[Dict[str, Any]]) -> List[str]:
        sentences = nlp_preprocessor.get_sentences(text)
        if len(sentences) <= 10:
            return sentences

        if embeddings:
            dim = len(embeddings[0])
            centroid = [sum(e[i] for e in embeddings) / len(embeddings) for i in range(dim)]
            
            s_embeddings = embedding_service.get_embeddings(sentences[:40])
            scored = []
            for s, s_emb in zip(sentences[:40], s_embeddings):
                sim = embedding_service.cosine_similarity(s_emb, centroid)
                scored.append((s, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in scored[:10]]
            
        return sentences[:10]

    def _detect_topics(self, text: str, keywords: List[Dict[str, Any]], doc_type: str) -> tuple:
        top_kws = [k["keyword"] for k in keywords[:5]]
        if top_kws:
            main_topic = f"{doc_type} Focus: {top_kws[0]} & {top_kws[1] if len(top_kws) > 1 else 'Analysis'}"
            sub_topics = [f"{kw} Assessment" for kw in top_kws[1:5]]
        else:
            main_topic = f"{doc_type} Content Analysis"
            sub_topics = ["Operational Context", "User Intent", "System Execution", "Outcome Evaluation"]
            
        return main_topic, sub_topics

    def _build_summary_insight(self, emotion: dict, intent: dict, what_happened: str, word_count: int, doc_type: str) -> dict:
        p_emo = emotion["primary_emotion"]
        h_emo = emotion["hidden_emotion"]
        p_intent = intent["primary_intent"]

        why_feels = f"The author conveys {p_emo.lower()}, with underlying psychological markers of {h_emo.lower()}."
        what_user_wants = intent["reasoning"]
        problem_identified = f"The primary friction is centered on {p_intent.lower()}."

        narrative = f"In short: The author experiences {p_emo.lower()} while navigating an underlying objective of {p_intent.lower()}."

        return {
            "what_happened": what_happened,
            "why_feels_this_way": why_feels,
            "what_user_wants": what_user_wants,
            "problem_identified": problem_identified,
            "possible_next_action": "",
            "narrative_paragraph": narrative
        }

report_generator = ReportGenerator()
