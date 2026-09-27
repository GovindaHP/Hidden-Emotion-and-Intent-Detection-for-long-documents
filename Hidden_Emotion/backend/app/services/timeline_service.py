import re
from typing import List, Dict, Any
from app.services.emotion_service import emotion_service
from app.services.intent_service import intent_service
from app.services.sentiment_service import sentiment_service

class TimelineService:
    @staticmethod
    def generate_timeline(chunks: List[Dict[str, Any]], doc_type: str = "General") -> List[Dict[str, Any]]:
        full_text = " ".join([c["text"] for c in chunks])
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if len(s.split()) >= 3]

        if not sentences:
            sentences = [full_text]

        phase_names = [
            ("Beginning", "Opening & Initial Impression"),
            ("Middle", "Core Details & Specific Grievances"),
            ("Later", "Peak Tension & Impact"),
            ("Ending", "Conclusion & Final Demand")
        ]

        phases = []
        n_sentences = len(sentences)

        if n_sentences == 1:
            # Single sentence document: 1 phase is sufficient to avoid duplication
            phases.append(TimelineService._analyze_phase("Overview", "Full Document Context", sentences[0], doc_type, 0))
        elif n_sentences <= 4:
            # 2 to 4 sentences: map 1 sentence per phase
            for idx in range(min(n_sentences, 4)):
                p_name, p_label = phase_names[idx]
                phases.append(TimelineService._analyze_phase(p_name, p_label, sentences[idx], doc_type, idx))
        else:
            # > 4 sentences: group sentences evenly into 4 distinct chronological buckets
            step = n_sentences / 4.0
            for idx, (p_name, p_label) in enumerate(phase_names):
                start_i = int(idx * step)
                end_i = int((idx + 1) * step) if idx < 3 else n_sentences
                seg_text = " ".join(sentences[start_i:end_i])
                phases.append(TimelineService._analyze_phase(p_name, p_label, seg_text, doc_type, idx))

        return phases

    @staticmethod
    def _analyze_phase(phase_name: str, phase_label: str, text: str, doc_type: str, phase_index: int) -> Dict[str, Any]:
        # Analyze STRICTLY this phase's text (not the full document)
        emo_res = emotion_service.analyze(text, [{"index": 0, "text": text}])
        intent_res = intent_service.analyze(text, doc_type)
        senti_res = sentiment_service.score_text(text)
        
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.split()) >= 3]
        excerpt = sentences[0] if sentences else text[:120]
        if len(excerpt) > 120:
            excerpt = excerpt[:117] + "..."

        p_emo = emo_res["primary_emotion"]
        h_emo = emo_res["hidden_emotion"]
        p_intent = intent_res["primary_intent"]

        # Phase-specific contextual flow descriptions
        if senti_res["compound"] >= 0.15:
            emo_flow_desc = f"Exhibits a positive tone of {p_emo.lower()}, expressing satisfaction or favorable feedback."
        elif senti_res["compound"] <= -0.15:
            emo_flow_desc = f"Reflects a critical tone of {p_emo.lower()}, highlighting grievances or frustration."
        else:
            emo_flow_desc = f"Maintains a neutral, observational tone centered around {p_emo.lower()}."

        if p_intent != "No Intent Detected":
            intent_flow_desc = f"Focuses on {p_intent.lower()}, seeking a specific outcome or clarification."
        else:
            intent_flow_desc = "Provides general background context without making an explicit request."

        return {
            "phase": phase_name,
            "phase_label": phase_label,
            "dominant_emotion": p_emo,
            "hidden_emotion": h_emo,
            "emotion_flow_description": emo_flow_desc,
            "phase_intent": p_intent,
            "intent_flow_description": intent_flow_desc,
            "sentiment_score": senti_res["compound"],
            "positive_pct": emo_res["positive_percentage"],
            "neutral_pct": emo_res["neutral_percentage"],
            "negative_pct": emo_res["negative_percentage"],
            "excerpt": excerpt
        }

timeline_service = TimelineService()
