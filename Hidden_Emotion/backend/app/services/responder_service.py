from typing import Dict, Any

class ResponderService:
    def generate_response_script(
        self, 
        primary_emotion: str, 
        hidden_emotion: str, 
        intent: str, 
        doc_type: str, 
        summary: str
    ) -> Dict[str, str]:
        """
        Generates an empathetic AI de-escalation response script tailored to 
        the author's explicit primary emotion and implicit hidden emotion.
        """
        # Subject line generation
        if "Complaint" in intent or "Complaint" in doc_type:
            subject = "Official Response & Priority Resolution Plan regarding your recent document"
            salutation = "Dear Valued Customer,"
            opening = f"Thank you for sharing your feedback with us. We understand from your document that {summary.lower().rstrip('.')}."
        elif "Technical" in intent:
            subject = "Technical Support Escalation & Immediate Action Plan"
            salutation = "Dear User,"
            opening = f"Thank you for reaching out regarding your technical issue. We recognize that {summary.lower().rstrip('.')}."
        elif "Medical" in intent:
            subject = "Care Guidance & Follow-up Support Information"
            salutation = "Dear Patient / Member,"
            opening = f"We have received your health inquiry. We note that {summary.lower().rstrip('.')}."
        elif "Legal" in intent:
            subject = "Acknowledgment of Legal Communication - Formal Reference"
            salutation = "Dear Sir / Madam,"
            opening = f"We confirm receipt of your formal legal communication regarding: {summary.rstrip('.')}."
        else:
            subject = "Response & Follow-up Update regarding your submission"
            salutation = "Dear Correspondent,"
            opening = f"Thank you for reaching out to us. We have thoroughly reviewed your document regarding {summary.lower().rstrip('.')}."

        # Empathy & De-escalation Body customized to emotions
        if "Frustration" in primary_emotion or "Anger" in primary_emotion:
            empathy_block = (
                "We sincerely regret the frustration and difficulty this situation has caused. "
                "Your experience is extremely important to us, and we are committed to making things right immediately."
            )
        elif "Disappointment" in primary_emotion or "Sadness" in primary_emotion:
            empathy_block = (
                "We deeply appreciate you sharing your experience with us. "
                "We recognize how disappointing this situation has been for you, and we want to ensure your concerns are addressed with full care."
            )
        elif "Anxiety" in primary_emotion or "Fear" in primary_emotion:
            empathy_block = (
                "We understand that this situation creates understandable worry and uncertainty. "
                "Please rest assured that our team is taking immediate steps to provide complete clarity and reassurance."
            )
        else:
            empathy_block = (
                "We appreciate the detailed information you have provided. "
                "Our objective is to ensure your inquiry receives the appropriate attention and resolution."
            )

        # Addressing Hidden Emotion Cues
        if "Repressed Frustration" in hidden_emotion or "Civility" in hidden_emotion:
            hidden_block = "We also want to acknowledge the ongoing friction behind this matter, even where expressed politely. We are addressing the root cause directly."
        elif "Masked Panic" in hidden_emotion or "Anxiety" in hidden_emotion:
            hidden_block = "We understand the time-sensitive urgency here and have prioritized your file for expedited review."
        elif "Hidden Optimism" in hidden_emotion:
            hidden_block = "We share your hope for a positive resolution and are working toward a swift, constructive outcome."
        else:
            hidden_block = "We are treating your communication with highest priority to deliver a seamless resolution."

        # Closing Action Block
        closing = (
            "Our team is actively reviewing your file. You will receive a direct follow-up update within 24 hours. "
            "If you have additional urgent context, please reply directly to this communication.\n\n"
            "Sincerely,\n"
            "Executive Resolution Team"
        )

        full_script = f"{salutation}\n\n{opening}\n\n{empathy_block} {hidden_block}\n\n{closing}"

        return {
            "subject": subject,
            "response_script": full_script,
            "target_intent": intent,
            "primary_emotion_addressed": primary_emotion,
            "hidden_emotion_addressed": hidden_emotion
        }

responder_service = ResponderService()
