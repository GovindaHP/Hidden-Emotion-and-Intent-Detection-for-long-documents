import pytest
from app.services.nlp_preprocessor import nlp_preprocessor
from app.services.emotion_service import emotion_service
from app.services.intent_service import intent_service
from app.services.report_generator import report_generator
from app.services.responder_service import responder_service

def test_nlp_preprocessor_chunking():
    sample_text = "This is sentence one. " * 50 + "This is sentence two. " * 50
    chunks = nlp_preprocessor.create_semantic_chunks(sample_text, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    assert "text" in chunks[0]
    assert "word_count" in chunks[0]

def test_emotion_service():
    sample_text = "I am so frustrated and outrageously angry because the system failed completely. I hate this service!"
    chunks = nlp_preprocessor.create_semantic_chunks(sample_text)
    res = emotion_service.analyze(sample_text, chunks)
    
    assert "primary_emotion" in res
    assert "hidden_emotion" in res
    assert "positive_percentage" in res
    assert "negative_percentage" in res
    assert res["negative_percentage"] > 20.0

def test_intent_service_technical():
    sample_text = "I need technical guidance to resolve a recurring error code and system crash bug in our configuration."
    res = intent_service.analyze(sample_text, doc_type="General")
    
    assert "Technical" in res["primary_intent"]
    assert "confidence" in res
    assert "reasoning" in res

def test_intent_service_long_range_refund():
    text = """
    I bought this gadget with high expectations. When problems occurred, I contacted customer support and performed multiple troubleshooting steps.
    They asked me to repeat steps I had already attempted. After extensive back and forth, the issues persist.
    I would rather return the item through the proper procedure and have the amount I paid handled according to the applicable policy.
    I believe returning the product and receiving the appropriate refund is the most practical way to achieve that.
    """
    res = intent_service.analyze(text, doc_type="General")
    assert "Return" in res["primary_intent"] or "Refund" in res["primary_intent"]

def test_intent_service_college_grievance():
    text = """
    The person I know goes to this college. There are many rules: no permission to go out without permission.
    Very strict and irritating teachers, those who teach computer science most don't know how to code the basics.
    Management keeps college on Saturday for extra class where teachers don't teach.
    Management fee is high and there is no good management. Please take the opinion of students.
    """
    res = intent_service.analyze(text, doc_type="Customer Feedback")
    assert "Academic" in res["primary_intent"] or "Grievance" in res["primary_intent"]

def test_calm_not_positive_emotion():
    text = """
    I am extremely disappointed and exhausted with repeated failures. Support gave useless replies.
    However, I am willing to cooperate with the return procedure and I hope there are no further misunderstandings.
    """
    chunks = nlp_preprocessor.create_semantic_chunks(text)
    res = emotion_service.analyze(text, chunks)
    assert res["primary_emotion"] != "Satisfaction & Optimism"
    assert ("Disappointment" in res["hidden_emotion"] or "Frustration" in res["hidden_emotion"] or "Determination" in res["hidden_emotion"])

def test_deescalation_responder():
    resp = responder_service.generate_response_script(
        primary_emotion="Frustration & Anger",
        hidden_emotion="Repressed Frustration",
        intent="Product Return & Refund Request",
        doc_type="Complaint Letter",
        summary="The user experienced product defects."
    )
    assert "subject" in resp
    assert "response_script" in resp
    assert "Official Response" in resp["subject"]

def test_report_generator():
    long_doc = """
    Dear Customer Support,
    I am writing to express my severe frustration regarding order #99281. The delivered item was completely broken upon arrival.
    I attempted to contact your helpline four times without receiving any response. This level of negligence is unacceptable.
    I demand an immediate refund or replacement product within 48 hours. If this issue is not resolved, I will be forced to escalate this matter to legal counsel and post public reviews.
    I look forward to your prompt response.
    Sincerely,
    John Smith
    """
    report = report_generator.generate(long_doc, filename="Complaint_Letter.txt", doc_type="Complaint Letter")
    
    assert report["report_id"].startswith("REP-")
    assert report["metadata"]["document_name"] == "Complaint_Letter.txt"
    assert report["emotion_analysis"]["primary_emotion"] != ""
    assert ("Refund" in report["intent_analysis"]["primary_intent"] or "Complaint" in report["intent_analysis"]["primary_intent"])
    assert len(report["keywords"]) > 0
    assert len(report["emotional_timeline"]) == 4
    assert report["final_conclusion"] != ""
    assert "deescalation_response" in report
    # Actionable AI recommendations list should be empty
    assert len(report["suggestions"]) == 0
