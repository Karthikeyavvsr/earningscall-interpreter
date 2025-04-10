from transformers import pipeline

# Load question-answering pipeline using DistilBERT model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def answer_question(context, question):
    """
    Uses a pre-trained model to answer a question given a context.

    Args:
        context (str): The transcript or report text.
        question (str): The user's query.

    Returns:
        tuple: (answer (str), confidence_score (float))
    """
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer'], result['score']
    except Exception as e:
        return f"Error: {str(e)}", 0.0
