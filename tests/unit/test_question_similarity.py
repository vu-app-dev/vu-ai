from services.interview.question_similarity import is_similar_question, question_similarity


def test_exact_duplicate_questions_are_similar():
    assert is_similar_question(
        "Explain how useEffect cleanup works.",
        "Explain how useEffect cleanup works!",
    )


def test_paraphrased_debugging_questions_are_similar():
    assert is_similar_question(
        "How do you approach debugging a complex issue in production?",
        "How would you debug a production problem?",
    )


def test_shared_broad_topic_can_still_be_different():
    assert not is_similar_question(
        "Explain how React hooks manage state in functional components.",
        "How would you optimize a slow React rendering path?",
    )


def test_similarity_scores_expose_combined_score():
    scores = question_similarity(
        "Compare REST and GraphQL for API design.",
        "What trade-offs matter when choosing REST versus GraphQL?",
    )
    assert scores.combined > 0
    assert 0 <= scores.token_jaccard <= 1
