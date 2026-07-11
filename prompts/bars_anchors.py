from prompts.persona import MOCK_TYPE_TECHNICAL, MOCK_TYPE_BEHAVIORAL, MOCK_TYPE_CODING

UNIVERSAL_ANCHORS = {
    "communication": [
        "Responses are incoherent, off-topic, or monosyllabic; cannot convey basic ideas",
        "Conveys simple ideas but with frequent unclear phrasing, long tangents, or confusion",
        "Communicates main points clearly; occasional ambiguity but generally understandable",
        "Articulate and well-organized; adjusts explanation depth to the question's complexity",
        "Exceptionally clear, concise, and persuasive; adapts language naturally to context",
    ],
    "problemSolving": [
        "Cannot identify the problem or suggests completely irrelevant approaches",
        "Recognizes the problem but proposes a flawed or incomplete approach with major gaps",
        "Identifies the core problem and proposes a reasonable approach; misses some edge cases",
        "Breaks down the problem systematically; considers trade-offs and alternative approaches",
        "Exceptional decomposition; anticipates edge cases, weighs trade-offs, and justifies choices",
    ],
    "technical": [
        "No understanding of relevant concepts; statements are factually incorrect",
        "Basic awareness of concepts but with significant misunderstandings or inaccuracies",
        "Understands core concepts correctly; some gaps in depth or related topics",
        "Strong grasp of concepts with accurate details; connects ideas across topics",
        "Deep, nuanced understanding; explains underlying principles and real-world implications",
    ],
    "clarityOfExplanation": [
        "Explanation is confusing, disorganized, or impossible to follow",
        "Attempts to explain but jumps between ideas without logical flow",
        "Explains ideas in a followable order; some parts could be clearer",
        "Well-structured explanation with logical progression and appropriate detail",
        "Crystal-clear explanation; builds understanding step-by-step with perfect pacing",
    ],
    "structuredThinking": [
        "No visible structure; response is a stream of disconnected thoughts",
        "Minimal structure; some points are made but without clear organization",
        "Shows a basic framework for thinking; follows a recognizable pattern",
        "Systematically organizes thoughts; uses a clear methodology or framework",
        "Highly disciplined thinking; structures the response optimally for the question type",
    ],
}

TECHNICAL_ANCHORS = {
    "communication": [
        "Cannot articulate technical concepts; responses are incoherent or irrelevant",
        "Conveys basic technical ideas but with imprecise terminology and unclear explanations",
        "Explains technical concepts correctly; uses appropriate terminology most of the time",
        "Communicates complex technical ideas clearly; uses precise terminology and good analogies",
        "Exceptional technical communication; explains complex systems with precision and clarity",
    ],
    "problemSolving": [
        "Cannot identify the technical problem or proposes entirely wrong approaches",
        "Recognizes the problem but misses critical technical constraints or requirements",
        "Proposes a workable solution; considers basic requirements but misses some edge cases",
        "Systematic approach; considers scalability, performance, and failure modes",
        "Exceptional; evaluates multiple approaches, justifies choices with technical reasoning",
    ],
    "technical": [
        "No understanding of the relevant technology, language, or system",
        "Surface-level knowledge with significant conceptual errors or confusion",
        "Solid understanding of core concepts; some gaps in advanced topics or internals",
        "Deep knowledge; understands implementation details, trade-offs, and best practices",
        "Expert-level understanding; explains internals, edge cases, and architectural implications",
    ],
    "clarityOfExplanation": [
        "Technical explanation is incomprehensible or completely disorganized",
        "Attempts to explain but mixes up concepts or skips critical steps",
        "Explains the technical approach in a logical order; some steps could be clearer",
        "Clear, methodical explanation that a peer engineer could follow and implement",
        "Textbook-quality explanation; builds from fundamentals to solution with perfect clarity",
    ],
    "structuredThinking": [
        "No structured approach; jumps to conclusions without analysis",
        "Some structure but skips important steps like requirements gathering or constraint analysis",
        "Follows a recognizable approach; considers inputs, outputs, and basic constraints",
        "Methodical; breaks the problem into components, addresses each systematically",
        "Rigorous engineering thinking; requirements, design, trade-offs, validation, and iteration",
    ],
}

BEHAVIORAL_ANCHORS = {
    "communication": [
        "Cannot describe past experiences coherently; responses are vague or rambling",
        "Tells a basic story but lacks structure; hard to follow the narrative",
        "Communicates experiences clearly; provides context and basic details",
        "Tells a well-structured story with clear context, actions, and outcomes",
        "Compelling storytelling; engages the listener with vivid, specific, and relevant details",
    ],
    "problemSolving": [
        "Cannot describe how they approached challenges; no evidence of problem-solving",
        "Describes reacting to problems but without intentional strategy or analysis",
        "Shows basic problem-solving; describes identifying issues and taking reasonable action",
        "Demonstrates strategic thinking; describes analyzing situations and choosing approaches",
        "Shows exceptional judgment; describes navigating ambiguity, weighing trade-offs, and driving outcomes",
    ],
    "technical": [
        "Cannot explain the technical context of their experience at all",
        "Mentions technology but cannot explain their role or the technical decisions made",
        "Explains the technical context adequately; describes their specific contributions",
        "Clearly articulates technical decisions, trade-offs, and their reasoning",
        "Deep technical insight into their decisions; explains why alternatives were rejected",
    ],
    "clarityOfExplanation": [
        "Story is confusing; cannot distinguish situation, action, and result",
        "Provides some context but the narrative jumps around; key details missing",
        "Tells the story in a logical order; situation, action, and result are identifiable",
        "Well-structured STAR narrative; each element is clear and proportionally detailed",
        "Masterful narrative structure; builds tension, highlights decisions, and quantifies impact",
    ],
    "structuredThinking": [
        "No recognizable structure; response is a stream of disconnected anecdotes",
        "Some structure but mixes multiple stories or loses focus on the question",
        "Uses a basic structure (roughly STAR); stays on topic with minor digressions",
        "Clear STAR structure; systematically covers situation, task, action, and result",
        "Exceptional structure; layers multiple insights, connects to broader lessons learned",
    ],
}

CODING_ANCHORS = {
    "communication": [
        "Cannot explain their approach to the problem; responses are incoherent",
        "Explains some steps but skips critical reasoning; hard to follow the thought process",
        "Communicates their approach clearly; explains what they would do and why",
        "Thinks aloud effectively; explains each step, decision point, and alternative considered",
        "Exceptional verbal problem-solving; narrates their thought process like a pair-programming session",
    ],
    "problemSolving": [
        "Cannot break down the coding problem; no approach or strategy evident",
        "Identifies a brute-force approach but cannot optimize or handle edge cases",
        "Proposes a reasonable algorithm; considers basic cases but misses some edge cases",
        "Strong algorithmic thinking; considers time/space complexity and handles edge cases",
        "Optimal solution with thorough edge case analysis; compares multiple approaches with complexity trade-offs",
    ],
    "technical": [
        "No understanding of relevant data structures, algorithms, or language features",
        "Basic awareness of common data structures but cannot apply them to the problem",
        "Knows appropriate data structures and algorithms; can explain their properties",
        "Deep understanding; selects optimal data structures and explains time/space trade-offs",
        "Expert-level; discusses amortized complexity, cache behavior, or language-specific optimizations",
    ],
    "clarityOfExplanation": [
        "Cannot explain their solution approach; jumps to conclusions without reasoning",
        "Explains parts of the solution but skips key steps or assumptions",
        "Walks through the solution step by step; explains the logic behind each step",
        "Clear, methodical walkthrough; a listener could implement the solution from the explanation",
        "Explains the solution at multiple levels of abstraction; from high-level strategy to implementation details",
    ],
    "structuredThinking": [
        "No structure; jumps directly to implementation details without planning",
        "Some planning but skips constraint analysis or input validation",
        "Follows a basic structure: understand problem, plan approach, walk through solution",
        "Systematic: clarifies constraints, considers approaches, selects and justifies one, traces through examples",
        "Rigorous: requirements, approach comparison, pseudocode walkthrough, complexity analysis, testing strategy",
    ],
}

_ANCHOR_SETS = {
    MOCK_TYPE_TECHNICAL: TECHNICAL_ANCHORS,
    MOCK_TYPE_BEHAVIORAL: BEHAVIORAL_ANCHORS,
    MOCK_TYPE_CODING: CODING_ANCHORS,
}

_DIMENSION_LABELS = {
    "communication": "Communication",
    "problemSolving": "Problem Solving",
    "technical": "Technical Knowledge",
    "clarityOfExplanation": "Clarity of Explanation",
    "structuredThinking": "Structured Thinking",
}


def get_bars_anchors(mock_type: str, active_dimensions: list[str] | None = None) -> str:
    anchors = _ANCHOR_SETS.get(mock_type, UNIVERSAL_ANCHORS)
    lines = []
    for key, label in _DIMENSION_LABELS.items():
        if active_dimensions is not None and key not in active_dimensions:
            continue
        descriptions = anchors[key]
        lines.append(f"\n{label} ({key}):")
        for level, desc in enumerate(descriptions, start=1):
            lines.append(f"  {level} — {desc}")
    return "\n".join(lines)
