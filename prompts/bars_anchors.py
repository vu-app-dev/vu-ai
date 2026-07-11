from prompts.persona import MOCK_TYPE_TECHNICAL, MOCK_TYPE_BEHAVIORAL, MOCK_TYPE_CODING

UNIVERSAL_ANCHORS = {
    "communication": [
        "Incoherent, off-topic, or monosyllabic",
        "Basic ideas conveyed but unclear phrasing, tangents",
        "Main points clear; minor ambiguity",
        "Articulate, well-organized, appropriate depth",
        "Exceptionally clear, concise, and persuasive",
    ],
    "problemSolving": [
        "Cannot identify problem or suggests irrelevant approaches",
        "Recognizes problem but flawed/incomplete approach",
        "Reasonable approach; misses some edge cases",
        "Systematic breakdown; considers trade-offs and alternatives",
        "Exceptional decomposition; anticipates edge cases, justifies choices",
    ],
    "technical": [
        "No understanding; factually incorrect statements",
        "Basic awareness but significant misunderstandings",
        "Core concepts correct; gaps in depth",
        "Strong grasp with accurate details; connects ideas",
        "Deep nuanced understanding; explains principles and implications",
    ],
    "clarityOfExplanation": [
        "Confusing, disorganized, impossible to follow",
        "Jumps between ideas without logical flow",
        "Followable order; some parts could be clearer",
        "Well-structured with logical progression",
        "Crystal-clear, step-by-step with perfect pacing",
    ],
    "structuredThinking": [
        "No structure; disconnected thoughts",
        "Minimal structure; points without organization",
        "Basic framework; recognizable pattern",
        "Systematic; clear methodology or framework",
        "Highly disciplined; optimal structure for the question",
    ],
}

TECHNICAL_ANCHORS = {
    "communication": [
        "Cannot articulate technical concepts",
        "Basic technical ideas but imprecise terminology",
        "Correct concepts; appropriate terminology",
        "Complex ideas clearly with precise terms and analogies",
        "Exceptional technical communication; precision and clarity",
    ],
    "problemSolving": [
        "Cannot identify technical problem or proposes wrong approaches",
        "Recognizes problem but misses critical constraints",
        "Workable solution; basic requirements but misses edge cases",
        "Systematic; considers scalability, performance, failure modes",
        "Evaluates multiple approaches; justifies with technical reasoning",
    ],
    "technical": [
        "No understanding of relevant technology",
        "Surface-level knowledge with conceptual errors",
        "Solid core concepts; gaps in advanced topics",
        "Deep knowledge; implementation details, trade-offs, best practices",
        "Expert-level; internals, edge cases, architectural implications",
    ],
    "clarityOfExplanation": [
        "Incomprehensible or disorganized technical explanation",
        "Mixes up concepts or skips critical steps",
        "Logical order; some steps could be clearer",
        "Clear, methodical; a peer could follow and implement",
        "Textbook-quality; fundamentals to solution with perfect clarity",
    ],
    "structuredThinking": [
        "No structured approach; jumps to conclusions",
        "Some structure but skips requirements or constraints",
        "Recognizable approach; considers inputs, outputs, constraints",
        "Methodical; breaks into components, addresses systematically",
        "Rigorous; requirements, design, trade-offs, validation, iteration",
    ],
}

BEHAVIORAL_ANCHORS = {
    "communication": [
        "Cannot describe experiences coherently; vague or rambling",
        "Basic story but lacks structure; hard to follow",
        "Clear experiences with context and basic details",
        "Well-structured story with clear context, actions, outcomes",
        "Compelling storytelling; vivid, specific, relevant details",
    ],
    "problemSolving": [
        "No evidence of problem-solving in experiences",
        "Describes reacting without strategy or analysis",
        "Basic problem-solving; identifies issues, takes reasonable action",
        "Strategic thinking; analyzes situations, chooses approaches",
        "Exceptional judgment; navigates ambiguity, drives outcomes",
    ],
    "technical": [
        "Cannot explain technical context at all",
        "Mentions technology but cannot explain role or decisions",
        "Adequate technical context; describes contributions",
        "Articulates technical decisions, trade-offs, reasoning",
        "Deep insight; explains why alternatives were rejected",
    ],
    "clarityOfExplanation": [
        "Confusing; cannot distinguish situation, action, result",
        "Some context but jumps around; key details missing",
        "Logical order; situation, action, result identifiable",
        "Well-structured STAR; clear and proportionally detailed",
        "Masterful narrative; builds tension, quantifies impact",
    ],
    "structuredThinking": [
        "No structure; disconnected anecdotes",
        "Some structure but mixes stories or loses focus",
        "Basic STAR structure; stays on topic",
        "Clear STAR; covers situation, task, action, result",
        "Exceptional; layers insights, connects to broader lessons",
    ],
}

CODING_ANCHORS = {
    "communication": [
        "Cannot explain approach; incoherent responses",
        "Some steps but skips reasoning; hard to follow",
        "Clear approach; explains what and why",
        "Thinks aloud effectively; explains decisions and alternatives",
        "Exceptional verbal problem-solving; pair-programming quality",
    ],
    "problemSolving": [
        "Cannot break down the problem; no strategy",
        "Brute-force only; cannot optimize or handle edge cases",
        "Reasonable algorithm; basic cases but misses edges",
        "Strong algorithmic thinking; time/space complexity, edge cases",
        "Optimal solution; thorough edge analysis, complexity trade-offs",
    ],
    "technical": [
        "No understanding of data structures/algorithms",
        "Basic awareness but cannot apply to problem",
        "Appropriate structures/algorithms; explains properties",
        "Optimal selections; explains time/space trade-offs",
        "Expert; amortized complexity, cache behavior, optimizations",
    ],
    "clarityOfExplanation": [
        "Cannot explain solution; jumps to conclusions",
        "Explains parts but skips key steps or assumptions",
        "Step-by-step walkthrough; logic behind each step",
        "Methodical; listener could implement from explanation",
        "Multiple abstraction levels; strategy to implementation details",
    ],
    "structuredThinking": [
        "No structure; jumps to implementation without planning",
        "Some planning but skips constraints or validation",
        "Basic: understand problem, plan approach, walk through",
        "Systematic: clarify constraints, select+justify approach, trace examples",
        "Rigorous: requirements, comparison, walkthrough, complexity, testing",
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
