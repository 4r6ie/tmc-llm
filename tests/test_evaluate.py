from pathlib import Path

import json

import pytest


def load_processed_dataset(name: str = "train") -> list[dict]:
    """Load a processed JSONL dataset."""
    path = Path(f"data/processed/{name}.jsonl")
    assert path.exists(), f"Dataset not found at {path}"
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_user_messages(entries: list[dict]) -> list[str]:
    """Extract user message content from dataset entries."""
    messages = []
    for entry in entries:
        msgs = entry.get("messages", [])
        if len(msgs) > 1:
            messages.append(msgs[1]["content"])  # user message
    return messages


def extract_assistant_messages(entries: list[dict]) -> list[str]:
    """Extract assistant message content from dataset entries."""
    messages = []
    for entry in entries:
        msgs = entry.get("messages", [])
        if len(msgs) > 2:
            messages.append(msgs[2]["content"])  # assistant message
    return messages


SYSTEM_PROMPT = (
    "You are TMC-LM, an offline assistant for Trinidad Municipal College. "
    "Answer using only the provided official TMC knowledge. "
    "If the source does not contain the answer, say that the available TMC source does not contain it."
)


NEGATIVE_ANSWER = (
    "The available TMC source does not contain this information. "
    "Please contact the appropriate TMC office for assistance."
)


@pytest.fixture
def train_entries():
    return load_processed_dataset("train")


@pytest.fixture
def val_entries():
    return load_processed_dataset("validation")


@pytest.fixture
def test_entries():
    return load_processed_dataset("test")


class TestSystemPrompt:
    def test_system_prompt_string(self) -> None:
        assert SYSTEM_PROMPT is not None
        assert "Trinidad Municipal College" in SYSTEM_PROMPT


class TestNegativeAnswerFormat:
    def test_negative_answer_contains_key_phrases(self) -> None:
        assert "TMC source does not contain" in NEGATIVE_ANSWER


class TestTrainDatasetHasContent:
    def test_train_has_entries(self, train_entries: list[dict]) -> None:
        assert len(train_entries) > 0, "train.jsonl should contain entries"

    def test_train_has_system_prompt(self, train_entries: list[dict]) -> None:
        assert all(
            msg.get("role") == "system" for msg in train_entries[0].get("messages", [])[:1]
        ), "Entries should start with system prompt"

    def test_train_user_messages_exist(self, train_entries: list[dict]) -> None:
        messages = extract_user_messages(train_entries)
        assert len(messages) > 0, "Should have user messages"


class TestMissionVisionProgramsCoverage:
    """Verify that the dataset covers key TMC information categories."""

    VISION_KEYWORDS = {"vision", "Vision", "VISION", "purpose", "PURPOSE"}
    MISSION_KEYWORDS = {"mission", "Mission", "MISSION", "goal", "GOAL"}
    PROGRAMS_KEYWORDS = {
        "program",
        "Program",
        "PROGRAM",
        "course",
        "Course",
        "COURSE",
        "diploma",
        "Diploma",
    }
    HISTORY_KEYWORDS = {"history", "History", "HISTORY", "establish", "ESTABLISH"}

    def test_vision_coverage(self, train_entries: list[dict]) -> None:
        user_msgs = extract_user_messages(train_entries)
        has_vision = any(
            any(kw in content for kw in self.VISION_KEYWORDS) for content in user_msgs
        )
        assert has_vision, "Expected vision-related content in training data"

    def test_mission_coverage(self, train_entries: list[dict]) -> None:
        user_msgs = extract_user_messages(train_entries)
        has_mission = any(
            any(kw in content for kw in self.MISSION_KEYWORDS) for content in user_msgs
        )
        assert has_mission, "Expected mission-related content in training data"

    def test_programs_coverage(self, train_entries: list[dict]) -> None:
        user_msgs = extract_user_messages(train_entries)
        has_programs = any(
            any(kw in content for kw in self.PROGRAMS_KEYWORDS) for content in user_msgs
        )
        assert has_programs, "Expected program-related content in training data"

    def test_history_coverage(self, train_entries: list[dict]) -> None:
        user_msgs = extract_user_messages(train_entries)
        has_history = any(
            any(kw in content for kw in self.HISTORY_KEYWORDS) for content in user_msgs
        )
        assert has_history, "Expected history-related content in training data"


class TestNegativeQuestionsHandling:
    """Test that negative Q&A patterns are properly defined and can be matched."""

    NEGATIVE_QUESTIONS = [
        (
            "What is the tuition fee for international students at TMC?",
            f"{NEGATIVE_ANSWER} Please contact the Registrar's Office or Accounting Office for details.",
        ),
        (
            "What is the exact salary of a TMC professor?",
            f"{NEGATIVE_ANSWER} TMC follows the Salary Standardization Table for LGU employees. Please contact the HRMO for details.",
        ),
        ("How many computers does the TMC computer lab have?", f"{NEGATIVE_ANSWER} Please contact the IT Department for details."),
        (
            "What is the Wi-Fi password at TMC?",
            f"{NEGATIVE_ANSWER} Please ask at the IT Department or Administration Office for Wi-Fi access details.",
        ),
        (
            "Does TMC have a swimming pool?",
            f"{NEGATIVE_ANSWER} Please contact the Administration Office for information about campus facilities.",
        ),
        (
            "What is the current enrollment deadline for next semester?",
            f"{NEGATIVE_ANSWER} Please check the TMC website or contact the Registrar's Office for the latest enrollment schedule.",
        ),
        (
            "Who is the current Registrar of TMC?",
            f"{NEGATIVE_ANSWER} Please contact the Registrar's Office directly for the name of the current Registrar.",
        ),
        (
            "What programming languages are taught in the BSIT program?",
            f"{NEGATIVE_ANSWER} The program covers programming, networking, database management, and web development. Please contact the College of Computer Studies for specifics.",
        ),
        (
            "How much is the graduation fee at TMC?",
            f"{NEGATIVE_ANSWER} Please contact the Accounting Office for current fee schedules.",
        ),
        (
            "What is the student-to-computer ratio at TMC?",
            f"{NEGATIVE_ANSWER} Please contact the College of Computer Studies or IT Department for details.",
        ),
        (
            "Does TMC offer online classes?",
            f"{NEGATIVE_ANSWER} TMC provides morning, afternoon, and evening class schedules. Please contact the Registrar's Office for current class modalities.",
        ),
        (
            "What is the address of TMC?",
            f"{NEGATIVE_ANSWER} Please contact the Administration Office for the exact street address.",
        ),
        (
            "Does TMC have a basketball court?",
            f"{NEGATIVE_ANSWER} TMC has open grounds and sports facilities on campus. Please contact the Administration Office for details.",
        ),
        (
            "What is the process for filing a grade appeal at TMC?",
            f"{NEGATIVE_ANSWER} Grades are final once recorded and signed by the Department Head. Grade changes require approval from the Department Head and Academic Affairs. Please contact the Registrar's Office for guidance.",
        ),
        (
            "How many sections per class does TMC have?",
            f"{NEGATIVE_ANSWER} Please contact the Academic Affairs Office for details.",
        ),
    ]

    def test_negative_questions_defined(self) -> None:
        assert len(self.NEGATIVE_QUESTIONS) > 0, "Should have negative questions defined"

    def test_negative_questions_have_answers(self) -> None:
        for question, answer in self.NEGATIVE_QUESTIONS:
            assert question.strip(), f"Question should not be empty: {question}"
            assert answer.strip(), f"Answer should not be empty for: {question}"

    def test_negative_answers_contain_key_phrase(self) -> None:
        for _question, answer in self.NEGATIVE_QUESTIONS:
            assert "TMC source does not contain" in answer, (
                f"Answer should contain '{NEGATIVE_ANSWER}' key phrase: {answer}"
            )