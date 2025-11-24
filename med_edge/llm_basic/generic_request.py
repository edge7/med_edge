import instructor
from typing import Literal
from pydantic import BaseModel, Field, field_validator

from med_edge.llm_basic.prompts import MEDICAL_BENCHMARK_SYSTEM_PROMPT


def create_mcq_response_model(valid_options: list[str]):
    """
    Create a dynamic Pydantic model for MCQ responses based on available options.

    Args:
        valid_options: List of valid option letters (e.g., ["a", "b", "c"])

    Returns:
        A Pydantic model class with validation for the specific options
    """
    # Create a Literal type with only the valid options
    options_literal = Literal[tuple(valid_options)]

    class MedicalMCQResponse(BaseModel):
        """Response model for medical multiple choice questions"""
        answer: options_literal = Field(
            description=f"The selected answer choice ({', '.join(valid_options)}). Must be exactly one of these lowercase letters."
        )

        @field_validator("answer")
        @classmethod
        def validate_answer(cls, v: str) -> str:
            """Ensure answer is lowercase and one of the valid choices"""
            v = v.lower()
            if v not in valid_options:
                raise ValueError(f"Answer must be one of {', '.join(valid_options)}. Got: {v}")
            return v

    return MedicalMCQResponse


def get_single_answer_benchmark(model, question: str, options: dict[str, str], verbose: bool = False):
    """
    Get a single answer for a medical benchmark question.

    Args:
        model: The model identifier (e.g., "openai/gpt-4")
        question: The medical question text
        options: Dictionary mapping choice letters to option text (e.g., {"a": "...", "b": "...", ...})
        verbose: If True, prints the exact messages being sent (for research verification)

    Returns:
        dict: {"answer": str}
    """
    client = instructor.from_provider(model)

    # Get valid options and create dynamic response model
    valid_options = sorted(options.keys())  # e.g., ['a', 'b', 'c', 'd']
    MedicalMCQResponse = create_mcq_response_model(valid_options)

    # Format options for the prompt
    options_text = "\n".join([f"{key.upper()}. {value}" for key, value in options.items()])
    user_prompt = f"{question}\n\nOptions:\n{options_text}"

    messages = [
        {"role": "system", "content": MEDICAL_BENCHMARK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    if verbose:
        print("\n" + "="*60)
        print("EXACT MESSAGES BEING SENT (instructor mode)")
        print("="*60)
        print(f"\nModel: {model}")
        print(f"\nMessages:")
        for i, msg in enumerate(messages, 1):
            print(f"\n--- Message {i} ({msg['role']}) ---")
            print(msg['content'])
        print("="*60 + "\n")

    (completion, raw) = client.chat.completions.create_with_completion(
        model=model.split("/")[-1],
        messages=messages,
        response_model=MedicalMCQResponse,
        max_retries=3,
    )

    return {"answer": completion.answer}