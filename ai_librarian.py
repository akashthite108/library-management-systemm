"""AI-assisted librarian workflows with explicit, verifiable library data."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol


SYSTEM_PROMPT = """\
You are an expert AI Librarian assistant for an academic institution.
Your role is to help with book cataloging, patron membership verification,
inventory search, and loan processing. Use only catalog and patron data
provided by trusted library systems; never invent availability, account
status, fees, or policy. Maintain a precise, helpful, and professional
academic tone.
"""

GENERATION_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_output_tokens": 512,
}

MODEL_NAME = os.environ.get("OPENAI_MODEL", "openai/gpt-6-luna")


class ChatCompletionsClient(Protocol):
    """The subset of the OpenAI client used by this module."""

    @property
    def chat(self) -> Any: ...


@dataclass(frozen=True)
class PatronIntent:
    title: str | None
    author: str | None
    action: str
    category: str | None


@dataclass(frozen=True)
class LoanDecision:
    approved: bool
    justification: str
    active_loan_count: int
    unpaid_fines: Decimal
    overdue_book_count: int

    def __str__(self) -> str:
        verdict = "APPROVED" if self.approved else "DENIED"
        return (
            f"{verdict}: {self.justification} "
            f"Active loans: {self.active_loan_count}. "
            f"Overdue books: {self.overdue_book_count}. "
            f"Unpaid fines: ${self.unpaid_fines:.2f}."
        )


def _get_client(client: ChatCompletionsClient | None) -> ChatCompletionsClient:
    if client is not None:
        return client

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY or pass an OpenAI-compatible client.")

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The openai package is required for model-backed librarian workflows."
        ) from error

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def parse_patron_intent(
    user_query: str,
    *,
    client: ChatCompletionsClient | None = None,
) -> PatronIntent:
    """Extract a patron request as validated structured data."""
    if not user_query.strip():
        raise ValueError("user_query must not be empty.")

    active_client = _get_client(client)
    response = active_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{SYSTEM_PROMPT}\n"
                    "Classify the patron's request and extract book metadata. "
                    "Return a JSON object with title, author, action, and category. "
                    "Use null for metadata not stated or reliably inferable. "
                    "The action must be one of search, borrow, return, renew, "
                    "membership, or unknown."
                ),
            },
            {
                "role": "user",
                "content": "Can I borrow Clean Code by Robert Martin?",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "title": "Clean Code",
                        "author": "Robert Martin",
                        "action": "borrow",
                        "category": "Software Engineering",
                    }
                ),
            },
            {
                "role": "user",
                "content": (
                    "Do you have Introduction to Algorithms in the "
                    "computer science section?"
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "title": "Introduction to Algorithms",
                        "author": "Thomas H. Cormen",
                        "action": "search",
                        "category": "Computer Science",
                    }
                ),
            },
            {"role": "user", "content": user_query},
        ],
        temperature=0.1,
        top_p=GENERATION_CONFIG["top_p"],
        max_completion_tokens=GENERATION_CONFIG["max_output_tokens"],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The model returned an empty intent response.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("The model returned invalid JSON for patron intent.") from error
    if not isinstance(data, dict):
        raise ValueError("The model's intent response must be a JSON object.")

    action = data.get("action")
    if action not in {
        "search",
        "borrow",
        "return",
        "renew",
        "membership",
        "unknown",
    }:
        raise ValueError(f"The model returned an unsupported action: {action!r}.")

    def optional_text(key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"The model's {key} value must be text or null.")
        return value.strip() or None

    return PatronIntent(
        title=optional_text("title"),
        author=optional_text("author"),
        action=action,
        category=optional_text("category"),
    )


def evaluate_loan_eligibility(
    member_id: str,
    book_title: str,
    *,
    active_loan_count: int,
    overdue_book_count: int,
    unpaid_fines: Decimal,
    max_active_loans: int = 3,
) -> LoanDecision:
    """Apply the stated loan-count and overdue checks to verified account data.

    Fines are reported but are not an approval condition because no fine
    threshold was specified. The caller must supply current values from the
    library's authoritative account system.
    """
    if not member_id.strip():
        raise ValueError("member_id must not be empty.")
    if not book_title.strip():
        raise ValueError("book_title must not be empty.")
    if active_loan_count < 0 or overdue_book_count < 0:
        raise ValueError("Loan counts cannot be negative.")
    if max_active_loans < 1:
        raise ValueError("max_active_loans must be at least 1.")
    if unpaid_fines < 0:
        raise ValueError("unpaid_fines cannot be negative.")

    reasons: list[str] = []
    if overdue_book_count:
        reasons.append(f"member {member_id} has overdue books")
    if active_loan_count >= max_active_loans:
        reasons.append(
            f"member {member_id} has reached the limit of "
            f"{max_active_loans} active loans"
        )

    approved = not reasons
    justification = (
        f"Member {member_id} may borrow '{book_title}'."
        if approved
        else f"Member {member_id} may not borrow '{book_title}': "
        + "; ".join(reasons)
        + "."
    )
    return LoanDecision(
        approved=approved,
        justification=justification,
        active_loan_count=active_loan_count,
        unpaid_fines=unpaid_fines,
        overdue_book_count=overdue_book_count,
    )


def calculate_overdue_fines(
    overdue_days_by_book: Sequence[int],
    *,
    daily_rate: Decimal = Decimal("0.50"),
) -> Decimal:
    """Calculate accrued overdue charges before subtracting any payments."""
    if daily_rate < 0:
        raise ValueError("daily_rate cannot be negative.")
    if any(
        not isinstance(days, int) or isinstance(days, bool) or days < 0
        for days in overdue_days_by_book
    ):
        raise ValueError("Overdue days must be non-negative integers.")
    return sum(
        (daily_rate * days for days in overdue_days_by_book),
        start=Decimal("0.00"),
    )


def inter_library_loan_workflow(
    user_request: str,
    *,
    local_catalog_search: Callable[[PatronIntent], Mapping[str, Any]],
    partner_network_search: Callable[[PatronIntent], Mapping[str, Any]],
    client: ChatCompletionsClient | None = None,
) -> str:
    """Search real catalog adapters and draft a transfer notice from their data."""
    intent = parse_patron_intent(user_request, client=client)
    if not intent.title:
        raise ValueError("The patron request did not identify a book title.")

    local_result = local_catalog_search(intent)
    if not isinstance(local_result.get("available"), bool):
        raise ValueError("Local catalog results must include a boolean 'available'.")
    if local_result["available"]:
        return "The book is available in the local library collection."

    partner_result = partner_network_search(intent)
    if not isinstance(partner_result.get("available"), bool):
        raise ValueError("Partner network results must include a boolean 'available'.")
    if not partner_result["available"]:
        return (
            f"'{intent.title}' is not currently available in the local collection "
            "or the partner network."
        )

    eta = partner_result.get("eta")
    if not isinstance(eta, str) or not eta.strip():
        raise ValueError("Available partner copies must include a verified ETA.")
    author = f" by {intent.author}" if intent.author else ""
    return (
        f"Subject: Inter-library loan request for {intent.title}\n\n"
        "Dear Patron,\n\n"
        f"'{intent.title}'{author} is currently unavailable in our local "
        "collection. A partner library has confirmed availability, with an "
        f"estimated transfer time of {eta.strip()}. Please reply to confirm "
        "whether you would like us to request the transfer.\n\n"
        "Best regards,\nAcademic Library"
    )


@dataclass(frozen=True)
class ReActTool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[[Mapping[str, Any]], Any]


class AILibrarianReActAgent:
    """Run a bounded tool-calling loop using real, caller-provided tools."""

    def __init__(
        self,
        tools: Sequence[ReActTool],
        *,
        client: ChatCompletionsClient | None = None,
        max_tool_rounds: int = 5,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1.")
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("Tool names must be unique.")
        self.tools = {tool.name: tool for tool in tools}
        self.client = client
        self.max_tool_rounds = max_tool_rounds

    def run(self, user_query: str) -> str:
        if not user_query.strip():
            raise ValueError("user_query must not be empty.")

        active_client = _get_client(self.client)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                },
            }
            for tool in self.tools.values()
        ]

        for round_index in range(self.max_tool_rounds + 1):
            response = active_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tool_schemas or None,
                temperature=GENERATION_CONFIG["temperature"],
                top_p=GENERATION_CONFIG["top_p"],
                max_completion_tokens=GENERATION_CONFIG["max_output_tokens"],
            )
            message = response.choices[0].message
            if not message.tool_calls:
                if not message.content:
                    raise ValueError("The model returned an empty final response.")
                return message.content

            if round_index == self.max_tool_rounds:
                break

            messages.append(message.model_dump(exclude_none=True))
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool = self.tools.get(tool_name)
                if tool is None:
                    raise ValueError(f"The model requested an unknown tool: {tool_name}.")
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"The model returned invalid arguments for tool {tool_name}."
                    ) from error
                if not isinstance(arguments, dict):
                    raise ValueError(
                        f"Arguments for tool {tool_name} must be a JSON object."
                    )

                result = tool.handler(arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str),
                    }
                )

        raise RuntimeError(
            f"The librarian agent exceeded {self.max_tool_rounds} tool rounds."
        )
