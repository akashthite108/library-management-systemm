import json
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

from modules.library_management.ai_librarian import (
    GENERATION_CONFIG,
    AILibrarianReActAgent,
    ReActTool,
    calculate_overdue_fines,
    evaluate_loan_eligibility,
    inter_library_loan_workflow,
    parse_patron_intent,
)


def fake_client(*messages):
    create = Mock(
        side_effect=[
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=message, tool_calls=None)
                    )
                ]
            )
            for message in messages
        ]
    )
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


class AILibrarianTests(unittest.TestCase):
    def test_generation_config_has_expected_values(self):
        self.assertEqual(
            GENERATION_CONFIG,
            {"temperature": 0.2, "top_p": 0.95, "max_output_tokens": 512},
        )

    def test_parse_patron_intent_validates_structured_response(self):
        client = fake_client(
            json.dumps(
                {
                    "title": "Clean Code",
                    "author": "Robert Martin",
                    "action": "borrow",
                    "category": "Software Engineering",
                }
            )
        )

        intent = parse_patron_intent("Can I borrow Clean Code?", client=client)

        self.assertEqual(intent.title, "Clean Code")
        self.assertEqual(intent.action, "borrow")

    def test_eligibility_uses_verified_inputs_and_reports_fines(self):
        decision = evaluate_loan_eligibility(
            "M-123",
            "Clean Code",
            active_loan_count=2,
            overdue_book_count=0,
            unpaid_fines=Decimal("2.50"),
        )

        self.assertTrue(decision.approved)
        self.assertIn("Unpaid fines: $2.50", str(decision))

    def test_overdue_books_deny_eligibility(self):
        decision = evaluate_loan_eligibility(
            "M-123",
            "Clean Code",
            active_loan_count=1,
            overdue_book_count=1,
            unpaid_fines=Decimal("0.00"),
        )

        self.assertFalse(decision.approved)

    def test_overdue_fines_use_configured_daily_rate(self):
        self.assertEqual(
            calculate_overdue_fines([2, 3]),
            Decimal("2.50"),
        )

    def test_interlibrary_workflow_skips_partner_when_local_copy_exists(self):
        client = fake_client(
            '{"title": "Clean Code", "author": "Robert Martin", '
            '"action": "search", "category": "Software Engineering"}'
        )
        partner_search = Mock()

        result = inter_library_loan_workflow(
            "Find Clean Code",
            local_catalog_search=lambda _intent: {"available": True},
            partner_network_search=partner_search,
            client=client,
        )

        self.assertIn("available in the local", result)
        partner_search.assert_not_called()

    def test_agent_calls_registered_tool_and_returns_model_response(self):
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(
                name="catalog_search",
                arguments='{"title": "Clean Code"}',
            ),
        )
        tool_message = SimpleNamespace(
            tool_calls=[tool_call],
            model_dump=lambda **_kwargs: {
                "role": "assistant",
                "tool_calls": [],
            },
        )
        client = fake_client()
        client.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(message=tool_message)]),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="One copy is available.", tool_calls=None
                        )
                    )
                ]
            ),
        ]
        handler = Mock(return_value={"available": True})
        agent = AILibrarianReActAgent(
            [
                ReActTool(
                    name="catalog_search",
                    description="Search the catalog.",
                    parameters={
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                    },
                    handler=handler,
                )
            ],
            client=client,
        )

        self.assertEqual(agent.run("Is Clean Code available?"), "One copy is available.")
        handler.assert_called_once_with({"title": "Clean Code"})


if __name__ == "__main__":
    unittest.main()
