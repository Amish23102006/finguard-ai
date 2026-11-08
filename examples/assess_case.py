"""Assess a demo case from the command line and print the Decision Passport summary.

Usage: python examples/assess_case.py FG-DEMO-0001
"""

from __future__ import annotations

import sys

from finguard.service import AssessmentService
from finguard.synthetic.generator import get_demo_case


def main(case_id: str) -> None:
    case = get_demo_case(case_id)
    if case is None:
        raise SystemExit(f"Unknown case {case_id}")
    decision = AssessmentService().assess(case)
    print(f"\n{case_id}: {decision.recommended_action.value.upper()}")
    for question, answer in decision.narrative.items():
        print(f"\n{question.replace('_', ' ').title()}:\n  {answer}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "FG-DEMO-0001")
