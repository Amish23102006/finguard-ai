"""FinGuard Decision Intelligence Console (Streamlit).

Run: ``streamlit run src/finguard/dashboard/app.py``. Uses the in-process service so it works
without the API; every screen answers: what happened, why, how certain, what to check next.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from finguard.common.schemas import CommitteeDecision, RecommendedAction
from finguard.service import AssessmentService
from finguard.synthetic.generator import generate_demo_cases

st.set_page_config(page_title="FinGuard AI", page_icon="🛡️", layout="wide")

ACTION_STYLE = {
    RecommendedAction.NORMAL_PROCESSING: ("success", "Standard processing"),
    RecommendedAction.REQUEST_MORE_INFORMATION: ("info", "Request more information"),
    RecommendedAction.HUMAN_REVIEW: ("warning", "Human review"),
    RecommendedAction.ENHANCED_HUMAN_VERIFICATION: ("error", "Enhanced human verification"),
    RecommendedAction.ABSTAIN: ("warning", "Abstain: route to a person"),
}
SCENARIO_TOUR = {
    "FG-DEMO-0002": (
        "Normal: every agent agrees the risk is low, and evidence and limitations are still shown."
    ),
    "FG-DEMO-0001": (
        "Conflicting evidence: credit looks healthy but behaviour is unusual. "
        "FinGuard measures the disagreement instead of averaging it away."
    ),
    "FG-DEMO-0003": (
        "Unknown pattern: the case is unlike the training data, so the system abstains."
    ),
    "FG-DEMO-0004": (
        "Missing data: critical fields are absent, so the system says 'insufficient evidence'."
    ),
}


@st.cache_resource(show_spinner="Loading models and policy…")
def get_service() -> AssessmentService:
    return AssessmentService()


@st.cache_data
def get_cases() -> dict:
    return {c.case_id: c for c in generate_demo_cases()}


def metric_row(d: CommitteeDecision) -> None:
    cols = st.columns(5)
    cols[0].metric(
        "Overall risk",
        f"{d.overall_risk:.0f} / 100",
        d.risk_band.replace("_", "-").upper(),
        delta_color="off",
    )
    cols[1].metric("Confidence", f"{d.confidence:.0%}")
    cols[2].metric("Data reliability", f"{d.data_reliability:.0%}")
    cols[3].metric(
        "Agent disagreement",
        f"{d.disagreement:.0f} / 100",
        d.disagreement_level.upper(),
        delta_color="off",
    )
    cols[4].metric("Unknown-pattern score", f"{d.unknown_pattern_score:.0%}")


def main() -> None:
    cases = get_cases()
    with st.sidebar:
        st.title("🛡️ FinGuard AI")
        st.caption("Decision Intelligence Console · synthetic data only")
        demo_mode = st.toggle(
            "Recruiter demo mode", value=True, help="Guided tour through the seeded scenarios"
        )
        order = list(SCENARIO_TOUR) if demo_mode else sorted(cases)
        case_id = st.radio("Case", order, format_func=lambda c: f"{c} · {cases[c].scenario}")
        plain = st.toggle("Plain-language mode", value=True)
        st.divider()
        st.caption(
            "Not financial advice. FinGuard never approves, denies or accuses. "
            "It recommends what a human should review."
        )

    case = cases[case_id]
    try:
        decision = get_service().assess(case)
    except Exception as exc:  # graceful degradation
        st.error(f"Assessment service unavailable: {exc}. No recommendation is shown.")
        st.stop()

    st.header(f"Case {case.case_id} · {case.entity_label}")
    if demo_mode:
        st.info(SCENARIO_TOUR.get(case_id, ""), icon="🧭")
    metric_row(decision)
    kind, label = ACTION_STYLE[decision.recommended_action]
    getattr(st, kind)(f"**Recommendation: {label}**  \n{decision.narrative['what_happened']}")

    tabs = st.tabs(["Overview", "Agent Room", "Data & Timeline", "Decision Passport"])
    with tabs[0]:
        q = decision.narrative
        c1, c2 = st.columns(2)
        c1.subheader("Why does the system think so?")
        c1.write(q["why"])
        c1.subheader("How certain is it?")
        c1.write(q["how_certain"])
        c2.subheader("What should a human check next?")
        for check in decision.human_checks:
            c2.checkbox(check, key=f"{case_id}-{check}")
        with c2.expander("Limitations", expanded=not plain):
            for lim in decision.limitations:
                st.write(f"- {lim}")
        if not plain:
            st.subheader("Policy rules fired")
            st.code("\n".join(decision.reasons))

    with tabs[1]:
        cols = st.columns(len(decision.agent_responses))
        for col, r in zip(cols, decision.agent_responses, strict=True):
            with col.container(border=True):
                st.markdown(f"#### {r.agent_name.replace('_', ' ').title()}")
                st.metric("Risk score", f"{r.risk_score:.0f}")
                st.progress(r.risk_score / 100)
                st.caption(
                    f"confidence {r.confidence:.0%} · data quality {r.data_quality:.0%} · "
                    f"OOD {r.ood_score:.0%} · `{r.model_version}`"
                )
                st.markdown("**Evidence**")
                for e in r.evidence:
                    st.write(f"- {e}")
                st.markdown("**Limitations**")
                for lim in r.limitations:
                    st.write(f"- {lim}")
                if r.contributions and not plain:
                    st.markdown("**Feature contributions (log-odds, linear SHAP)**")
                    st.bar_chart(
                        pd.DataFrame(
                            {"contribution": [c.contribution for c in r.contributions]},
                            index=[c.feature for c in r.contributions],
                        ),
                        horizontal=True,
                    )

    with tabs[2]:
        st.subheader("Borrower profile (synthetic)")
        st.dataframe(pd.DataFrame([case.profile.model_dump()]), hide_index=True)
        if case.transactions:
            st.subheader("Daily outgoing amount")
            tx = pd.DataFrame([t.model_dump() for t in case.transactions])
            tx["date"] = pd.to_datetime(tx["timestamp"]).dt.date
            daily = tx[tx.direction == "out"].groupby("date")["amount"].sum()
            st.area_chart(daily)
            with st.expander("Last 25 transactions"):
                st.dataframe(tx.sort_values("timestamp", ascending=False).head(25), hide_index=True)

    with tabs[3]:
        st.write("A reproducible audit record of inputs, versions, evidence and outcome.")
        passport = decision.model_dump(mode="json")
        st.download_button(
            "⬇️ Download Decision Passport (JSON)",
            json.dumps(passport, indent=2),
            file_name=f"{case_id}_passport.json",
            mime="application/json",
        )
        st.json(passport, expanded=False)


main()
