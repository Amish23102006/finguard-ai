"""FinGuard Decision Intelligence Console (Streamlit).

Run: ``streamlit run src/finguard/dashboard/app.py``. Uses the in-process service so it works
without the API; every screen answers: what happened, why, how certain, what to check next.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from finguard.common.schemas import BorrowerProfile, Case, CommitteeDecision, RecommendedAction
from finguard.service import AssessmentService
from finguard.synthetic.builder import BEHAVIOURS, build_custom_case, verified_date
from finguard.synthetic.generator import generate_demo_cases

MAX_UPLOAD_BYTES = 2_000_000

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


def _optional_number(label: str, default: float, key: str, **kw) -> float | None:
    """Number input with an 'unknown' checkbox so missing data can be simulated."""
    c1, c2 = st.columns([3, 1])
    unknown = c2.checkbox("Unknown", key=f"{key}-unknown")
    value = c1.number_input(label, value=default, key=key, disabled=unknown, **kw)
    return None if unknown else value


def custom_case_form() -> Case | None:
    """Form to enter a synthetic borrower. Returns a Case once submitted."""
    st.header("Enter your own synthetic case")
    st.caption("Use made-up values only. Never enter real personal or customer data.")
    with st.form("custom-case"):
        st.subheader("Borrower profile")
        a, b = st.columns(2)
        with a:
            income = _optional_number(
                "Annual income", 70000.0, "income", min_value=0.0, step=1000.0
            )
            months = _optional_number(
                "Income verified how many months ago?", 2.0, "verified", min_value=0.0, step=1.0
            )
            dti = _optional_number(
                "Debt-to-income ratio (0.3 = 30%)", 0.25, "dti", min_value=0.0, step=0.05
            )
            history = _optional_number(
                "Credit history (months)", 96.0, "hist", min_value=0.0, step=6.0
            )
        with b:
            missed = st.number_input("Missed payments (last 12 months)", 0, 50, 0)
            util = st.number_input("Credit utilisation (0.3 = 30%)", 0.0, 5.0, 0.3, 0.05)
            employ = st.number_input("Years employed", 0.0, 60.0, 5.0, 1.0)
            loan = st.number_input("Requested loan amount", 0.0, value=15000.0, step=1000.0)

        st.subheader("Transaction behaviour (simulated)")
        c, d = st.columns(2)
        options: list[str] = list(BEHAVIOURS)
        behaviour: str = (
            c.selectbox(
                "Behaviour in the last 7 days", options, format_func=lambda k: BEHAVIOURS[k]
            )
            or "normal"
        )
        history_days = c.slider("Days of transaction history", 5, 365, 180)
        rate = d.slider("Average transactions per day", 0.2, 5.0, 1.5, 0.1)
        amount = d.number_input("Typical payment amount", 1.0, value=60.0, step=10.0)
        seed = d.number_input("Random seed (same seed = same result)", 0, 9999, 7)
        submitted = st.form_submit_button("Assess this case", type="primary")

    if submitted:
        try:
            verified = verified_date(None if months is None else int(months))
            profile = BorrowerProfile(
                annual_income=income,
                income_verified_at=verified.date() if verified else None,
                debt_to_income=dti,
                credit_history_months=None if history is None else int(history),
                missed_payments_12m=int(missed),
                credit_utilization=util,
                employment_years=employ,
                loan_amount=loan,
            )
            st.session_state["custom_case"] = build_custom_case(
                profile,
                behaviour=behaviour,
                history_days=history_days,
                daily_rate=rate,
                typical_amount=amount,
                seed=int(seed),
            )
        except (ValidationError, ValueError) as exc:
            st.error(f"Invalid input: {exc}")
    return st.session_state.get("custom_case")


def upload_case() -> Case | None:
    """Upload a Case JSON (same format as data/sample/*.json)."""
    st.header("Upload a case file")
    st.caption(
        "JSON in the same format as `data/sample/FG-DEMO-0001.json`. `synthetic` must be true."
    )
    file = st.file_uploader("Case JSON", type=["json"])
    if file is None:
        return None
    if file.size > MAX_UPLOAD_BYTES:
        st.error("File too large (max 2 MB).")
        return None
    try:
        return Case.model_validate_json(file.getvalue())
    except ValidationError as exc:
        st.error("The file does not match the Case schema:")
        st.code(str(exc))
        return None


def main() -> None:
    cases = get_cases()
    with st.sidebar:
        st.title("🛡️ FinGuard AI")
        st.caption("Decision Intelligence Console · synthetic data only")
        source = st.radio("Input", ["Demo cases", "Enter my own", "Upload JSON"], horizontal=False)
        demo_mode = False
        case_id = None
        if source == "Demo cases":
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

    if source == "Enter my own":
        case = custom_case_form()
    elif source == "Upload JSON":
        case = upload_case()
    else:
        case = cases[case_id]
    if case is None:
        st.stop()
    if source != "Demo cases":
        st.divider()

    try:
        decision = get_service().assess(case)
    except Exception as exc:  # graceful degradation
        st.error(f"Assessment service unavailable: {exc}. No recommendation is shown.")
        st.stop()

    st.header(f"Case {case.case_id} · {case.entity_label}")
    if demo_mode:
        st.info(SCENARIO_TOUR.get(case.case_id, ""), icon="🧭")
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
            c2.checkbox(check, key=f"{case.case_id}-{check}")
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
            file_name=f"{case.case_id}_passport.json",
            mime="application/json",
        )
        st.json(passport, expanded=False)


main()
