"""
AXIS-Gen prototype -- ics_policy.py

A small but semantically realistic ABAC policy for a water-treatment-plant
style ICS (Purdue-model zones, safety interlock override, vendor remote
access restriction). This is the "policy under test" (PUT) for the
mutation-testing experiment.
"""

from policy_model import (
    AttrRef, BoolExpr, Effect, Policy, Predicate, Rule,
)

S = lambda a: AttrRef("subject", a)
R = lambda a: AttrRef("resource", a)
A = lambda a: AttrRef("action", a)
E = lambda a: AttrRef("environment", a)

L = BoolExpr.leaf_
AND = BoolExpr.AND
OR = BoolExpr.OR
TRUE = BoolExpr.TRUE


def build_ics_policy() -> Policy:
    rules = []

    # R1: operators may read HMIs/PLCs in control/supervisory zones.
    rules.append(Rule(
        rule_id="R1_operator_read",
        effect=Effect.PERMIT,
        target=AND(
            L(Predicate(S("role"), "eq", "operator")),
            L(Predicate(A("id"), "eq", "read")),
        ),
        condition=L(Predicate(R("zone"), "in", {"level1_control", "level2_supervisory"})),
        description="Operators can read HMI/PLC state in control & supervisory zones.",
    ))

    # R2: engineers may read/write in field..supervisory zones, but not
    # during an emergency plant mode.
    rules.append(Rule(
        rule_id="R2_engineer_readwrite",
        effect=Effect.PERMIT,
        target=AND(
            L(Predicate(S("role"), "eq", "engineer")),
            L(Predicate(A("id"), "in", {"read", "write"})),
        ),
        condition=AND(
            L(Predicate(R("zone"), "in", {"level0_field", "level1_control", "level2_supervisory"})),
            L(Predicate(E("plant_mode"), "neq", "emergency")),
        ),
        description="Engineers can read/write field..supervisory assets outside emergencies.",
    ))

    # R3: nobody may override a safety interlock while the plant is in a
    # normal safety state (must be in maintenance state to touch it at all).
    rules.append(Rule(
        rule_id="R3_deny_override_when_normal",
        effect=Effect.DENY,
        target=AND(
            L(Predicate(A("id"), "eq", "override_interlock")),
            L(Predicate(R("criticality"), "eq", "safety")),
        ),
        condition=L(Predicate(E("safety_state"), "eq", "normal")),
        description="Safety interlocks cannot be overridden while safety_state=normal.",
    ))

    # R4: a sufficiently-cleared, on-site engineer MAY override a safety
    # interlock, but only while the plant is explicitly in maintenance.
    # This rule is the richest MC/DC target: 4 ANDed atomic conditions.
    rules.append(Rule(
        rule_id="R4_permit_engineer_override",
        effect=Effect.PERMIT,
        target=AND(
            L(Predicate(S("role"), "eq", "engineer")),
            L(Predicate(A("id"), "eq", "override_interlock")),
            L(Predicate(R("criticality"), "eq", "safety")),
        ),
        condition=AND(
            L(Predicate(E("safety_state"), "eq", "maintenance")),
            L(Predicate(S("clearance_level"), "gte", 4)),
            L(Predicate(S("location"), "eq", "onsite")),
        ),
        description="Cleared on-site engineers may override a safety interlock during maintenance.",
    ))

    # R5: vendors are denied any access to level0 field devices.
    rules.append(Rule(
        rule_id="R5_deny_vendor_field",
        effect=Effect.DENY,
        target=L(Predicate(S("role"), "eq", "vendor")),
        condition=L(Predicate(R("zone"), "eq", "level0_field")),
        description="Vendors are denied access to level-0 field-zone devices.",
    ))

    # R6: vendors may read the historian, but only during normal production.
    rules.append(Rule(
        rule_id="R6_permit_vendor_historian_read",
        effect=Effect.PERMIT,
        target=AND(
            L(Predicate(S("role"), "eq", "vendor")),
            L(Predicate(A("id"), "eq", "read")),
            L(Predicate(R("type"), "eq", "historian")),
        ),
        condition=L(Predicate(E("plant_mode"), "eq", "production")),
        description="Vendors may read the historian during normal production.",
    ))

    # NOTE ON DESIGN: we deliberately do NOT add an always-applicable
    # "default-deny" rule (Target=TRUE) at the Policy level. Under
    # deny-overrides, such a catch-all rule would evaluate to DENY for
    # every request and therefore dominate the combined decision
    # regardless of what any other rule does -- masking the effect of
    # almost every other mutation and making top-level decisions useless
    # as a test oracle (a real and under-appreciated pitfall in XACML
    # policy testing). In a real deployment the enclosing PolicySet (or
    # the PEP) supplies deny-as-default handling of NOT_APPLICABLE;
    # here we keep NOT_APPLICABLE observable so the PDP oracle can
    # distinguish "no rule matched" from "a rule explicitly decided",
    # which is exactly the information a mutation-based test oracle needs.

    return Policy(policy_id="ICS-WaterTreatment-ABAC-v1", combining_algorithm="deny-overrides", rules=rules)
