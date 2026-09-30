"""
AXIS-Gen prototype -- policy_model.py

A faithful-but-simplified model of an XACML 3.0 attribute-based access
control (ABAC) policy for an industrial control system (ICS), together
with:

  * a Policy Decision Point (PDP) evaluator implementing the standard
    XACML rule/policy combining algorithms (deny-overrides,
    permit-overrides, first-applicable), and
  * a serializer that emits *real* OASIS XACML 3.0 XML for the policy
    and for individual requests, so the artifact produced by the
    pipeline is a genuine XACML request, not a toy JSON stand-in.

This is intentionally a research-prototype subset of XACML 3.0
(no obligations/advice, no XPath categories, no multiple-decision
profile) -- enough to host realistic mutation testing and coverage
experiments without requiring a full Java PDP such as AuthzForce/Balana.
"""

from __future__ import annotations

import itertools
import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from lxml import etree

# --------------------------------------------------------------------------
# Attribute domains (the "schema" an LLM / combinatorial engine must respect)
# --------------------------------------------------------------------------

# category -> attribute -> domain (explicit finite set, or ("range", lo, hi))
AttributeDomain = Dict[str, Dict[str, Any]]

ICS_ATTRIBUTE_DOMAINS: AttributeDomain = {
    "subject": {
        "role": ["operator", "engineer", "vendor", "auditor"],
        "clearance_level": ("range", 1, 4),
        "location": ["onsite", "remote"],
    },
    "resource": {
        "type": ["hmi", "plc", "valve_actuator", "historian", "safety_plc"],
        "zone": ["level0_field", "level1_control", "level2_supervisory", "level3_operations"],
        "criticality": ["low", "medium", "high", "safety"],
    },
    "action": {
        "id": ["read", "write", "reconfigure", "override_interlock"],
    },
    "environment": {
        "safety_state": ["normal", "alarm", "maintenance"],
        "plant_mode": ["production", "maintenance", "emergency"],
    },
}

Effect = Enum("Effect", ["PERMIT", "DENY"])
Decision = Enum("Decision", ["PERMIT", "DENY", "NOT_APPLICABLE", "INDETERMINATE"])

_OPS: Dict[str, Callable[[Any, Any], bool]] = {
    "eq": operator.eq,
    "neq": operator.ne,
    "gte": operator.ge,
    "lte": operator.le,
    "gt": operator.gt,
    "lt": operator.lt,
    "in": lambda v, s: v in s,
    "not_in": lambda v, s: v not in s,
}


@dataclass(frozen=True)
class AttrRef:
    category: str   # subject | resource | action | environment
    attribute: str


@dataclass(frozen=True)
class Predicate:
    """One atomic comparison, e.g. subject.role eq 'engineer'."""
    ref: AttrRef
    op: str
    value: Any

    def eval(self, request: "Request") -> Optional[bool]:
        actual = request.get(self.ref.category, self.ref.attribute)
        if actual is None:
            return None  # missing attribute -> Indeterminate at this predicate
        try:
            return _OPS[self.op](actual, self.value)
        except TypeError:
            return None


@dataclass(frozen=True)
class BoolExpr:
    """A small boolean expression tree over Predicates (AND/OR/NOT/leaf)."""
    kind: str  # "and" | "or" | "not" | "leaf" | "true"
    children: Tuple["BoolExpr", ...] = ()
    leaf: Optional[Predicate] = None

    @staticmethod
    def leaf_(p: Predicate) -> "BoolExpr":
        return BoolExpr("leaf", leaf=p)

    @staticmethod
    def AND(*es: "BoolExpr") -> "BoolExpr":
        return BoolExpr("and", children=tuple(es))

    @staticmethod
    def OR(*es: "BoolExpr") -> "BoolExpr":
        return BoolExpr("or", children=tuple(es))

    @staticmethod
    def TRUE() -> "BoolExpr":
        return BoolExpr("true")

    def eval(self, request: "Request") -> Optional[bool]:
        if self.kind == "true":
            return True
        if self.kind == "leaf":
            return self.leaf.eval(request)
        if self.kind == "not":
            v = self.children[0].eval(request)
            return None if v is None else (not v)
        results = [c.eval(request) for c in self.children]
        if self.kind == "and":
            if any(r is False for r in results):
                return False
            if any(r is None for r in results):
                return None
            return True
        if self.kind == "or":
            if any(r is True for r in results):
                return True
            if any(r is None for r in results):
                return None
            return False
        raise ValueError(self.kind)

    def leaves(self) -> List[Predicate]:
        if self.kind == "leaf":
            return [self.leaf]
        out: List[Predicate] = []
        for c in self.children:
            out.extend(c.leaves())
        return out


@dataclass
class Rule:
    rule_id: str
    effect: Effect
    target: BoolExpr           # applicability test (Target)
    condition: BoolExpr        # additional Condition (must also hold)
    description: str = ""

    def evaluate(self, request: "Request") -> Decision:
        t = self.target.eval(request)
        if t is False:
            return Decision.NOT_APPLICABLE
        if t is None:
            return Decision.INDETERMINATE
        c = self.condition.eval(request)
        if c is True:
            return Decision.PERMIT if self.effect == Effect.PERMIT else Decision.DENY
        if c is False:
            return Decision.NOT_APPLICABLE
        return Decision.INDETERMINATE


@dataclass
class Policy:
    policy_id: str
    combining_algorithm: str  # "deny-overrides" | "permit-overrides" | "first-applicable"
    rules: List[Rule] = field(default_factory=list)

    def evaluate(self, request: "Request") -> Tuple[Decision, List[Tuple[str, Decision]]]:
        trace: List[Tuple[str, Decision]] = []
        for r in self.rules:
            d = r.evaluate(request)
            trace.append((r.rule_id, d))
        decisions = [d for _, d in trace]

        if self.combining_algorithm == "first-applicable":
            for d in decisions:
                if d in (Decision.PERMIT, Decision.DENY):
                    return d, trace
            return Decision.NOT_APPLICABLE, trace

        if self.combining_algorithm == "deny-overrides":
            if Decision.DENY in decisions:
                return Decision.DENY, trace
            if Decision.INDETERMINATE in decisions:
                return Decision.INDETERMINATE, trace
            if Decision.PERMIT in decisions:
                return Decision.PERMIT, trace
            return Decision.NOT_APPLICABLE, trace

        if self.combining_algorithm == "permit-overrides":
            if Decision.PERMIT in decisions:
                return Decision.PERMIT, trace
            if Decision.INDETERMINATE in decisions:
                return Decision.INDETERMINATE, trace
            if Decision.DENY in decisions:
                return Decision.DENY, trace
            return Decision.NOT_APPLICABLE, trace

        raise ValueError(f"unknown combining algorithm: {self.combining_algorithm}")


@dataclass(frozen=True)
class Request:
    values: Dict[Tuple[str, str], Any]

    def get(self, category: str, attribute: str) -> Any:
        return self.values.get((category, attribute))

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for (cat, attr), val in self.values.items():
            out.setdefault(cat, {})[attr] = val
        return out

    @staticmethod
    def from_dict(d: Dict[str, Dict[str, Any]]) -> "Request":
        values = {}
        for cat, attrs in d.items():
            for attr, val in attrs.items():
                values[(cat, attr)] = val
        return Request(values)


# --------------------------------------------------------------------------
# Real OASIS XACML 3.0 XML export (policy + individual requests)
# --------------------------------------------------------------------------

XACML_NS = "urn:oasis:names:tc:xacml:3.0:core:schema:wd-17"
_CAT_URN = {
    "subject": "urn:oasis:names:tc:xacml:1.0:subject-category:access-subject",
    "resource": "urn:oasis:names:tc:xacml:3.0:attribute-category:resource",
    "action": "urn:oasis:names:tc:xacml:3.0:attribute-category:action",
    "environment": "urn:oasis:names:tc:xacml:3.0:attribute-category:environment",
}
_OP_URN = {
    "eq": "urn:oasis:names:tc:xacml:1.0:function:string-equal",
    "gte": "urn:oasis:names:tc:xacml:1.0:function:integer-greater-than-or-equal",
    "lte": "urn:oasis:names:tc:xacml:1.0:function:integer-less-than-or-equal",
}
_ALG_URN = {
    "deny-overrides": "urn:oasis:names:tc:xacml:1.0:rule-combining-algorithm:deny-overrides",
    "permit-overrides": "urn:oasis:names:tc:xacml:1.0:rule-combining-algorithm:permit-overrides",
    "first-applicable": "urn:oasis:names:tc:xacml:1.0:rule-combining-algorithm:first-applicable",
}


def request_to_xacml_xml(request: Request) -> str:
    """Serialize a Request into a real XACML 3.0 <Request> document."""
    root = etree.Element("{%s}Request" % XACML_NS, nsmap={None: XACML_NS})
    root.set("CombinedDecision", "false")
    root.set("ReturnPolicyIdList", "false")
    grouped = request.as_dict()
    for category, attrs in grouped.items():
        attrs_el = etree.SubElement(root, "{%s}Attributes" % XACML_NS)
        attrs_el.set("Category", _CAT_URN.get(category, category))
        for name, value in attrs.items():
            attr_el = etree.SubElement(attrs_el, "{%s}Attribute" % XACML_NS)
            attr_el.set("AttributeId", f"ics:{category}:{name}")
            attr_el.set("IncludeInResult", "true")
            dtype = "http://www.w3.org/2001/XMLSchema#integer" if isinstance(value, int) \
                else "http://www.w3.org/2001/XMLSchema#string"
            val_el = etree.SubElement(attr_el, "{%s}AttributeValue" % XACML_NS)
            val_el.set("DataType", dtype)
            val_el.text = str(value)
    return etree.tostring(root, pretty_print=True, encoding="unicode")


def policy_to_xacml_xml(policy: Policy) -> str:
    """Serialize the Policy into (an illustrative, non-exhaustive) XACML 3.0
    <Policy> document -- enough to show the artifact is genuinely XACML
    shaped, not a full round-trippable serializer for every predicate."""
    root = etree.Element("{%s}Policy" % XACML_NS, nsmap={None: XACML_NS})
    root.set("PolicyId", policy.policy_id)
    root.set("Version", "1.0")
    root.set("RuleCombiningAlgId", _ALG_URN[policy.combining_algorithm])
    etree.SubElement(root, "{%s}Target" % XACML_NS)
    for r in policy.rules:
        rule_el = etree.SubElement(root, "{%s}Rule" % XACML_NS)
        rule_el.set("RuleId", r.rule_id)
        rule_el.set("Effect", "Permit" if r.effect == Effect.PERMIT else "Deny")
        target_el = etree.SubElement(rule_el, "{%s}Target" % XACML_NS)
        for pred in r.target.leaves():
            _emit_match(target_el, pred)
        if r.condition.kind != "true":
            cond_el = etree.SubElement(rule_el, "{%s}Condition" % XACML_NS)
            comment = etree.Comment(
                " condition: " + _describe_bool(r.condition) + " "
            )
            cond_el.append(comment)
    return etree.tostring(root, pretty_print=True, encoding="unicode")


def _emit_match(parent, pred: Predicate) -> None:
    match_el = etree.SubElement(parent, "{%s}AllOf" % XACML_NS)
    m = etree.SubElement(match_el, "{%s}Match" % XACML_NS)
    m.set("MatchId", _OP_URN.get(pred.op, "urn:oasis:names:tc:xacml:1.0:function:string-equal"))
    v = etree.SubElement(m, "{%s}AttributeValue" % XACML_NS)
    v.text = str(pred.value)
    d = etree.SubElement(m, "{%s}AttributeDesignator" % XACML_NS)
    d.set("Category", _CAT_URN.get(pred.ref.category, pred.ref.category))
    d.set("AttributeId", f"ics:{pred.ref.category}:{pred.ref.attribute}")


def _describe_bool(e: BoolExpr) -> str:
    if e.kind == "true":
        return "true"
    if e.kind == "leaf":
        p = e.leaf
        return f"{p.ref.category}.{p.ref.attribute} {p.op} {p.value!r}"
    if e.kind == "not":
        return "NOT (" + _describe_bool(e.children[0]) + ")"
    joiner = " AND " if e.kind == "and" else " OR "
    return "(" + joiner.join(_describe_bool(c) for c in e.children) + ")"


def domain_values(ref: AttrRef, domains: AttributeDomain = ICS_ATTRIBUTE_DOMAINS) -> List[Any]:
    dom = domains[ref.category][ref.attribute]
    if isinstance(dom, tuple) and dom[0] == "range":
        return list(range(dom[1], dom[2] + 1))
    return list(dom)


def all_attr_refs(domains: AttributeDomain = ICS_ATTRIBUTE_DOMAINS) -> List[AttrRef]:
    return [AttrRef(cat, attr) for cat, attrs in domains.items() for attr in attrs]
