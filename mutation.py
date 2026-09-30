"""
AXIS-Gen prototype -- mutation.py

Mutation operators implementing the well-established XACML fault model
(Martin & Xie 2007 "A fault model and mutation testing of access control
policies"; Bertolino et al. XACMUT 2010/2012) adapted to our internal
policy representation. Each operator produces one "mutant" Policy that
differs from the original by exactly one seeded fault.

Operators implemented:
  RCM  - Rule Combining algorithm Mutation
  CEM  - rule effeCt (Effect) Mutation                (Permit <-> Deny)
  CPM  - Comparison oPerator Mutation                 (>= -> <, eq -> neq, ...)
  CVM  - Constant Value boundary Mutation              (threshold +/- 1, category swap)
  TRM  - Target Removal (broadens a rule's applicability by dropping 1 leaf)
  LOM  - Logical Operator Mutation                     (AND -> OR)
  MRD  - Missing Rule Deletion                         (drop a whole rule)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List

from policy_model import BoolExpr, Effect, Policy, Predicate, Rule

_OP_FLIP = {
    "eq": "neq", "neq": "eq",
    "gte": "lt", "lte": "gt",
    "gt": "lte", "lt": "gte",
    "in": "not_in", "not_in": "in",
}

_ALGS = ["deny-overrides", "permit-overrides", "first-applicable"]


@dataclass
class Mutant:
    mutant_id: str
    operator: str
    description: str
    policy: Policy


def _clone(policy: Policy) -> Policy:
    return copy.deepcopy(policy)


def _map_leaves(expr: BoolExpr, fn, only_index: List[int], counter: List[int]) -> BoolExpr:
    """Rebuild expr, applying fn(pred) to the k-th leaf encountered (k in only_index)."""
    if expr.kind == "leaf":
        idx = counter[0]
        counter[0] += 1
        if idx in only_index:
            return BoolExpr.leaf_(fn(expr.leaf))
        return expr
    if expr.kind in ("true",):
        return expr
    if expr.kind == "not":
        return BoolExpr("not", children=(_map_leaves(expr.children[0], fn, only_index, counter),))
    return BoolExpr(expr.kind, children=tuple(_map_leaves(c, fn, only_index, counter) for c in expr.children))


def _drop_leaf(expr: BoolExpr, drop_index: int, counter: List[int]) -> BoolExpr:
    """Return expr with the drop_index-th leaf removed from an AND/OR list
    (folds into TRUE if it becomes the identity); used for TRM."""
    if expr.kind == "leaf":
        idx = counter[0]
        counter[0] += 1
        if idx == drop_index:
            return BoolExpr.TRUE()
        return expr
    if expr.kind == "true":
        return expr
    if expr.kind == "not":
        return BoolExpr("not", children=(_drop_leaf(expr.children[0], drop_index, counter),))
    new_children = tuple(_drop_leaf(c, drop_index, counter) for c in expr.children)
    # collapse trivial TRUE children out of AND/OR so the rule genuinely broadens
    kept = [c for c in new_children if c.kind != "true"]
    if not kept:
        return BoolExpr.TRUE()
    if len(kept) == 1:
        return kept[0]
    return BoolExpr(expr.kind, children=tuple(kept))


def generate_mutants(policy: Policy) -> List[Mutant]:
    mutants: List[Mutant] = []

    # --- RCM: rule combining algorithm mutation --------------------------
    for alg in _ALGS:
        if alg != policy.combining_algorithm:
            m = _clone(policy)
            m.combining_algorithm = alg
            mutants.append(Mutant(
                f"RCM_{alg}", "RCM",
                f"combining algorithm {policy.combining_algorithm} -> {alg}", m,
            ))

    for ridx, rule in enumerate(policy.rules):
        # --- CEM: flip rule effect ----------------------------------------
        m = _clone(policy)
        m.rules[ridx].effect = Effect.DENY if rule.effect == Effect.PERMIT else Effect.PERMIT
        mutants.append(Mutant(
            f"CEM_{rule.rule_id}", "CEM",
            f"{rule.rule_id}: effect {rule.effect.name} -> {m.rules[ridx].effect.name}", m,
        ))

        # --- CPM: flip each comparison operator in target+condition -------
        for part_name, part in (("target", rule.target), ("condition", rule.condition)):
            n_leaves = len(part.leaves())
            for i in range(n_leaves):
                orig_pred = part.leaves()[i]
                if orig_pred.op not in _OP_FLIP:
                    continue
                m = _clone(policy)
                target_rule = m.rules[ridx]
                base = target_rule.target if part_name == "target" else target_rule.condition
                new_expr = _map_leaves(
                    base,
                    lambda p: Predicate(p.ref, _OP_FLIP[p.op], p.value),
                    [i], [0],
                )
                if part_name == "target":
                    target_rule.target = new_expr
                else:
                    target_rule.condition = new_expr
                mutants.append(Mutant(
                    f"CPM_{rule.rule_id}_{part_name}{i}", "CPM",
                    f"{rule.rule_id}.{part_name}[{i}] op {orig_pred.op} -> {_OP_FLIP[orig_pred.op]}", m,
                ))

        # --- CVM: boundary-shift constant mutation on numeric predicates --
        for part_name, part in (("target", rule.target), ("condition", rule.condition)):
            for i, pred in enumerate(part.leaves()):
                if isinstance(pred.value, int) and pred.op in ("gte", "lte", "gt", "lt"):
                    m = _clone(policy)
                    target_rule = m.rules[ridx]
                    base = target_rule.target if part_name == "target" else target_rule.condition
                    new_expr = _map_leaves(
                        base, lambda p: Predicate(p.ref, p.op, p.value + 1), [i], [0],
                    )
                    if part_name == "target":
                        target_rule.target = new_expr
                    else:
                        target_rule.condition = new_expr
                    mutants.append(Mutant(
                        f"CVM_{rule.rule_id}_{part_name}{i}", "CVM",
                        f"{rule.rule_id}.{part_name}[{i}] threshold {pred.value} -> {pred.value + 1}", m,
                    ))

        # --- TRM: drop one leaf from target (broadens applicability) ------
        n_target_leaves = len(rule.target.leaves())
        for i in range(n_target_leaves):
            if n_target_leaves <= 1:
                continue  # dropping the only leaf makes Target vacuous/degenerate; skip
            m = _clone(policy)
            target_rule = m.rules[ridx]
            target_rule.target = _drop_leaf(target_rule.target, i, [0])
            mutants.append(Mutant(
                f"TRM_{rule.rule_id}_t{i}", "TRM",
                f"{rule.rule_id}: dropped target leaf #{i}", m,
            ))

        # --- LOM: AND -> OR on the condition's top-level connective --------
        if rule.condition.kind == "and" and len(rule.condition.children) > 1:
            m = _clone(policy)
            target_rule = m.rules[ridx]
            target_rule.condition = BoolExpr("or", children=target_rule.condition.children)
            mutants.append(Mutant(
                f"LOM_{rule.rule_id}", "LOM",
                f"{rule.rule_id}: condition AND -> OR", m,
            ))

        # --- MRD: delete the whole rule -------------------------------------
        m = _clone(policy)
        del m.rules[ridx]
        mutants.append(Mutant(
            f"MRD_{rule.rule_id}", "MRD",
            f"deleted rule {rule.rule_id}", m,
        ))

    return mutants
