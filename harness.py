"""
AXIS-Gen prototype -- harness.py

Runs a test suite (list of Request) against the original policy and every
mutant, and computes the metrics used in the experiment section of the
research document:

  * mutation_score   : fraction of mutants killed (decision differs from
                        the original policy on at least one request)
  * rule_coverage    : fraction of rules whose Permit/Deny outcome was
                        exercised at least once (each rule reached a
                        non-NOT_APPLICABLE decision)
  * requests_to_95pct: how many requests (in generation order) were needed
                        before mutation score first reached >=95% of final
  * decision_balance : Permit / Deny / NotApplicable distribution, as a
                        cheap sanity signal against trivial always-deny
                        suites inflating "safety" numbers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from mutation import Mutant, generate_mutants
from policy_model import Decision, Policy, Request


@dataclass
class SuiteResult:
    method: str
    n_requests: int
    mutation_score: float
    killed: int
    total_mutants: int
    rule_coverage: float
    requests_to_95pct: int
    requests_to_fixed: int  # requests needed to reach a fixed, cross-method absolute kill count
    decision_counts: Dict[str, int]


def _decision_of(policy: Policy, req: Request) -> Decision:
    d, _ = policy.evaluate(req)
    return d


def evaluate_suite(method: str, requests: List[Request], policy: Policy, mutants: List[Mutant],
                    fixed_kill_target: int = 20) -> SuiteResult:
    n = len(requests)
    baseline_decisions = [_decision_of(policy, r) for r in requests]

    decision_counts = {"PERMIT": 0, "DENY": 0, "NOT_APPLICABLE": 0, "INDETERMINATE": 0}
    for d in baseline_decisions:
        decision_counts[d.name] += 1

    # --- rule coverage: for each rule, did any request cause it to reach
    #     a non-NOT_APPLICABLE verdict (i.e. its Target matched)?
    reached = [False] * len(policy.rules)
    for req in requests:
        for i, rule in enumerate(policy.rules):
            if rule.evaluate(req) != Decision.NOT_APPLICABLE:
                reached[i] = True
    rule_coverage = sum(reached) / len(policy.rules) if policy.rules else 0.0

    # --- mutation score, tracked incrementally to also report
    #     requests_to_95pct (a cheap efficiency proxy)
    killed_so_far = set()
    progression: List[int] = []
    for i, req in enumerate(requests):
        base_d = baseline_decisions[i]
        for m in mutants:
            if m.mutant_id in killed_so_far:
                continue
            if _decision_of(m.policy, req) != base_d:
                killed_so_far.add(m.mutant_id)
        progression.append(len(killed_so_far))

    total = len(mutants)
    final_killed = len(killed_so_far)
    mutation_score = final_killed / total if total else 0.0

    target95 = 0.95 * final_killed
    req_to_95 = n
    for i, k in enumerate(progression, start=1):
        if k >= target95:
            req_to_95 = i
            break

    req_to_fixed = n + 1  # "not reached within budget" sentinel
    for i, k in enumerate(progression, start=1):
        if k >= fixed_kill_target:
            req_to_fixed = i
            break

    return SuiteResult(
        method=method,
        n_requests=n,
        mutation_score=mutation_score,
        killed=final_killed,
        total_mutants=total,
        rule_coverage=rule_coverage,
        requests_to_95pct=req_to_95,
        requests_to_fixed=req_to_fixed,
        decision_counts=decision_counts,
    )


def print_report(results: List[SuiteResult]) -> None:
    header = (f"{'method':22s} {'#req':>5s} {'mut.score':>10s} {'killed/total':>13s} "
              f"{'rule cov':>9s} {'req->20killed':>13s}  decisions(P/D/NA)")
    print(header)
    print("-" * len(header))
    for r in results:
        dc = r.decision_counts
        r20 = "n/a" if r.requests_to_fixed > r.n_requests else str(r.requests_to_fixed)
        print(
            f"{r.method:22s} {r.n_requests:5d} {r.mutation_score*100:9.1f}% "
            f"{r.killed:5d}/{r.total_mutants:<7d} {r.rule_coverage*100:8.1f}% "
            f"{r20:>13s}  {dc['PERMIT']}/{dc['DENY']}/{dc['NOT_APPLICABLE']}"
        )
