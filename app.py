"""Watch LAMA learn affordances in the numpy warehouse, and compare it with the baselines.

    streamlit run app.py

NumPy only (no PyTorch), so it also runs in the browser through Pyodide/stlite.
"""
import functools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from lama.env.warehouse import Warehouse
from lama.evaluation import Evaluator
from lama.evaluation.baselines import NoveltyPolicy, RandomPolicy, uncertainty_only_policy
from lama.memory.memory import AffordanceMemory
from lama.verification.loop import run_episode
from lama.verification.select import select_next

st.set_page_config(page_title="LAMA: affordance learning", page_icon="🤖", layout="wide")

POLICIES = {
    "LAMA (full system)": lambda seed: select_next,
    "Uncertainty only": lambda seed: uncertainty_only_policy,
    "Novelty (Plan2Explore-style)": lambda seed: NoveltyPolicy(seed=seed),
    "Random": lambda seed: RandomPolicy(seed=seed),
}
RESULTS = Path("outputs/baseline_comparison_with_ablation.json")

st.title("LAMA: learning what objects afford, on a budget")
st.caption("An agent in a warehouse of objects with hidden mass and hidden affordances. It imagines what each verb "
           "would do, tests only what it is most uncertain about, and keeps only what verification confirms.")

with st.sidebar:
    policy_name = st.selectbox("Selection policy", list(POLICIES))
    seed = st.number_input("Seed", 0, 1000, 0)
    episodes = st.slider("Episodes (one persistent memory)", 1, 60, 20)
    budget = st.slider("Interaction budget per episode", 20.0, 120.0, 60.0, 5.0)
    n_objects = st.slider("Objects in the warehouse", 4, 16, 10)

tab_run, tab_cmp, tab_paper = st.tabs(["Run the agent", "Compare policies", "Published results"])


def run_lifelong(policy_name, seed, episodes, budget, n_objects):
    policy = POLICIES[policy_name](seed)
    memory = AffordanceMemory()
    per_episode, last_steps, last_obs = [], [], None
    for ep in range(episodes):
        env = Warehouse(seed=seed * 1000 + ep, budget=budget, n_objects=n_objects)
        steps = run_episode(env, memory, seed=seed * 1000 + ep, select_fn=policy)
        tested = [s for s in steps if s.hypothesis is not None]
        per_episode.append({"episode": ep + 1, "interactions": len(steps), "tests": len(tested),
                            "confirmed beliefs": len(memory.bank.confirmed()),
                            "beliefs held": len(memory.bank.beliefs())})
        last_steps = steps
        if steps:
            last_obs = steps[-1].step.observation
    return memory, pd.DataFrame(per_episode), last_steps, last_obs


with tab_run:
    memory, curve, steps, obs = run_lifelong(policy_name, int(seed), episodes, budget, n_objects)
    c1, c2, c3 = st.columns(3)
    c1.metric("Confirmed affordances", len(memory.bank.confirmed()))
    c2.metric("Beliefs held (any status)", len(memory.bank.beliefs()))
    c3.metric("Concepts tested (formed from appearance)", len({b.key[0] for b in memory.bank.beliefs()}))
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Knowledge over episodes**")
        st.line_chart(curve.set_index("episode")[["confirmed beliefs", "beliefs held"]])
    with right:
        st.markdown("**Warehouse at the end of the last episode**")
        if obs is not None:
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            pos = np.array([o.position for o in obs.objects])
            ax.scatter(pos[:, 0], pos[:, 1], s=80, c=[int(o.object_id.split("_")[-1]) % 10 for o in obs.objects], cmap="tab10")
            for o in obs.objects:
                ax.annotate(o.object_id.replace("obj_", ""), o.position, fontsize=8, ha="center", va="center")
            ax.scatter(*obs.agent_position, marker="*", s=250, color="k", label="agent")
            ax.set_aspect("equal")
            ax.legend(loc="upper right")
            st.pyplot(fig)
    rows = []
    for b in sorted(memory.bank.beliefs(), key=lambda b: (-b.total_attempts, str(b.key))):
        concept, action, tool = b.key
        rows.append({"target concept": concept, "verb": action.name.lower(), "tool concept": tool,
                     "status": b.status.name.lower(), "P(works)": round(b.mean, 2),
                     "attempts": b.total_attempts, "usual effect": b.dominant_effect.name.lower()
                     if b.dominant_effect is not None else ""})
    st.markdown("**Affordance bank** (only *confirmed* rows may be acted on)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("Step log of the last episode"):
        log = []
        for s in steps:
            r = s.step.record
            log.append({"t": r.t, "verb": r.interaction.action.name.lower(), "target": r.interaction.target,
                        "holding": r.tool_id, "effect": r.outcome.effect.name.lower(), "cost": r.cost,
                        "why": "test" if s.hypothesis is not None else "walk toward"})
        st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)

with tab_cmp:
    st.markdown("Every policy runs through the **same** environment, memory and adjudication; only the choice of "
                "what to test differs. Scored against the hidden ground truth, by affordance role.")
    k_seeds = st.slider("Seeds", 1, 5, 2)
    k_eps = st.slider("Episodes per seed", 5, 30, 10)
    if st.button("Run comparison"):
        rows = []
        for name in POLICIES:
            for s in range(k_seeds):
                pol = POLICIES[name](s)
                ev = Evaluator(env_factory=lambda seed=s, **kw: Warehouse(seed=seed, budget=budget,
                                                                           n_objects=n_objects, **kw),
                               memory_factory=AffordanceMemory,
                               run_episode_fn=functools.partial(run_episode, select_fn=pol))
                pr = ev.evaluate_precision_recall(n_episodes=k_eps, seed=s)
                eff = ev.evaluate_efficiency(n_episodes=k_eps, seed=s)
                rows.append({"policy": name, "precision": pr["precision"], "recall": pr["recall"],
                             "primary recall": pr["by_role"]["primary"]["recall"],
                             "secondary recall": pr["by_role"]["secondary"]["recall"],
                             "interactions per confirmed": eff["interactions_per_confirmed"]})
        df = pd.DataFrame(rows).groupby("policy", sort=False).mean().round(3)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df[["primary recall", "secondary recall"]])

with tab_paper:
    if RESULTS.exists():
        d = json.loads(RESULTS.read_text())
        cfg = d["config"]
        st.markdown(f"From `{RESULTS}`: **{cfg['seeds']} seeds × {cfg['episodes']} episodes**, budget "
                    f"{cfg['budget']}, {cfg['n_objects']} objects (mean ± standard deviation over seeds).")
        table = {name: {m: f"{v['mean']:.3f} ± {v['std']:.3f}" for m, v in s.items()}
                 for name, s in d["summary"].items()}
        st.dataframe(pd.DataFrame(table).T, use_container_width=True)
        st.markdown("LAMA wins on overall recall, primary recall and interactions per confirmed affordance, but not "
                    "on **secondary** recall, where uncertainty-only selection is ahead: the curiosity and safety "
                    "terms deprioritise `place_on` and `tip`, where secondary affordances live. "
                    "See `docs/RESEARCH_FINDINGS.md`.")
