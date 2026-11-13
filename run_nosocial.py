"""
Run the no-social-layer baseline for all settings and save SQLB state trajectories.

What this file does:
1. Loads the same experiment grid as `run.py` from `settings.csv`
   (same names, seeds, organizations, initial opinions, and AI accuracies).
2. Builds each organization + technology exactly as `run.py` does, so every
   baseline run starts from the same network, initial opinions, initial trust,
   and technology seed as the social run with the same name.
3. Removes the social layer (influence matrix W = I): each agent keeps its own
   opinion, which changes only through that agent's own AI outcomes.
   With W = I the consensus steps return every opinion unchanged, and team /
   organization opinions and trust no longer feed back into agent opinions,
   so `NoSocialModel` skips those steps and runs only technology use. This
   gives exactly the same agent-opinion trajectories as running the full
   update loop with W = I (the AI outcomes use the same RNG stream in the same
   order), but much faster.
4. Runs the model for the configured `steps`.
5. Classifies agents into SQLB states with the same rules as `state.py` and
   saves the per-step fractions to `states_nosocial/{name}.csv`
   (same format as `states/{name}.csv`). Because trust is never updated
   without the social layer, loudness (Q vs L) uses the initial trust; use
   Q + L (total adoption) for comparisons.
6. Executes in parallel with a process pool and skips existing outputs,
   so an interrupted run can simply be started again.

Model JSONs are not saved (the social run's `models/` folder is ~28 GB);
only the state trajectories are kept.

How it is used in the pipeline:
- Provides the individual-only counterfactual for each social run.
  Each file in `states_nosocial/` is paired with the file of the same name
  in `states/`.
- `compare_social_vs_nosocial.py` reads both folders and builds the comparison.
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import trustdynamics as td

from utils import BASE_DIR, load_settings
from config import steps, technology_use_cutoff_opinion
from state import sqlb_state, get_inbound_trust, LOUDNESS_BOUND

STATES_NOSOCIAL_DIR = BASE_DIR / "states_nosocial"
STATES_NOSOCIAL_DIR.mkdir(parents=True, exist_ok=True)


class NoSocialModel(td.Model):
    """
    The same trustdynamics model with the social layer removed (W = I).

    Each step keeps only the technology-use update. The skipped steps (team
    and organization consensus, team and agent trust updates) cannot change
    agent opinions when W = I, so agent-opinion trajectories are identical to
    the full loop run with W = I.
    """

    def update(self):
        if self.organization.require_initialization is True:
            self.organization.initialize()
        self.agents_use_technology()


def build_model(setting: dict) -> td.Model:
    """Build one model exactly as `run.py` does, then remove social influence."""
    seed = int(setting["seed"])

    organization = td.organization.generate_random_organization_structure(
        teams_num=int(setting["teams_num"]),
        agents_num=int(setting["agents_num"]),
        agents_connection_probability_inside_team=float(setting["agents_connection_probability"]),
        teams_connection_probability=float(setting["teams_connection_probability"]),
        technology_use_cutoff_opinion=float(technology_use_cutoff_opinion),
        seed=seed,
    )

    organization.initialize(
        agents_average_initial_opinion=float(setting["agents_average_initial_opinion"]),
        seed=seed,
    )

    technology = td.Technology(
        success_rate=float(setting["technology_success_rate"]),
        seed=seed,
    )

    return NoSocialModel(organization, technology)  # <- no social layer (W = I)


def write_states(model: td.Model, path) -> None:
    """Write SQLB fractions per step, same rules and format as `state.py`."""
    organization = model.organization
    agents = list(organization.all_agent_ids)
    n_agents = len(agents)

    # Trust is never updated without the social layer, so each agent's inbound
    # trust (used for loudness) is its initial value and is computed once.
    inbound_trust = {agent: get_inbound_trust(model, agent, 0) for agent in agents}
    opinions = {agent: organization.get_agent_opinions_history(agent) for agent in agents}

    tmp_path = path.with_name(path.name + ".tmp")  # avoid half-written files if interrupted
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["t", "ratio_S", "ratio_Q", "ratio_L", "ratio_B"],
        )
        writer.writeheader()

        for t in range(steps):
            counts = {"S": 0, "Q": 0, "L": 0, "B": 0}
            for agent in agents:
                opinion = float(opinions[agent][t])
                loudness = opinion * inbound_trust[agent]
                st = sqlb_state(opinion, loudness, loudness_bound=LOUDNESS_BOUND)
                counts[st] += 1

            writer.writerow(
                {
                    "t": t,
                    "ratio_S": counts["S"] / n_agents,
                    "ratio_Q": counts["Q"] / n_agents,
                    "ratio_L": counts["L"] / n_agents,
                    "ratio_B": counts["B"] / n_agents,
                }
            )

    os.replace(tmp_path, path)


def run_one(setting: dict) -> tuple[str, str]:
    name = str(setting["name"])
    path = STATES_NOSOCIAL_DIR / f"{name}.csv"
    if path.exists():
        return name, "skipped (exists)"

    model = build_model(setting)
    if len(list(model.organization.all_agent_ids)) == 0:
        return name, "skipped (no agents)"

    model.run(steps=int(steps), show_progress=False)
    write_states(model, path)

    return name, "saved"


def main() -> None:
    print(f"trustdynamics version: {td.__version__}")
    settings = load_settings()

    max_workers = max(1, (os.cpu_count() or 2) - 2)
    total = len(settings)
    done = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one, st) for st in settings]
        for fut in as_completed(futures):
            name, status = fut.result()
            done += 1
            print(f"[{done}/{total}] {name}: {status}")


if __name__ == "__main__":
    main()
