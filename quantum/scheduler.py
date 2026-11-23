import itertools
from typing import List, Dict, Any, Optional

RISK_WEIGHT = 2.5
LOAD_BALANCE_PENALTY = 0.5
MAINTENANCE_PENALTY = 50.0

DEFAULT_JOBS = [
    {"id": "J1", "duration": 12, "energy": 8.0},
    {"id": "J2", "duration": 25, "energy": 14.0},
    {"id": "J3", "duration": 8, "energy": 5.0},
    {"id": "J4", "duration": 18, "energy": 11.0},
]

def compute_schedule_cost(
    assignment: Dict[str, str],
    jobs: List[Dict[str, Any]],
    risk_by_machine: Dict[str, float],
    machines: List[str],
    offline_machines: Optional[List[str]] = None
) -> float:
    """
    Computes total objective cost:
    Cost = sum_{j,m} [ energy_j + RISK_WEIGHT * risk_m * duration_j ]
           + LOAD_BALANCE_PENALTY * sum_m ( (total_duration_m - avg_duration)^2 )
           + penalty if high risk machine assigned jobs without maintenance
    """
    if offline_machines is None:
        offline_machines = []

    base_cost = 0.0
    machine_durations = {m: 0.0 for m in machines}

    for job in jobs:
        j_id = job["id"]
        m = assignment.get(j_id)
        if not m:
            continue
        dur = job.get("duration", 10.0)
        energy = job.get("energy", 5.0)
        risk = risk_by_machine.get(m, 0.0)

        # Severe penalty if assigned to an offline machine
        if m in offline_machines:
            base_cost += 500.0

        base_cost += energy + (RISK_WEIGHT * risk * dur)
        machine_durations[m] += dur

    # Penalty if machine has risk > 0.75 and takes more than 1 job or heavy load
    for m in machines:
        risk = risk_by_machine.get(m, 0.0)
        if risk >= 0.75 and machine_durations[m] > 15.0:
            base_cost += MAINTENANCE_PENALTY * (risk - 0.75) * 2.0

    # Load balancing penalty
    active_machines = [m for m in machines if m not in offline_machines]
    if active_machines:
        total_dur = sum(j.get("duration", 10.0) for j in jobs)
        target_dur = total_dur / len(active_machines)
        balance_penalty = LOAD_BALANCE_PENALTY * sum((machine_durations[m] - target_dur) ** 2 for m in active_machines)
    else:
        balance_penalty = 1000.0

    return round(base_cost + balance_penalty, 3)

def classical_exact_solve(
    jobs: List[Dict[str, Any]],
    risk_by_machine: Dict[str, float],
    machines: List[str],
    offline_machines: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Exact brute-force search over all machine assignments (M^J combinations)."""
    if offline_machines is None:
        offline_machines = []

    valid_machines = [m for m in machines if m not in offline_machines]
    if not valid_machines:
        valid_machines = machines  # Fallback if all offline

    job_ids = [j["id"] for j in jobs]
    best_assignment = None
    best_cost = float("inf")

    for combo in itertools.product(valid_machines, repeat=len(jobs)):
        assignment = dict(zip(job_ids, combo))
        cost = compute_schedule_cost(assignment, jobs, risk_by_machine, machines, offline_machines=offline_machines)
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment

    return {
        "assignment": best_assignment,
        "cost": best_cost
    }

def optimize(
    jobs: Optional[List[Dict[str, Any]]] = None,
    risk_by_machine: Optional[Dict[str, float]] = None,
    machines: Optional[List[str]] = None,
    offline_machines: Optional[List[str]] = None,
    include_maintenance_job: bool = False
) -> Dict[str, Any]:
    """
    Formulates job-to-machine assignment as a QUBO and attempts to solve via Qiskit QAOA.
    Gracefully falls back to classical exact optimization if Qiskit is absent or encounters error.
    Always computes classical optimum for benchmarking.
    """
    if jobs is None:
        jobs = [dict(j) for j in DEFAULT_JOBS]
    else:
        jobs = [dict(j) for j in jobs]

    if machines is None:
        machines = ["M1", "M2", "M3"]
    if risk_by_machine is None:
        risk_by_machine = {m: 0.1 for m in machines}
    if offline_machines is None:
        offline_machines = []

    # Optional maintenance task insertion: high risk machines receive dedicated maintenance slot
    if include_maintenance_job:
        for m, r in risk_by_machine.items():
            if r >= 0.70 and m not in offline_machines:
                jobs.append({"id": f"MAINT_{m}", "duration": 20, "energy": 2.0, "target_machine": m})

    # Classical optimal solve
    classical_res = classical_exact_solve(jobs, risk_by_machine, machines, offline_machines=offline_machines)
    classical_optimal_cost = classical_res["cost"]

    # Try quantum QAOA solve
    quantum_assignment = None
    quantum_cost = None
    solver_used = "classical-fallback"

    try:
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit.primitives import Sampler

        available_machines = [m for m in machines if m not in offline_machines]
        if not available_machines:
            available_machines = machines

        qp = QuadraticProgram(name="JobScheduling")
        num_jobs = len(jobs)
        num_machines = len(available_machines)
        var_names = {}

        for j_idx, job in enumerate(jobs):
            for m_idx, m in enumerate(available_machines):
                vname = f"x_{j_idx}_{m_idx}"
                qp.binary_var(name=vname)
                var_names[(j_idx, m_idx)] = vname

        # Constraint: each job assigned to exactly one machine
        for j_idx in range(num_jobs):
            qp.linear_constraint(
                linear={var_names[(j_idx, m_idx)]: 1 for m_idx in range(num_machines)},
                sense="==",
                rhs=1,
                name=f"one_machine_job_{j_idx}"
            )

        # Objective linear terms: sum(energy_j + RISK_WEIGHT * risk_m * dur_j)
        linear_obj = {}
        for j_idx, job in enumerate(jobs):
            dur = job.get("duration", 10.0)
            energy = job.get("energy", 5.0)
            target_m = job.get("target_machine")
            for m_idx, m in enumerate(available_machines):
                risk = risk_by_machine.get(m, 0.0)
                vname = var_names[(j_idx, m_idx)]
                weight = energy + (RISK_WEIGHT * risk * dur)
                if target_m and m != target_m:
                    weight += 100.0  # Force maintenance job on specified machine
                linear_obj[vname] = weight

        qp.minimize(linear=linear_obj)

        sampler = Sampler()
        qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=20), reps=1)
        optimizer = MinimumEigenOptimizer(qaoa)
        result = optimizer.solve(qp)

        # Interpret candidate
        sol_dict = result.variables_dict
        assignment_candidate = {}
        for j_idx, job in enumerate(jobs):
            for m_idx, m in enumerate(available_machines):
                vname = var_names[(j_idx, m_idx)]
                if sol_dict.get(vname, 0) > 0.5:
                    assignment_candidate[job["id"]] = m
                    break

        if len(assignment_candidate) == num_jobs:
            quantum_assignment = assignment_candidate
            quantum_cost = compute_schedule_cost(quantum_assignment, jobs, risk_by_machine, machines, offline_machines)
            solver_used = "QAOA"
    except Exception:
        solver_used = "classical-fallback"

    if solver_used == "QAOA" and quantum_assignment is not None:
        return {
            "assignment": quantum_assignment,
            "cost": quantum_cost,
            "solver": "QAOA",
            "classical_optimal_cost": classical_optimal_cost
        }
    else:
        return {
            "assignment": classical_res["assignment"],
            "cost": classical_res["cost"],
            "solver": "classical-fallback",
            "classical_optimal_cost": classical_optimal_cost
        }

if __name__ == "__main__":
    print("--- Testing Quantum/Classical Job Scheduler ---")
    mock_risk = {"M1": 0.05, "M2": 0.2, "M3": 0.85}
    res = optimize(risk_by_machine=mock_risk, offline_machines=["M3"])
    print(f"Solver used: {res['solver']}")
    print(f"Optimal schedule: {res['assignment']}")
    print(f"Schedule Cost: {res['cost']}")
    print(f"Classical Optimal Cost: {res['classical_optimal_cost']}")
