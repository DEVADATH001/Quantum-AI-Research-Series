"""Research script for automated VQE architecture search and ablation studies."""

import copy
import os
import sys
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pes_generator import PESGenerator
from src.plotting import plot_pareto_front

def run_ablation_study(molecule: str = "LiH"):
    config_path = ROOT / "config" / "simulation_config.yaml"
    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Research Objective: Analyze how different rotation sets and reps affect accuracy vs cost
    # Ablation Search Space
    reps_range = [1, 2, 3]
    gate_sets = [["ry"], ["ry", "rz"]]
    entanglements = ["linear", "circular"]

    study_results = []
    
    print(f"Starting VQE Architecture Search for {molecule}...")
    
    for reps in reps_range:
        for gates in gate_sets:
            for ent in entanglements:
                config = copy.deepcopy(base_config)
                
                # Configure the specific ansatz variant
                gate_label = "+".join(gates)
                ansatz_name = f"EffSU2_{gate_label}_R{reps}_{ent}"
                
                config["vqe"]["ansatz"] = [{
                    "name": "EfficientSU2",
                    "reps": reps,
                    "entanglement": ent,
                    "su2_gates": gates
                }]
                
                # Run PES for a single characteristic bond length to evaluate architecture
                if molecule == "LiH":
                    target_bond = 1.6
                else:
                    target_bond = 0.74
                    
                config["molecules"][molecule]["distances"] = {
                    "start": target_bond,
                    "end": target_bond,
                    "step": 0.1
                }
                
                generator = PESGenerator(config)
                results = generator.run(molecule)
                
                # Extract results
                exact = results["exact_energies"][0]
                vqe = results["vqe_energies"]["EfficientSU2"][0]
                error = abs(vqe - exact)
                
                # Calculate Hardware Cost (CNOT count)
                # We can get the ansatz from the mapping_stats or build it again
                from src.ansatz_factory import get_ansatz
                from src.molecule_driver import get_molecule_problem
                from src.problem_builder import build_mapped_hamiltonian
                
                problem, _ = get_molecule_problem(molecule, target_bond)
                mapping = build_mapped_hamiltonian(problem)
                ansatz = get_ansatz("EfficientSU2", problem, mapping.mapper, 
                                   reps=reps, entanglement=ent, su2_gates=gates)
                
                # Count CNOTs (transpilation would be more accurate, but this is a good proxy)
                cnot_count = ansatz.decompose().count_ops().get("cx", 0)
                
                print(f"  Architecture: {ansatz_name} | Error: {error:.6f} | CNOTs: {cnot_count}")
                
                study_results.append({
                    "label": ansatz_name,
                    "error": error,
                    "cost": cnot_count,
                    "reps": reps,
                    "gates": gates,
                    "entanglement": ent
                })

    # Generate Pareto Front Plot
    errors = [res["error"] for res in study_results]
    costs = [res["cost"] for res in study_results]
    labels = [res["label"] for res in study_results]
    
    plot_pareto_front(errors, costs, labels, f"{molecule}_Ablation")
    
    # Save study data
    output_path = ROOT / "results" / "raw_data" / f"ablation_study_{molecule}.json"
    with open(output_path, "w") as f:
        json.dump(study_results, f, indent=2)
        
    print(f"\nStudy complete. Pareto front saved to results/figures/pareto_front_{molecule}_Ablation.png")
    print(f"Full study data saved to {output_path}")

if __name__ == "__main__":
    run_ablation_study("H2")
