"""
Simple script to run the simulation.
Run this file to execute the diabetic patient consumable acquisition simulation.
"""

from model.model import ConsumableAcquisitionModel
from model.patient import create_patient_population

if __name__ == "__main__":
    # Set seed for reproducibility (use None for varying results each run)
    # Change 42 to any number, or set to None for random results
    SEED = None  # Set to a number (e.g., 42) for reproducible results
    
    # Create patient population following the requested percentages and arrival flow
    patients = create_patient_population(seed=SEED)
    
    # Initialize and run simulation
    model = ConsumableAcquisitionModel(seed=SEED)
    model.run_simulation(patients)
    
    # Export results
    model.export_to_csv()
    
    # Plot results
    model.plot_results()
    
    # Print summary
    print("\n=== Simulation KPIs ===")
    print(f"Average total time (days): {model.kpis.get('avg_total_minutes', 0) / (24 * 60):.2f}")
    print(f"Average pharmacy time (days): {model.kpis.get('avg_pharmacy_minutes', 0) / (24 * 60):.2f}")
    print(f"Average 1177 time (days): {model.kpis.get('avg_1177_minutes', 0) / (24 * 60):.2f}")
    print(f"Return rate: {model.kpis.get('return_rate', 0) * 100:.1f}%")

