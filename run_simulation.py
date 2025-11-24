"""
Simple script to run the simulation.
Run this file to execute the diabetic patient consumable acquisition simulation.
"""

from model.model import ConsumableAcquisitionModel
from model.patient import create_patient_personas

if __name__ == "__main__":
    # Create patient personas
    patients_with_stock = create_patient_personas()
    
    # Initialize and run simulation
    model = ConsumableAcquisitionModel()
    model.run_simulation(patients_with_stock)
    
    # Export results
    model.export_to_csv()
    
    # Plot results
    model.plot_results()
    
    # Print summary
    print("\n=== Simulation Summary ===")
    for result in model.results:
        print(f"\n{result['patient_name']} (ID: {result['patient_id']}):")
        print(f"  Wants: {'Insulin' if result['wants_insulin'] else ''} {'Pump' if result['wants_pump'] else ''}".strip())
        print(f"  Pharmacy time: {result['pharmacy_total_time_days']:.2f} days ({result['pharmacy_total_time_minutes']:.1f} minutes)")
        print(f"  1177 Platform time: {result['1177_total_time_days']:.2f} days ({result['1177_total_time_minutes']:.1f} minutes)")
        print(f"  Total time: {result['total_time_days']:.2f} days ({result['total_time_minutes']:.1f} minutes)")
        print(f"  Items received from pharmacy: {result['pharmacy_items_received']}")

