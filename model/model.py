"""
Simulation model for diabetic patient consumable acquisition.
"""

import random
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from .patient import Patient


class ConsumableAcquisitionModel:
    """Simulates the process of acquiring diabetic consumables."""
    
    def __init__(self, seed: int = None):
        """
        Initialize the simulation model.
        
        Args:
            seed: Optional random seed for reproducibility. If None, uses system time.
        """
        self.results = []
        if seed is not None:
            random.seed(seed)
        # Otherwise, use system time (default behavior) for varying results
    
    def simulate_pharmacy_visit(
        self, 
        patient: Patient, 
        pharmacy_stock: Dict[str, bool]
    ) -> Dict[str, float]:
        """
        Simulates patient's pharmacy visit and ordering process.
        
        Args:
            patient: Patient object
            pharmacy_stock: Dict with stock status {"insulin": bool, "pump": bool}
        
        Returns:
            Dict with timing information:
            - travel_time: minutes
            - queue_time: minutes
            - handover_time: minutes (if items in stock)
            - order_wait_days: days (if items need to be ordered)
            - return_travel_time: minutes (if patient needs to return)
            - total_pharmacy_time: total minutes
            - items_received: list of items received
        """
        needs = patient.get_pharmacy_needs()
        timing = {
            "travel_time": patient.pharmacy_distance_min,
            "queue_time": random.uniform(4, 6),  # ~5 mins
            "handover_time": 0,
            "order_wait_days": 0,
            "return_travel_time": 0,
            "items_received": []
        }
        
        # Check stock for each needed item
        items_in_stock = []
        items_out_of_stock = []
        
        for item in needs:
            if pharmacy_stock.get(item, False):
                items_in_stock.append(item)
            else:
                items_out_of_stock.append(item)
        
        # If items are in stock, handover happens immediately
        initial_handover_time = 0
        return_handover_time = 0
        
        if items_in_stock:
            initial_handover_time = random.uniform(4, 6)  # ~5 mins
            timing["items_received"].extend(items_in_stock)
        
        # If some items are out of stock, pharmacy needs to order
        if items_out_of_stock:
            # Order wait time: 1-3 days converted to minutes
            timing["order_wait_days"] = random.uniform(1, 3)
            
            # Patient returns to pharmacy (travel time again)
            timing["return_travel_time"] = patient.pharmacy_distance_min
            
            # Handover when returning (always happens when returning)
            return_handover_time = random.uniform(4, 6)
            
            # Add items received after ordering
            timing["items_received"].extend(items_out_of_stock)
        
        timing["handover_time"] = initial_handover_time + return_handover_time
        
        # Calculate total pharmacy time in minutes
        total_minutes = (
            timing["travel_time"] +
            timing["queue_time"] +
            timing["handover_time"] +
            timing["return_travel_time"]
        )
        
        # Add order wait time if applicable
        if timing["order_wait_days"] > 0:
            total_minutes += (timing["order_wait_days"] * 24 * 60)
        
        timing["total_pharmacy_time"] = total_minutes
        
        return timing
    
    def simulate_1177_ordering(self, patient: Patient) -> Dict[str, float]:
        """
        Simulates patient's 1177 platform ordering process.
        
        Args:
            patient: Patient object
        
        Returns:
            Dict with timing information:
            - ordering_time: minutes (5-20 mins)
            - delivery_wait_days: days (2-7 days)
            - pickup_travel_time: minutes (10-30 mins)
            - total_1177_time: total minutes
        """
        timing = {
            "ordering_time": random.uniform(5, 20),
            "delivery_wait_days": random.uniform(2, 7),
            "pickup_travel_time": patient.pickup_distance_min,
            "items_received": ["cgm", "blood_stripes", "ketone_stripes"]
        }
        
        # Calculate total time in minutes
        delivery_wait_minutes = timing["delivery_wait_days"] * 24 * 60
        timing["total_1177_time"] = (
            timing["ordering_time"] +
            delivery_wait_minutes +
            timing["pickup_travel_time"]
        )
        
        return timing
    
    def simulate_patient(self, patient: Patient, pharmacy_stock: Dict[str, bool]) -> Dict:
        """
        Simulates complete process for a single patient.
        
        Args:
            patient: Patient object
            pharmacy_stock: Dict with stock status
        
        Returns:
            Dict with complete simulation results
        """
        # Step 1: Pharmacy visit
        pharmacy_timing = self.simulate_pharmacy_visit(patient, pharmacy_stock)
        
        # Step 2: 1177 platform ordering (happens after pharmacy)
        # Note: Patient can order online while waiting for pharmacy items if needed
        # But for simplicity, we'll model it sequentially
        platform_timing = self.simulate_1177_ordering(patient)
        
        # Total time is the maximum of both processes (since they can overlap)
        # But for this model, we'll treat them sequentially as specified
        total_time = pharmacy_timing["total_pharmacy_time"] + platform_timing["total_1177_time"]
        
        result = {
            "patient_id": patient.id,
            "patient_name": patient.name,
            "wants_insulin": patient.wants_insulin,
            "wants_pump": patient.wants_pump,
            "pharmacy_travel_time": pharmacy_timing["travel_time"],
            "pharmacy_queue_time": pharmacy_timing["queue_time"],
            "pharmacy_handover_time": pharmacy_timing["handover_time"],
            "pharmacy_order_wait_days": pharmacy_timing["order_wait_days"],
            "pharmacy_return_travel_time": pharmacy_timing["return_travel_time"],
            "pharmacy_total_time_minutes": pharmacy_timing["total_pharmacy_time"],
            "pharmacy_total_time_days": pharmacy_timing["total_pharmacy_time"] / (24 * 60),
            "pharmacy_items_received": ", ".join(pharmacy_timing["items_received"]),
            "1177_ordering_time": platform_timing["ordering_time"],
            "1177_delivery_wait_days": platform_timing["delivery_wait_days"],
            "1177_pickup_travel_time": platform_timing["pickup_travel_time"],
            "1177_total_time_minutes": platform_timing["total_1177_time"],
            "1177_total_time_days": platform_timing["total_1177_time"] / (24 * 60),
            "total_time_minutes": total_time,
            "total_time_days": total_time / (24 * 60)
        }
        
        return result
    
    def run_simulation(self, patients_with_stock: List[Tuple[Patient, Dict[str, bool]]]):
        """
        Runs simulation for all patients.
        
        Args:
            patients_with_stock: List of (Patient, pharmacy_stock) tuples
        """
        self.results = []
        
        for patient, pharmacy_stock in patients_with_stock:
            result = self.simulate_patient(patient, pharmacy_stock)
            self.results.append(result)
    
    def export_to_csv(self, output_dir: str = "result"):
        """
        Exports simulation results to CSV file with timestamp.
        
        Args:
            output_dir: Directory to save CSV file
        """
        if not self.results:
            raise ValueError("No simulation results to export. Run simulation first.")
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Write CSV
        fieldnames = [
            "patient_id", "patient_name", "wants_insulin", "wants_pump",
            "pharmacy_travel_time", "pharmacy_queue_time", "pharmacy_handover_time",
            "pharmacy_order_wait_days", "pharmacy_return_travel_time",
            "pharmacy_total_time_minutes", "pharmacy_total_time_days",
            "pharmacy_items_received",
            "1177_ordering_time", "1177_delivery_wait_days", "1177_pickup_travel_time",
            "1177_total_time_minutes", "1177_total_time_days",
            "total_time_minutes", "total_time_days"
        ]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"Results exported to {filepath}")
        return filepath
    
    def plot_results(self):
        """
        Creates visualization plots for the simulation results.
        """
        if not self.results:
            raise ValueError("No simulation results to plot. Run simulation first.")
        
        # Extract data
        patient_names = [r["patient_name"] for r in self.results]
        pharmacy_times = [r["pharmacy_total_time_days"] for r in self.results]
        platform_times = [r["1177_total_time_days"] for r in self.results]
        total_times = [r["total_time_days"] for r in self.results]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Diabetic Patient Consumable Acquisition Simulation Results', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Pharmacy time by patient
        bars1 = axes[0, 0].bar(patient_names, pharmacy_times, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Pharmacy Acquisition Time', fontweight='bold')
        axes[0, 0].set_ylabel('Time (days)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        # Add value labels on top of bars
        for bar in bars1:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=9)
        
        # Plot 2: 1177 Platform time by patient
        bars2 = axes[0, 1].bar(patient_names, platform_times, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('1177 Platform Acquisition Time', fontweight='bold')
        axes[0, 1].set_ylabel('Time (days)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        # Add value labels on top of bars
        for bar in bars2:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Total time comparison
        x = np.arange(len(patient_names))
        width = 0.25
        bars3a = axes[1, 0].bar(x - width, pharmacy_times, width, label='Pharmacy', 
                       color='skyblue', alpha=0.7)
        bars3b = axes[1, 0].bar(x, platform_times, width, label='1177 Platform', 
                       color='lightcoral', alpha=0.7)
        bars3c = axes[1, 0].bar(x + width, total_times, width, label='Total', 
                       color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Time Comparison by Source', fontweight='bold')
        axes[1, 0].set_ylabel('Time (days)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(patient_names, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        # Add value labels on top of bars
        for bars in [bars3a, bars3b, bars3c]:
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}',
                               ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Total time by patient
        bars4 = axes[1, 1].bar(patient_names, total_times, color='mediumseagreen', alpha=0.7)
        axes[1, 1].set_title('Total Acquisition Time', fontweight='bold')
        axes[1, 1].set_ylabel('Time (days)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        # Add value labels on top of bars
        for bar in bars4:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"simulation_plot_{timestamp}.png"
        plot_path = os.path.join("result", plot_filename)
        os.makedirs("result", exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        
        plt.show()


def main():
    """Main function to run the simulation."""
    from .patient import create_patient_personas
    
    # Set seed for reproducibility (use None for varying results each run)
    # Change 42 to any number, or set to None for random results
    SEED = None  # Set to a number (e.g., 42) for reproducible results
    
    # Create patient personas
    patients_with_stock = create_patient_personas(seed=SEED)
    
    # Initialize and run simulation
    model = ConsumableAcquisitionModel(seed=SEED)
    model.run_simulation(patients_with_stock)
    
    # Export results
    model.export_to_csv()
    
    # Plot results
    model.plot_results()
    
    # Print summary
    print("\n=== Simulation Summary ===")
    for result in model.results:
        print(f"\n{result['patient_name']} (ID: {result['patient_id']}):")
        print(f"  Pharmacy time: {result['pharmacy_total_time_days']:.2f} days")
        print(f"  1177 Platform time: {result['1177_total_time_days']:.2f} days")
        print(f"  Total time: {result['total_time_days']:.2f} days")


if __name__ == "__main__":
    main()

