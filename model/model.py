"""
Simulation model for diabetic patient consumable acquisition.
"""

import csv
import os
import random
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .patient import Patient


class ConsumableAcquisitionModel:
    """Simulates the process of acquiring diabetic consumables."""
    
    ITEM_UNITS = {"insulin": 10, "pump": 5}
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the simulation model and pharmacy state.
        
        Args:
            seed: Optional random seed for reproducibility. If None, uses system time.
        """
        self.results: List[Dict] = []
        self.kpis: Dict[str, float] = {}
        self.max_stock = {"insulin": 100, "pump": 30}
        self.restock_threshold = {item: capacity * 0.2 for item, capacity in self.max_stock.items()}
        if seed is not None:
            random.seed(seed)
        self._reset_state()
    
    def _reset_state(self):
        """Reset mutable pharmacy state so simulations start from a clean slate."""
        self.stock = dict(self.max_stock)
        self.restock_eta = {item: None for item in self.max_stock}
        self.backlog_units: Dict[str, List[float]] = {item: [] for item in self.max_stock}
    
    def _schedule_restock(self, item: str, current_time: float) -> float:
        """Schedule a new restock for an item if one is not already on the way."""
        delay_hours = random.uniform(12, 24)  # Pharmacies restock within roughly half to one day
        eta = current_time + delay_hours * 60
        if self.restock_eta[item] is None or eta < self.restock_eta[item]:
            self.restock_eta[item] = eta
        return self.restock_eta[item]
    
    def _ensure_restock(self, item: str, current_time: float) -> float:
        """Ensure a restock is scheduled and return the wait time in minutes."""
        eta = self.restock_eta[item]
        if eta is None or eta < current_time:
            eta = self._schedule_restock(item, current_time)
        return max(0, eta - current_time)
    
    def _maybe_trigger_restock(self, item: str, current_time: float):
        """Trigger a restock when inventory falls under 20% capacity."""
        if self.stock[item] <= self.restock_threshold[item] and self.restock_eta[item] is None:
            self._schedule_restock(item, current_time)
    
    def _refresh_stock_levels(self, current_time: float):
        """Apply completed restocks and honor queued backorders."""
        for item in self.stock:
            eta = self.restock_eta[item]
            if eta is not None and current_time >= eta:
                self.stock[item] = self.max_stock[item]
                self.restock_eta[item] = None
                
                # Serve backordered units in FIFO order
                while self.backlog_units[item]:
                    units_needed = self.backlog_units[item][0]
                    if self.stock[item] >= units_needed:
                        self.stock[item] -= units_needed
                        self.backlog_units[item].pop(0)
                    else:
                        # Not enough to fulfill the next backorder; wait for the next delivery.
                        remaining = units_needed - self.stock[item]
                        self.backlog_units[item][0] = remaining
                        self.stock[item] = 0
                        self._schedule_restock(item, current_time)
                        break
                
                # Immediately queue another shipment if we dipped below the safety threshold again
                if self.stock[item] <= self.restock_threshold[item] and self.restock_eta[item] is None:
                    self._schedule_restock(item, current_time)
    
    def simulate_pharmacy_visit(self, patient: Patient, current_time: float) -> Dict[str, float]:
        """Simulate the patient's pharmacy visit considering travel, queue, service, and stock."""
        self._refresh_stock_levels(current_time)
        
        travel_time = patient.pharmacy_distance_min
        queue_time = random.uniform(3, 8)
        service_time = random.uniform(4, 6)
        
        restock_wait_minutes = 0
        return_travel_time = 0
        return_queue_time = 0
        return_service_time = 0
        items_now: List[str] = []
        items_after_restock: List[str] = []
        
        for item in patient.get_pharmacy_needs():
            units_needed = self.ITEM_UNITS[item]
            if self.stock[item] >= units_needed:
                self.stock[item] -= units_needed
                items_now.append(item)
                self._maybe_trigger_restock(item, current_time)
            else:
                patient.return_pharmacy = True
                wait_time = self._ensure_restock(item, current_time)
                restock_wait_minutes = max(restock_wait_minutes, wait_time)
                items_after_restock.append(item)
                self.backlog_units[item].append(units_needed)
        
        if patient.return_pharmacy:
            return_travel_time = patient.pharmacy_distance_min
            return_queue_time = random.uniform(3, 8)
            return_service_time = random.uniform(4, 6)
        
        total_minutes = (
            travel_time +
            queue_time +
            service_time +
            restock_wait_minutes +
            return_travel_time +
            return_queue_time +
            return_service_time
        )
        
        return {
            "arrival_day": patient.arrival_day_name,
            "arrival_time": patient.arrival_time_hhmm,
            "travel_time": travel_time,
            "queue_time": queue_time,
            "service_time": service_time,
            "restock_wait_minutes": restock_wait_minutes,
            "return_travel_time": return_travel_time,
            "return_queue_time": return_queue_time,
            "return_service_time": return_service_time,
            "total_pharmacy_time_minutes": total_minutes,
            "items_received": items_now + items_after_restock,
        }
    
    def simulate_1177_ordering(self, patient: Patient) -> Dict[str, float]:
        """
        Simulates the patient's 1177 platform ordering process.
        
        Returns:
            Dict with ordering/delivery metrics.
        """
        timing = {
            "ordering_time": random.uniform(5, 20),
            "delivery_wait_days": random.uniform(2, 7),
            "pickup_travel_time": patient.pickup_distance_min,
            "items_received": ["cgm", "blood_stripes", "ketone_stripes"],
        }
        delivery_wait_minutes = timing["delivery_wait_days"] * 24 * 60
        timing["total_1177_time_minutes"] = (
            timing["ordering_time"] + delivery_wait_minutes + timing["pickup_travel_time"]
        )
        return timing
    
    def simulate_patient(self, patient: Patient) -> Dict:
        """Simulate the complete acquisition journey for a single patient."""
        arrival_time = patient.arrival_minutes_since_start
        pharmacy_timing = self.simulate_pharmacy_visit(patient, arrival_time)
        platform_timing = self.simulate_1177_ordering(patient)
        
        total_time_minutes = max(
            pharmacy_timing["total_pharmacy_time_minutes"],
            platform_timing["total_1177_time_minutes"],
        )
        
        return {
            "patient_id": patient.id,
            "patient_name": patient.name,
            "arrival_day": pharmacy_timing["arrival_day"],
            "arrival_time": pharmacy_timing["arrival_time"],
            "wants_insulin": patient.wants_insulin,
            "wants_pump": patient.wants_pump,
            "return_pharmacy": patient.return_pharmacy,
            "pharmacy_travel_time": pharmacy_timing["travel_time"],
            "pharmacy_queue_time": pharmacy_timing["queue_time"],
            "pharmacy_service_time": pharmacy_timing["service_time"],
            "pharmacy_restock_wait_minutes": pharmacy_timing["restock_wait_minutes"],
            "pharmacy_return_travel_time": pharmacy_timing["return_travel_time"],
            "pharmacy_return_queue_time": pharmacy_timing["return_queue_time"],
            "pharmacy_return_service_time": pharmacy_timing["return_service_time"],
            "pharmacy_total_time_minutes": pharmacy_timing["total_pharmacy_time_minutes"],
            "pharmacy_total_time_days": pharmacy_timing["total_pharmacy_time_minutes"] / (24 * 60),
            "pharmacy_items_received": ", ".join(pharmacy_timing["items_received"]),
            "1177_ordering_time": platform_timing["ordering_time"],
            "1177_delivery_wait_days": platform_timing["delivery_wait_days"],
            "1177_pickup_travel_time": platform_timing["pickup_travel_time"],
            "1177_total_time_minutes": platform_timing["total_1177_time_minutes"],
            "1177_total_time_days": platform_timing["total_1177_time_minutes"] / (24 * 60),
            "total_time_minutes": total_time_minutes,
            "total_time_days": total_time_minutes / (24 * 60),
        }
    
    def run_simulation(self, patients: List[Patient]):
        """Runs the simulation for all provided patients."""
        if not patients:
            raise ValueError("No patients provided to simulation.")
        
        self.results = []
        self._reset_state()
        
        sorted_patients = sorted(patients, key=lambda p: p.arrival_minutes_since_start)
        for patient in sorted_patients:
            patient.return_pharmacy = False  # reset per run
            result = self.simulate_patient(patient)
            self.results.append(result)
        
        self._calculate_kpis()
    
    def _calculate_kpis(self):
        """Calculate headline KPIs from the simulation output."""
        if not self.results:
            self.kpis = {}
            return
        
        pharmacy_times = np.array([r["pharmacy_total_time_minutes"] for r in self.results])
        platform_times = np.array([r["1177_total_time_minutes"] for r in self.results])
        total_times = np.array([r["total_time_minutes"] for r in self.results])
        return_ratio = sum(1 for r in self.results if r["return_pharmacy"]) / len(self.results)
        
        self.kpis = {
            "avg_total_minutes": float(np.mean(total_times)),
            "avg_pharmacy_minutes": float(np.mean(pharmacy_times)),
            "avg_1177_minutes": float(np.mean(platform_times)),
            "return_rate": return_ratio,
        }
    
    def export_to_csv(self, output_dir: str = "result"):
        """Exports simulation results to a timestamped CSV file."""
        if not self.results:
            raise ValueError("No simulation results to export. Run simulation first.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        
        fieldnames = [
            "patient_id", "patient_name", "arrival_day", "arrival_time",
            "wants_insulin", "wants_pump", "return_pharmacy",
            "pharmacy_travel_time", "pharmacy_queue_time", "pharmacy_service_time",
            "pharmacy_restock_wait_minutes", "pharmacy_return_travel_time",
            "pharmacy_return_queue_time", "pharmacy_return_service_time",
            "pharmacy_total_time_minutes", "pharmacy_total_time_days",
            "pharmacy_items_received",
            "1177_ordering_time", "1177_delivery_wait_days", "1177_pickup_travel_time",
            "1177_total_time_minutes", "1177_total_time_days",
            "total_time_minutes", "total_time_days",
        ]
        
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"Results exported to {filepath}")
        return filepath
    
    def plot_results(self):
        """Create visualization plots for the simulation results."""
        if not self.results:
            raise ValueError("No simulation results to plot. Run simulation first.")
        
        # Prepare data frames
        pharmacy_times = np.array([r["pharmacy_total_time_days"] for r in self.results])
        platform_times = np.array([r["1177_total_time_days"] for r in self.results])
        total_times = np.array([r["total_time_days"] for r in self.results])
        returns = np.array([r["return_pharmacy"] for r in self.results])
        arrival_days = np.array([r["arrival_day"] for r in self.results])
        
        # Categorize needs
        needs_category = []
        for r in self.results:
            if r["wants_insulin"] and r["wants_pump"]:
                needs_category.append("Both")
            elif r["wants_insulin"]:
                needs_category.append("Insulin only")
            else:
                needs_category.append("Pump only")
        needs_category = np.array(needs_category)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Diabetic Patient Consumable Acquisition Simulation Results",
                     fontsize=16, fontweight="bold")
        
        # Plot 1: Return Status by Need Category
        categories = ["Insulin only", "Pump only", "Both"]
        colors = {"Insulin only": "skyblue", "Pump only": "orange", "Both": "green"}
        
        ax1 = axes[0, 0]
        ax1.set_title("Return Status by Patient Need")
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["No Return", "Return Needed"])
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels(categories)
        ax1.grid(axis="y", alpha=0.3)
        
        for i, cat in enumerate(categories):
            mask = needs_category == cat
            cat_returns = returns[mask]
            # Jitter y-values
            y_jitter = cat_returns + np.random.normal(0, 0.05, size=len(cat_returns))
            y_jitter = np.clip(y_jitter, -0.2, 1.2)
            # Jitter x-values
            x_jitter = i + np.random.normal(0, 0.1, size=len(cat_returns))
            ax1.scatter(x_jitter, y_jitter, label=cat, alpha=0.6, c=colors[cat])
        ax1.legend()

        # Helper for time plots
        def plot_time_scatter(ax, time_data, title, y_label):
            mask_return = returns == True
            mask_no_return = returns == False
            
            ax.scatter(np.arange(len(total_times))[mask_no_return], 
                      time_data[mask_no_return], 
                      c='blue', alpha=0.5, label='No Return')
            
            ax.scatter(np.arange(len(total_times))[mask_return], 
                      time_data[mask_return], 
                      c='red', marker='x', alpha=0.7, label='Return Needed')
            
            ax.set_title(title)
            ax.set_ylabel(y_label)
            ax.set_xlabel("Patient Index (Chronological Arrival)")
            ax.grid(alpha=0.3)
            ax.legend()

        # Plot 2: Total Time
        plot_time_scatter(axes[0, 1], total_times, "Total Acquisition Time", "Time (days)")
        
        # Plot 3: Pharmacy Time
        plot_time_scatter(axes[1, 0], pharmacy_times, "Pharmacy Time", "Time (days)")
        
        # Plot 4: 1177 Time
        plot_time_scatter(axes[1, 1], platform_times, "1177 Platform Time", "Time (days)")
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join("result", f"simulation_plot_{timestamp}.png")
        os.makedirs("result", exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
        plt.show()


def main():
    """Main function to run the simulation."""
    from .patient import create_patient_population
    
    SEED = None  # Set to a number (e.g., 42) for reproducible results
    
    patients = create_patient_population(seed=SEED)
    model = ConsumableAcquisitionModel(seed=SEED)
    model.run_simulation(patients)
    model.export_to_csv()
    model.plot_results()
    
    print("\n=== Simulation Summary ===")
    print(f"Average total time (days): {model.kpis.get('avg_total_minutes', 0) / (24 * 60):.2f}")
    print(f"Average pharmacy time (days): {model.kpis.get('avg_pharmacy_minutes', 0) / (24 * 60):.2f}")
    print(f"Average 1177 time (days): {model.kpis.get('avg_1177_minutes', 0) / (24 * 60):.2f}")
    print(f"Return rate: {model.kpis.get('return_rate', 0) * 100:.1f}%")


if __name__ == "__main__":
    main()

