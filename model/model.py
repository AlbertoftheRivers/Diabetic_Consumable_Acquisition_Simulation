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
        self.stock_history: List[Dict] = []  # Track stock over time: {"time": t, "insulin": v, "pump": v}
        self._log_stock(0)
    
    def _log_stock(self, current_time: float):
        """Record current stock levels."""
        self.stock_history.append({
            "time": current_time,
            "insulin": self.stock["insulin"],
            "pump": self.stock["pump"]
        })
    
    def _schedule_restock(self, item: str, current_time: float) -> float:
        """Schedule a new restock for an item if one is not already on the way."""
        delay_days = random.uniform(1, 3)  # Restock takes 1 to 3 days
        eta = current_time + delay_days * 24 * 60
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
        stock_changed = False
        for item in self.stock:
            eta = self.restock_eta[item]
            if eta is not None and current_time >= eta:
                self.stock[item] = self.max_stock[item]
                self.restock_eta[item] = None
                stock_changed = True
                
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
                        # Trigger restocking because we are at 0, which is <= threshold
                        self._schedule_restock(item, current_time)
                        break
                
                # Immediately queue another shipment if we are below the safety threshold
                # This covers both the case where backlog consumed stock, or if we just refilled but
                # had so much backlog we're already low again (though the loop handles the 0 case)
                # But more importantly, if we served backlog and are now at e.g. 10 units (threshold 20),
                # we need to reorder.
                if self.stock[item] <= self.restock_threshold[item] and self.restock_eta[item] is None:
                    self._schedule_restock(item, current_time)
        
        if stock_changed:
            self._log_stock(current_time)
    
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
            
            # Allow consumption if physical stock is sufficient.
            if self.stock[item] >= units_needed:
                self.stock[item] -= units_needed
                items_now.append(item)
                # Check if we crossed the reorder threshold (20%)
                self._maybe_trigger_restock(item, current_time)
            else:
                # Not enough physical stock
                patient.return_pharmacy = True
                items_after_restock.append(item)
                
                # Ensure restock is coming if we are out/low
                if self.restock_eta[item] is None:
                    self._schedule_restock(item, current_time)
                
                wait_time = self._ensure_restock(item, current_time)
                restock_wait_minutes = max(restock_wait_minutes, wait_time)
                self.backlog_units[item].append(units_needed)
        
        self._log_stock(current_time)
        
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
            "arrival_absolute_minutes": arrival_time,
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
            "arrival_absolute_minutes", "wants_insulin", "wants_pump", "return_pharmacy",
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
        pharmacy_times = np.array([r["pharmacy_total_time_minutes"] for r in self.results])
        platform_times = np.array([r["1177_total_time_minutes"] for r in self.results])
        total_times = np.array([r["total_time_minutes"] for r in self.results])
        arrival_absolute = np.array([r.get("arrival_absolute_minutes", 0) for r in self.results])
        returns = np.array([r["return_pharmacy"] for r in self.results])
        
        # Calculate start and end times for Gantt
        start_days = arrival_absolute / (24 * 60)
        durations_days = pharmacy_times / (24 * 60)
        end_days = start_days + durations_days
        
        patient_indices = np.arange(len(self.results)) + 1
        
        # Categorize needs for coloring
        colors = []
        for r in self.results:
            if r["wants_insulin"] and r["wants_pump"]:
                colors.append("green")
            elif r["wants_insulin"]:
                colors.append("skyblue")
            else:
                colors.append("orange")
        
        # Figure 1: The 3 Bar Charts + Calendar/Gantt
        fig1, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig1.suptitle("Diabetic Patient Consumable Acquisition Simulation Results", fontsize=16, fontweight="bold")
        
        # 1) Total Pharmacy Time per Patient (Bar chart, colored by need)
        ax1 = axes[0, 0]
        ax1.bar(patient_indices, pharmacy_times, color=colors, alpha=0.7)
        ax1.set_title("1) Total Pharmacy Time per Patient")
        ax1.set_xlabel("Patient ID")
        ax1.set_ylabel("Time (minutes)")
        
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='skyblue', lw=4),
                        Line2D([0], [0], color='orange', lw=4),
                        Line2D([0], [0], color='green', lw=4)]
        ax1.legend(custom_lines, ['Insulin', 'Pump', 'Both'])
        ax1.grid(axis='y', alpha=0.3)

        # 2) Total 1177 Time per Patient (Bar chart, all blue)
        ax2 = axes[0, 1]
        ax2.bar(patient_indices, platform_times, color='blue', alpha=0.6)
        ax2.set_title("2) Total 1177 Time per Patient")
        ax2.set_xlabel("Patient ID")
        ax2.set_ylabel("Time (minutes)")
        ax2.grid(axis='y', alpha=0.3)

        # 3) Total Time per Patient (Bar chart, colored by need)
        ax3 = axes[1, 0]
        ax3.bar(patient_indices, total_times, color=colors, alpha=0.7)
        ax3.set_title("3) Total Acquisition Time per Patient")
        ax3.set_xlabel("Patient ID")
        ax3.set_ylabel("Time (minutes)")
        ax3.legend(custom_lines, ['Insulin', 'Pump', 'Both'])
        ax3.grid(axis='y', alpha=0.3)

        # 4) Daily Returns Histogram
        ax4 = axes[1, 1]
        
        # We need to count returns per calendar day
        # Each return happens on a specific day. We can use the arrival day or the day the return was triggered.
        # We'll use the arrival_day string from results which maps to the absolute day index
        
        # First, recover absolute day index from arrival_absolute
        # We know day = floor(minutes / (24*60))
        # But results also has arrival_day string (e.g. "Monday").
        # Let's bin by absolute day number to keep the sequence correct (Week 1 Mon, Week 2 Mon...)
        
        day_indices = (arrival_absolute // (24 * 60)).astype(int)
        max_day = int(day_indices.max())
        
        # Count returns per day
        daily_returns = np.zeros(max_day + 1)
        for i, r in enumerate(returns):
            if r:
                day_idx = day_indices[i]
                daily_returns[day_idx] += 1
        
        # X axis: Day 0, 1, 2...
        days = np.arange(max_day + 1)
        ax4.bar(days, daily_returns, color='red', alpha=0.6)
        ax4.set_title("4) Number of Returns to Pharmacy per Day")
        ax4.set_xlabel("Day (Simulation Timeline)")
        ax4.set_ylabel("Count of Returns")
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path1 = os.path.join("result", f"simulation_plot_{timestamp}.png")
        os.makedirs("result", exist_ok=True)
        fig1.savefig(plot_path1, dpi=300, bbox_inches="tight")
        print(f"Main dashboard saved to {plot_path1}")
        
        # Figure 2: Gantt Chart Colored by Need with Arrow/Bar Style + Stock Levels
        fig2, (ax5, ax6) = plt.subplots(2, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [2, 1]})
        fig2.suptitle("Detailed Patient Pharmacy Timeline & Stock Levels", fontsize=16, fontweight="bold")
        
        # Subplot 1: Gantt Chart (Same as before)
        # Y axis = Patient ID
        # X axis = Time (Days)
        
        for i in range(len(self.results)):
            pid = patient_indices[i]
            start = start_days[i]
            duration = durations_days[i]
            color = colors[i]
            is_return = returns[i]
            
            if not is_return:
                # Normal bar for in-stock
                ax5.barh(pid, duration, left=start, height=0.6, color=color, alpha=0.7)
            else:
                # Arrow for out-of-stock
                ax5.arrow(start, pid, duration, 0, 
                         head_width=0.4, head_length=0.1, fc=color, ec=color, 
                         length_includes_head=True, alpha=0.9)
        
        ax5.set_ylabel("Patient ID")
        ax5.grid(axis='x', alpha=0.3)
        # Use same X axis limits for both plots
        max_day_plot = max(np.max(end_days), self.stock_history[-1]["time"]/(24*60)) if self.stock_history else np.max(end_days)
        ax5.set_xlim(0, max_day_plot + 1)
        
        # Legend for Gantt
        legend_elements = [
            Line2D([0], [0], color='skyblue', lw=4, label='Insulin'),
            Line2D([0], [0], color='orange', lw=4, label='Pump'),
            Line2D([0], [0], color='green', lw=4, label='Both'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='In Stock (Bar)'),
            Line2D([0], [0], marker=r'$\rightarrow$', color='gray', markersize=15, label='Stock Out (Arrow)')
        ]
        ax5.legend(handles=legend_elements, loc='upper right')
        
        # Subplot 2: Stock Levels over Time
        if hasattr(self, 'stock_history') and self.stock_history:
            stock_times = [x["time"] / (24 * 60) for x in self.stock_history]
            insulin_levels = [x["insulin"] for x in self.stock_history]
            pump_levels = [x["pump"] for x in self.stock_history]
            
            # Step plot for inventory levels
            ax6.step(stock_times, insulin_levels, where='post', label='Insulin Stock', color='skyblue', lw=2)
            ax6.step(stock_times, pump_levels, where='post', label='Pump Stock', color='orange', lw=2)
            
            # Add threshold lines
            ax6.axhline(y=self.restock_threshold["insulin"], color='skyblue', linestyle='--', alpha=0.5, label='Insulin Reorder Point')
            ax6.axhline(y=self.restock_threshold["pump"], color='orange', linestyle='--', alpha=0.5, label='Pump Reorder Point')
            
            ax6.set_xlabel("Time (Days)")
            ax6.set_ylabel("Units in Stock")
            ax6.set_xlim(0, max_day_plot + 1)
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, "Stock history not available", ha='center')
        
        plot_path2 = os.path.join("result", f"timeline_plot_{timestamp}.png")
        fig2.savefig(plot_path2, dpi=300, bbox_inches="tight")
        print(f"Timeline plot saved to {plot_path2}")
        
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

