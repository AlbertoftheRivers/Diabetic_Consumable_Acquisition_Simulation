"""
Patient generation utilities for the diabetic consumable acquisition simulation.
"""

import math
import random
from dataclasses import dataclass
from typing import List

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WORKDAY_START_MIN = 7 * 60  # 07:00
WORKDAY_END_MIN = 22 * 60  # 22:00


@dataclass
class Patient:
    """Represents a diabetic patient with their attributes, needs and arrival slot."""
    
    id: int
    name: str
    wants_insulin: bool
    wants_pump: bool
    pharmacy_distance_min: float  # Travel time to pharmacy in minutes (10-30)
    pickup_distance_min: float  # Travel time to pickup point in minutes (10-30)
    arrival_day_index: int  # Absolute day index since start of simulation (0=Mon, 1=Tue, 7=Next Mon...)
    arrival_minutes_since_start: float  # Absolute minutes from simulation start (00:00 Day 0)
    return_pharmacy: bool = False  # Flag set by the simulation if a second visit is required
    
    def __post_init__(self):
        """Validate patient attributes."""
        if not (self.wants_insulin or self.wants_pump):
            raise ValueError("Patient must want at least one pharmacy consumable")
        if self.arrival_day_index < 0:
            raise ValueError("Arrival day index must be non-negative")
    
    def get_pharmacy_needs(self) -> List[str]:
        """Returns list of pharmacy consumables the patient needs."""
        needs = []
        if self.wants_insulin:
            needs.append("insulin")
        if self.wants_pump:
            needs.append("pump")
        return needs
    
    @property
    def arrival_day_name(self) -> str:
        """Readable name for the arrival weekday (Monday, Tuesday, etc.)."""
        # Map absolute day index to day name (0 -> Monday, 6 -> Sunday, 7 -> Monday...)
        return DAY_NAMES[self.arrival_day_index % 7]
    
    @property
    def arrival_week_num(self) -> int:
        """Returns 1-based week number."""
        return (self.arrival_day_index // 7) + 1
    
    @property
    def arrival_time_hhmm(self) -> str:
        """Readable clock time (HH:MM) for the arrival slot."""
        # Calculate minutes into the specific day
        minutes_into_day = self.arrival_minutes_since_start - (self.arrival_day_index * 24 * 60)
        hours = int(minutes_into_day // 60)
        minutes = int(minutes_into_day % 60)
        return f"{hours:02d}:{minutes:02d}"


def create_patient_population(num_patients: int = 100, seed: int | None = None) -> List[Patient]:
    """
    Creates a patient population distributed over multiple weeks until N patients are served.
    
    Cyclical weekly flow pattern:
      Monday:    7 patients
      Tuesday:   6 patients
      Wednesday: 5 patients
      Thursday:  4 patients
      Friday:    3 patients
      Saturday:  2 patients
      Sunday:    1 patient
      (Total 28 patients per week)
      
    This pattern repeats until all 100 patients are assigned.
    
    Args:
        num_patients: Total number of simulated patients (default 100).
        seed: Optional random seed for reproducibility.
    
    Returns:
        List with randomized Patient instances.
    """
    if seed is not None:
        random.seed(seed)
    
    # 1. Define Need Profiles (50% insulin, 30% pump, 20% both)
    num_insulin_only = int(num_patients * 0.5)
    num_pump_only = int(num_patients * 0.3)
    num_both = num_patients - num_insulin_only - num_pump_only
    
    need_profiles = (
        [(True, False)] * num_insulin_only +
        [(False, True)] * num_pump_only +
        [(True, True)] * num_both
    )
    random.shuffle(need_profiles)
    
    # 2. Define Daily Flow Pattern
    # Weekday index 0=Mon, 6=Sun
    daily_capacity = [7, 6, 5, 4, 3, 2, 1]  # Mon->7 ... Sun->1
    
    patients: List[Patient] = []
    
    # We assign patients sequentially to days based on the capacity
    current_patient_idx = 0
    day_counter = 0  # Absolute day index (0, 1, 2, ... spanning weeks)
    
    while current_patient_idx < num_patients:
        # Determine capacity for this specific day of the week
        weekday_idx = day_counter % 7
        count_for_today = daily_capacity[weekday_idx]
        
        # Determine how many patients we actually take today (don't exceed total 100)
        remaining = num_patients - current_patient_idx
        count_for_today = min(count_for_today, remaining)
        
        if count_for_today > 0:
            # Time window for today
            working_minutes = WORKDAY_END_MIN - WORKDAY_START_MIN
            interval = working_minutes / count_for_today
            
            for i in range(count_for_today):
                # Grab the next need profile
                wants_insulin, wants_pump = need_profiles[current_patient_idx]
                patient_id = current_patient_idx + 1
                
                # Calculate arrival time
                # Spread evenly starting from 07:00
                minute_offset = i * interval
                arrival_abs_minutes = (day_counter * 24 * 60) + WORKDAY_START_MIN + minute_offset
                
                patients.append(
                    Patient(
                        id=patient_id,
                        name=f"Patient_{patient_id}",
                        wants_insulin=wants_insulin,
                        wants_pump=wants_pump,
                        pharmacy_distance_min=random.uniform(10, 30),
                        pickup_distance_min=random.uniform(10, 30),
                        arrival_day_index=day_counter,
                        arrival_minutes_since_start=arrival_abs_minutes,
                    )
                )
                current_patient_idx += 1
        
        # Move to next day
        day_counter += 1
    
    # We return the list sorted by arrival time naturally, but simulation will sort anyway.
    return patients

