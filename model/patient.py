"""
Patient personas for diabetic patient consumable acquisition simulation.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Patient:
    """Represents a diabetic patient with their attributes and needs."""
    
    id: int
    name: str
    wants_insulin: bool
    wants_pump: bool
    pharmacy_distance_min: float  # Travel time to pharmacy in minutes (10-30)
    pickup_distance_min: float  # Travel time to pickup point in minutes (10-30)
    
    def __post_init__(self):
        """Validate patient attributes."""
        if not (self.wants_insulin or self.wants_pump):
            raise ValueError("Patient must want at least one pharmacy consumable")
    
    def get_pharmacy_needs(self) -> List[str]:
        """Returns list of pharmacy consumables the patient needs."""
        needs = []
        if self.wants_insulin:
            needs.append("insulin")
        if self.wants_pump:
            needs.append("pump")
        return needs


def create_patient_personas(seed: int = None) -> List[Tuple[Patient, dict]]:
    """
    Creates 6 patient personas with different scenarios.
    
    Args:
        seed: Optional random seed for reproducibility. If None, uses system time.
    
    Returns:
        List of tuples: (Patient object, pharmacy_stock dict)
        pharmacy_stock format: {"insulin": bool, "pump": bool}
    """
    if seed is not None:
        random.seed(seed)  # For reproducibility when seed is provided
    # Otherwise, use system time (default behavior) for varying results
    
    personas = []
    
    # Scenario 1: Patient wants just insulin, in stock
    p1 = Patient(
        id=1,
        name="Anna",
        wants_insulin=True,
        wants_pump=False,
        pharmacy_distance_min=random.uniform(10, 30),
        pickup_distance_min=random.uniform(10, 30)
    )
    personas.append((p1, {"insulin": True, "pump": True}))
    
    # Scenario 2: Patient wants just insulin, not in stock
    p2 = Patient(
        id=2,
        name="Björn",
        wants_insulin=True,
        wants_pump=False,
        pharmacy_distance_min=random.uniform(10, 30),
        pickup_distance_min=random.uniform(10, 30)
    )
    personas.append((p2, {"insulin": False, "pump": True}))
    
    # Scenario 3: Patient wants just pump, in stock
    p3 = Patient(
        id=3,
        name="Cecilia",
        wants_insulin=False,
        wants_pump=True,
        pharmacy_distance_min=random.uniform(10, 30),
        pickup_distance_min=random.uniform(10, 30)
    )
    personas.append((p3, {"insulin": True, "pump": True}))
    
    # Scenario 4: Patient wants just pump, not in stock
    p4 = Patient(
        id=4,
        name="David",
        wants_insulin=False,
        wants_pump=True,
        pharmacy_distance_min=random.uniform(10, 30),
        pickup_distance_min=random.uniform(10, 30)
    )
    personas.append((p4, {"insulin": True, "pump": False}))
    
    # Scenario 5: Patient wants both, both in stock
    p5 = Patient(
        id=5,
        name="Eva",
        wants_insulin=True,
        wants_pump=True,
        pharmacy_distance_min=random.uniform(10, 30),
        pickup_distance_min=random.uniform(10, 30)
    )
    personas.append((p5, {"insulin": True, "pump": True}))
    
    # Scenario 6: Patient wants both, not in stock
    p6 = Patient(
        id=6,
        name="Fredrik",
        wants_insulin=True,
        wants_pump=True,
        pharmacy_distance_min=random.uniform(10, 30),
        pickup_distance_min=random.uniform(10, 30)
    )
    personas.append((p6, {"insulin": False, "pump": False}))
    
    return personas

