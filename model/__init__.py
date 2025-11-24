"""
Diabetic patient consumable acquisition simulation model.
"""

from .patient import Patient, create_patient_population
from .model import ConsumableAcquisitionModel

__all__ = ['Patient', 'create_patient_population', 'ConsumableAcquisitionModel']

