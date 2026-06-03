# integrators/base.py
from abc import ABC, abstractmethod

class BaseIntegrator(ABC):
    
    @abstractmethod
    def step(self):
        """Effectue un pas de simulation"""
        ...
    
    @abstractmethod
    def get_positions(self):
        """Retourne les positions actuelles"""
        ...