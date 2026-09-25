"""
Soil Health Intelligence and Soil Health Index (SHI) Engine.

Rule-based agronomic evaluation engine based strictly on ICAR, FAO, and USDA standards.
"""

from soil_health.shi import SoilHealthIndexEngine, get_shi_engine

__all__ = [
    "SoilHealthIndexEngine",
    "get_shi_engine",
]
