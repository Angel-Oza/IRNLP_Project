"""
Domain-Grounded Agronomic Recommendation Synthesizer Templates.

Defines deterministic, evidence-backed remediation protocols structured directly from
ICAR Soil Health Card guidelines, FAO Soils Bulletins (64, 76), and USDA Handbooks.
"""

from typing import Dict, List, Any, Optional


class AgronomicRemediationTemplates:
    """
    Standard remediation knowledge mapping soil chemical/physical stresses to verified actions.
    """

    @staticmethod
    def get_remediation_spec(param: str, value: float, score: float) -> Optional[Dict[str, str]]:
        """
        Returns remediation action, rationale, urgency, and RAG search query for a deficient parameter.
        """
        # 1. Soil Organic Carbon (OC)
        if param == "OC" and value < 0.50:
            return {
                "topic": "Soil Organic Carbon & Microbiome Restoration",
                "issue_detected": f"Critically low Soil Organic Carbon ({value:.2f}% vs. optimal >= 0.75%)",
                "action": (
                    "Apply 5.0 to 10.0 tonnes/ha of well-decomposed Farmyard Manure (FYM) or vermicompost annually. "
                    "Incorporate green manure cover crops (Sesbania / Sunnhemp at 45 days) and retain 30% crop residues."
                ),
                "rationale": (
                    "Restores the active biological carbon pool, enhances water-holding capacity, "
                    "rebuilds Cation Exchange Capacity (CEC), and stimulates beneficial soil microbiome."
                ),
                "urgency": "HIGH",
                "query": "Soil organic carbon restoration, farmyard manure application, green manuring, and biological health.",
            }

        # 2. Available Nitrogen (N)
        if param == "N" and value < 280.0:
            urgency = "HIGH" if value < 150.0 else "MEDIUM"
            return {
                "topic": "Primary Nitrogen Nutrient Management",
                "issue_detected": f"Deficient available Nitrogen ({value:.1f} kg/ha vs. ICAR medium range 280-560 kg/ha)",
                "action": (
                    "Implement 4R split Nitrogen application: 50% basal dose, 25% at active tillering/vegetative phase, "
                    "and 25% at panicle/flower initiation. Supplement with neem-coated urea and biofertilizers (Azotobacter/Rhizobium)."
                ),
                "rationale": (
                    "Split application prevents nitrate leaching and ammonia volatilization while providing continuous "
                    "nitrogen supply for chlorophyll synthesis, vegetative biomass, and tillering."
                ),
                "urgency": urgency,
                "query": "Nitrogen deficiency correction, 4R split fertilizer application, and neem coated urea management.",
            }

        # 3. Available Phosphorus (P)
        if param == "P" and value < 10.0:
            return {
                "topic": "Phosphorus Fertility & Root Development",
                "issue_detected": f"Low available Phosphorus ({value:.2f} kg/ha vs. ICAR medium range 10-25 kg/ha)",
                "action": (
                    "Apply Single Super Phosphate (SSP) or DAP as a basal band placement 5 cm beside and below the seed row. "
                    "Inoculate seeds with Phosphate Solubilizing Bacteria (PSB)."
                ),
                "rationale": (
                    "Band placement minimizes contact with soil fixing agents (Fe/Al/Ca), enhancing early root proliferation, "
                    "ATP energy metabolism, and grain development."
                ),
                "urgency": "HIGH" if value < 6.0 else "MEDIUM",
                "query": "Phosphorus deficiency, band placement of phosphatic fertilizers, and phosphate solubilizing bacteria.",
            }

        # 4. Available Potassium (K)
        if param == "K" and value < 110.0:
            return {
                "topic": "Potassium Nutrition & Crop Stalk Strength",
                "issue_detected": f"Sub-optimal Potassium ({value:.1f} kg/ha vs. ICAR medium range 110-280 kg/ha)",
                "action": (
                    "Apply Muriate of Potash (MOP, 60% K2O) as a basal dressing, with supplemental top-dressing at flowering "
                    "in light/sandy soil textures."
                ),
                "rationale": (
                    "Activates metabolic enzymes, strengthens plant stalks against lodging, regulates stomatal opening, "
                    "and mitigates moisture stress vulnerability."
                ),
                "urgency": "MEDIUM",
                "query": "Potassium deficiency symptoms, muriate of potash fertilization, and drought tolerance dynamics.",
            }

        # 5. Soil pH Reaction (Acidity / Alkalinity)
        if param == "pH":
            if value < 6.0:
                return {
                    "topic": "Soil Acidity Remediation & Liming",
                    "issue_detected": f"Acidic soil reaction (pH {value:.2f} vs. neutral 6.0-7.5)",
                    "action": (
                        "Incorporate agricultural limestone (CaCO3) or dolomitic lime at 2.0 to 4.0 tonnes/ha uniformly "
                        "into the top 15 cm of soil 4 to 6 weeks prior to planting."
                    ),
                    "rationale": (
                        "Neutralizes exchangeable aluminum (Al3+) and manganese toxicity while unfixing phosphorus and "
                        "restoring microbial nitrification."
                    ),
                    "urgency": "HIGH",
                    "query": "Acidic soil remediation, agricultural liming protocols, and phosphorus availability.",
                }
            elif value > 7.5:
                return {
                    "topic": "Alkaline / Calcareous Soil Management",
                    "issue_detected": f"Alkaline soil reaction (pH {value:.2f} vs. neutral 6.0-7.5)",
                    "action": (
                        "Incorporate agricultural gypsum (CaSO4.2H2O) or elemental sulphur (0.5-1.0 t/ha), and apply organic "
                        "composts to buffer rhizosphere pH. Use chelated micronutrients (Fe-EDDHA / Zn-EDTA)."
                    ),
                    "rationale": (
                        "Releases organic/sulphuric acids to mobilize insoluble calcium phosphates and prevent micronutrient "
                        "hydroxide precipitation."
                    ),
                    "urgency": "MEDIUM",
                    "query": "Alkaline soil management, gypsum application, elemental sulphur, and micronutrient bioavailability.",
                }

        # 6. Electrical Conductivity (EC / Salinity)
        if param == "EC" and value > 0.8:
            return {
                "topic": "Soil Salinity & Osmotic Stress Management",
                "issue_detected": f"Elevated Electrical Conductivity ({value:.2f} dS/m vs. non-saline < 0.8 dS/m)",
                "action": (
                    "Apply leaching irrigation with good-quality water to flush soluble salts below the root zone. "
                    "Ensure adequate subsurface drainage and apply surface organic mulch to prevent upward capillary salinization."
                ),
                "rationale": (
                    "Reduces soil osmotic drought pressure and prevents toxic sodium (Na+) and chloride (Cl-) leaf accumulation."
                ),
                "urgency": "HIGH" if value > 1.6 else "MEDIUM",
                "query": "Soil salinity electrical conductivity management, leaching requirement, and osmotic stress remediation.",
            }

        # 7. Micronutrients
        if param == "S" and value < 10.0:
            return {
                "topic": "Sulphur Secondary Nutrient Correction",
                "issue_detected": f"Low available Sulphur ({value:.2f} ppm vs. critical limit 10.0 ppm)",
                "action": "Apply Agricultural Gypsum (200-250 kg/ha) or Single Super Phosphate (SSP containing 12% S) at sowing.",
                "rationale": "Essential for sulphur-containing amino acids (methionine, cysteine) and oil synthesis in oilseeds and pulses.",
                "urgency": "MEDIUM",
                "query": "Sulphur deficiency in soil, gypsum application, and oilseed crop nutrition.",
            }

        if param == "Zn" and value < 0.60:
            return {
                "topic": "Zinc Micronutrient Nutrition",
                "issue_detected": f"Deficient available Zinc ({value:.2f} ppm vs. critical limit 0.60 ppm)",
                "action": "Soil apply Zinc Sulphate heptahydrate (ZnSO4.7H2O) at 25 kg/ha, or foliar spray 0.5% ZnSO4 + 0.25% lime.",
                "rationale": "Prevents Khaira disease in rice, white bud in maize, and restores auxin phytohormone biosynthesis.",
                "urgency": "HIGH",
                "query": "Zinc deficiency in soil, Khaira disease, and zinc sulphate foliar remediation.",
            }

        if param == "B" and value < 0.50:
            return {
                "topic": "Boron Micronutrient Nutrition",
                "issue_detected": f"Deficient available Boron ({value:.2f} ppm vs. critical limit 0.50 ppm)",
                "action": "Soil apply Borax (10 kg/ha) or foliar spray 0.2% Solubor at flower initiation.",
                "rationale": "Ensures pollen viability, grain setting, and cell wall structural integrity.",
                "urgency": "MEDIUM",
                "query": "Boron deficiency in soil, borax soil application, and pollen viability.",
            }

        if param == "Fe" and value < 4.50:
            return {
                "topic": "Iron Micronutrient Nutrition",
                "issue_detected": f"Deficient available Iron ({value:.2f} ppm vs. critical limit 4.50 ppm)",
                "action": "Foliar spray 1.0% Ferrous Sulphate (FeSO4) + 0.1% citric acid, or soil apply Fe-EDDHA chelate.",
                "rationale": "Corrects interveinal chlorosis on young leaves and restores chlorophyll biosynthesis.",
                "urgency": "MEDIUM",
                "query": "Iron deficiency in soil, lime induced chlorosis, and ferrous sulphate foliar application.",
            }

        if param == "Mn" and value < 2.00:
            return {
                "topic": "Manganese Micronutrient Nutrition",
                "issue_detected": f"Deficient available Manganese ({value:.2f} ppm vs. critical limit 2.00 ppm)",
                "action": "Foliar spray 0.5% Manganese Sulphate (MnSO4) at active tillering.",
                "rationale": "Supports photosynthetic water-splitting in photosystem II and nitrogen assimilation.",
                "urgency": "LOW",
                "query": "Manganese deficiency in soil and manganese sulphate foliar correction.",
            }

        if param == "Cu" and value < 0.20:
            return {
                "topic": "Copper Micronutrient Nutrition",
                "issue_detected": f"Deficient available Copper ({value:.2f} ppm vs. critical limit 0.20 ppm)",
                "action": "Soil apply Copper Sulphate (CuSO4.5H2O) at 5.0 kg/ha or foliar spray 0.2% CuSO4.",
                "rationale": "Essential for plastocyanin electron transport in photosynthesis and lignin synthesis.",
                "urgency": "LOW",
                "query": "Copper deficiency in soil and copper sulphate remediation.",
            }

        return None
