"""
Scientifically Grounded Soil Health Index (SHI) Engine.

Calculates a transparent, explainable 0–100 Soil Health Index (SHI) based strictly on
established agronomic response standards:
1. ICAR Soil Health Card Scheme & Fertility Guidelines (N, P, K, S, Zn, B, Fe, Mn, Cu)
2. FAO Soils Bulletin 64 (pH Reaction and Nutrient Bioavailability)
3. FAO Soils Bulletin 76 (Soil Organic Matter and Carbon)
4. USDA Agriculture Handbook No. 60 / FAO Paper 39 (Electrical Conductivity & Salinity)
5. Weighted Additive Soil Quality Indexing Framework (Andrews et al., 2002; SMAF)

Every threshold, scoring curve, and weight preserves full academic provenance.
"""

import os
import sys
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class SoilHealthIndexEngine:
    """
    Expert-Derived Rule-Based Soil Health Index (SHI) Scoring Engine.
    """

    # Baseline Weight Distribution (Sum = 1.00)
    DEFAULT_WEIGHTS = {
        "pH": 0.15,
        "EC": 0.10,
        "OC": 0.20,
        "N": 0.15,
        "P": 0.10,
        "K": 0.10,
        "S": 0.04,
        "Zn": 0.04,
        "B": 0.03,
        "Fe": 0.03,
        "Mn": 0.03,
        "Cu": 0.03,
    }

    # Parameter Metadata & Scientific Provenance
    PARAMETER_METADATA = {
        "pH": {
            "name": "Soil pH Reaction",
            "unit": "Standard pH",
            "desirable_range": "6.0 - 7.5",
            "scoring_type": "optimum_range",
            "provenance": "FAO Soils Bulletin 64 / ICAR NBSS&LUP Guidelines",
            "critical_limits": {"lower_optimal": 6.0, "upper_optimal": 7.5, "extreme_acid": 5.0, "extreme_alkaline": 8.5},
        },
        "EC": {
            "name": "Electrical Conductivity",
            "unit": "dS/m",
            "desirable_range": "< 0.8 dS/m (Non-saline)",
            "scoring_type": "less_is_better",
            "provenance": "USDA Handbook 60 / FAO Irrigation & Drainage Paper 39",
            "critical_limits": {"non_saline": 0.8, "slightly_saline": 1.6, "severely_saline": 3.0},
        },
        "OC": {
            "name": "Soil Organic Carbon",
            "unit": "%",
            "desirable_range": ">= 0.75%",
            "scoring_type": "more_is_better_plateau",
            "provenance": "FAO Soils Bulletin 76 / ICAR Rating Scale",
            "critical_limits": {"low": 0.50, "medium": 0.75, "optimal": 1.0},
        },
        "N": {
            "name": "Available Nitrogen",
            "unit": "kg/ha",
            "desirable_range": "280 - 560 kg/ha",
            "scoring_type": "optimum_range_with_excess_penalty",
            "provenance": "ICAR Soil Health Card Scheme Norms",
            "critical_limits": {"low": 280.0, "optimal": 560.0, "excess": 700.0},
        },
        "P": {
            "name": "Available Phosphorus",
            "unit": "kg/ha",
            "desirable_range": "10 - 25 kg/ha",
            "scoring_type": "optimum_range_with_excess_penalty",
            "provenance": "ICAR Soil Health Card Norms / FAO Bulletin 64",
            "critical_limits": {"low": 10.0, "optimal": 25.0, "excess": 50.0},
        },
        "K": {
            "name": "Available Potassium",
            "unit": "kg/ha",
            "desirable_range": "110 - 280 kg/ha",
            "scoring_type": "optimum_range_with_excess_penalty",
            "provenance": "ICAR Soil Health Card Norms / IFA Guidelines",
            "critical_limits": {"low": 110.0, "optimal": 280.0, "excess": 500.0},
        },
        "S": {
            "name": "Available Sulphur",
            "unit": "ppm",
            "desirable_range": ">= 10.0 ppm",
            "scoring_type": "more_is_better_plateau",
            "provenance": "ICAR Micronutrient & Secondary Nutrient Compendium",
            "critical_limits": {"critical": 10.0},
        },
        "Zn": {
            "name": "Available Zinc",
            "unit": "ppm",
            "desirable_range": ">= 0.60 ppm",
            "scoring_type": "more_is_better_plateau",
            "provenance": "ICAR AICRP on Micro & Secondary Nutrients (DTPA)",
            "critical_limits": {"critical": 0.60},
        },
        "B": {
            "name": "Available Boron",
            "unit": "ppm",
            "desirable_range": ">= 0.50 ppm",
            "scoring_type": "more_is_better_plateau",
            "provenance": "ICAR AICRP on Micro & Secondary Nutrients (Hot Water)",
            "critical_limits": {"critical": 0.50},
        },
        "Fe": {
            "name": "Available Iron",
            "unit": "ppm",
            "desirable_range": ">= 4.50 ppm",
            "scoring_type": "more_is_better_plateau",
            "provenance": "ICAR AICRP on Micro & Secondary Nutrients (DTPA)",
            "critical_limits": {"critical": 4.50},
        },
        "Mn": {
            "name": "Available Manganese",
            "unit": "ppm",
            "desirable_range": ">= 2.00 ppm",
            "scoring_type": "more_is_better_plateau",
            "provenance": "ICAR AICRP on Micro & Secondary Nutrients (DTPA)",
            "critical_limits": {"critical": 2.00},
        },
        "Cu": {
            "name": "Available Copper",
            "unit": "ppm",
            "desirable_range": ">= 0.20 ppm",
            "scoring_type": "more_is_better_plateau",
            "provenance": "ICAR AICRP on Micro & Secondary Nutrients (DTPA)",
            "critical_limits": {"critical": 0.20},
        },
    }

    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        if custom_weights:
            # Validate custom weights
            total = sum(custom_weights.values())
            if not np.isclose(total, 1.0, atol=1e-3):
                raise ValueError(f"Custom weights must sum to 1.0 (got {total:.4f})")
            self.weights = dict(custom_weights)
        else:
            self.weights = dict(self.DEFAULT_WEIGHTS)

    # --------------------------------------------------------------------------
    # Individual Parameter Scoring Functions S_i(x_i) in [0, 100]
    # --------------------------------------------------------------------------
    def score_ph(self, val: float) -> Tuple[float, str]:
        """Scores Soil pH Reaction (FAO Bulletin 64 / ICAR)."""
        if pd.isna(val) or val < 0 or val > 14:
            return 0.0, "Invalid / Out-of-Range pH"

        if 6.0 <= val <= 7.5:
            return 100.0, "Optimal Neutral Range"
        elif val < 6.0:
            if val < 5.0:
                score = max(0.0, 100.0 - ((6.0 - val) / 1.5) ** 2 * 80.0)
                return score, "Strongly Acidic (Severe P-Fixation & Al Toxicity Risk)"
            else:
                score = 100.0 - ((6.0 - val) / 1.0) * 40.0
                return max(0.0, score), "Moderately Acidic"
        else: # val > 7.5
            if val > 8.5:
                score = max(0.0, 100.0 - ((val - 7.5) / 1.5) ** 2 * 80.0)
                return score, "Strongly Alkaline / Sodic (Micronutrient Precipitation Risk)"
            else:
                score = 100.0 - ((val - 7.5) / 1.0) * 40.0
                return max(0.0, score), "Moderately Alkaline / Calcareous"

    def score_ec(self, val: float) -> Tuple[float, str]:
        """Scores Electrical Conductivity / Salinity (USDA Handbook 60)."""
        if pd.isna(val) or val < 0:
            return 0.0, "Invalid / Negative EC"

        if val <= 0.8:
            return 100.0, "Optimal Non-Saline"
        elif val <= 1.6:
            score = 100.0 - ((val - 0.8) / 0.8) * 40.0
            return score, "Slightly Saline (Sensitive Crop Yield Impact)"
        elif val <= 3.0:
            score = 60.0 - ((val - 1.6) / 1.4) * 40.0
            return max(0.0, score), "Moderately Saline (Osmotic Water Stress)"
        else:
            score = max(0.0, 20.0 - ((val - 3.0) / 2.0) * 20.0)
            return score, "Severely Saline (Severe Osmotic Drought)"

    def score_oc(self, val: float) -> Tuple[float, str]:
        """Scores Soil Organic Carbon (FAO Bulletin 76 / ICAR)."""
        if pd.isna(val) or val < 0:
            return 0.0, "Invalid / Negative OC"

        if val >= 0.75:
            # Optimal biological activity plateau
            score = min(100.0, 90.0 + min(10.0, (val - 0.75) * 10.0))
            return score, "High / Optimal Biological Activity"
        elif val >= 0.50:
            score = 60.0 + ((val - 0.50) / 0.25) * 30.0
            return score, "Medium / Moderate Organic Matter"
        else:
            score = (val / 0.50) * 60.0
            return max(0.0, score), "Critically Low Organic Carbon (Degraded Microbiome)"

    def score_nitrogen(self, val: float) -> Tuple[float, str]:
        """Scores Available Nitrogen (ICAR Norms: 280-560 kg/ha)."""
        if pd.isna(val) or val < 0:
            return 0.0, "Invalid / Negative Nitrogen"

        if val < 280.0:
            score = (val / 280.0) * 70.0
            return max(0.0, score), "Deficient / Low Nitrogen (< 280 kg/ha)"
        elif val <= 560.0:
            score = 70.0 + ((val - 280.0) / 280.0) * 30.0
            return score, "Medium / Adequate Nitrogen"
        else:
            # Mild excess penalty for lodging & nitrate leaching
            score = max(70.0, 100.0 - ((val - 560.0) / 280.0) * 25.0)
            return score, "High / Excessive Nitrogen (> 560 kg/ha)"

    def score_phosphorus(self, val: float) -> Tuple[float, str]:
        """Scores Available Phosphorus (ICAR Norms: 10-25 kg/ha)."""
        if pd.isna(val) or val < 0:
            return 0.0, "Invalid / Negative Phosphorus"

        if val < 10.0:
            score = (val / 10.0) * 60.0
            return max(0.0, score), "Deficient / Low Phosphorus (< 10 kg/ha)"
        elif val <= 25.0:
            score = 60.0 + ((val - 10.0) / 15.0) * 40.0
            return score, "Medium / Adequate Phosphorus"
        else:
            score = max(75.0, 100.0 - ((val - 25.0) / 25.0) * 20.0)
            return score, "High / Excessive Phosphorus (Zinc Antagonism Risk)"

    def score_potassium(self, val: float) -> Tuple[float, str]:
        """Scores Available Potassium (ICAR Norms: 110-280 kg/ha)."""
        if pd.isna(val) or val < 0:
            return 0.0, "Invalid / Negative Potassium"

        if val < 110.0:
            score = (val / 110.0) * 60.0
            return max(0.0, score), "Deficient / Low Potassium (< 110 kg/ha)"
        elif val <= 280.0:
            score = 60.0 + ((val - 110.0) / 170.0) * 40.0
            return score, "Medium / Adequate Potassium"
        else:
            score = max(80.0, 100.0 - ((val - 280.0) / 280.0) * 15.0)
            return score, "High / Abundant Potassium"

    def score_micronutrient(
        self, param: str, val: float, critical_limit: float
    ) -> Tuple[float, str]:
        """Scores secondary/micronutrients against ICAR critical limits."""
        if pd.isna(val) or val < 0:
            return 0.0, f"Invalid / Negative {param}"

        if val >= critical_limit:
            return 100.0, f"Adequate {param} (>= {critical_limit:.2f} ppm)"
        else:
            score = (val / critical_limit) * 80.0
            return max(0.0, score), f"Deficient {param} (< {critical_limit:.2f} ppm critical limit)"

    # --------------------------------------------------------------------------
    # Composite Parameter Scoring & SHI Aggregation
    # --------------------------------------------------------------------------
    def calculate_parameter_scores(
        self, sample: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculates individual 0-100 scores, status, and provenance for all available parameters.
        """
        results = {}

        # 1. pH
        if "pH" in sample and pd.notnull(sample["pH"]):
            val_ph = float(sample["pH"])
            score, status = self.score_ph(val_ph)
            results["pH"] = {
                "raw_value": val_ph,
                "score": round(score, 2),
                "status": status,
                "is_deficient": val_ph < 6.0 or val_ph > 7.5,
                "metadata": self.PARAMETER_METADATA["pH"],
            }

        # 2. EC
        if "EC" in sample and pd.notnull(sample["EC"]):
            val_ec = float(sample["EC"])
            score, status = self.score_ec(val_ec)
            results["EC"] = {
                "raw_value": val_ec,
                "score": round(score, 2),
                "status": status,
                "is_deficient": val_ec > 0.8,
                "metadata": self.PARAMETER_METADATA["EC"],
            }

        # 3. OC
        if "OC" in sample and pd.notnull(sample["OC"]):
            val_oc = float(sample["OC"])
            score, status = self.score_oc(val_oc)
            results["OC"] = {
                "raw_value": val_oc,
                "score": round(score, 2),
                "status": status,
                "is_deficient": val_oc < 0.75,
                "metadata": self.PARAMETER_METADATA["OC"],
            }

        # 4. Nitrogen
        if "N" in sample and pd.notnull(sample["N"]):
            score, status = self.score_nitrogen(float(sample["N"]))
            results["N"] = {
                "raw_value": float(sample["N"]),
                "score": round(score, 2),
                "status": status,
                "is_deficient": score < 70.0,
                "metadata": self.PARAMETER_METADATA["N"],
            }

        # 5. Phosphorus
        if "P" in sample and pd.notnull(sample["P"]):
            score, status = self.score_phosphorus(float(sample["P"]))
            results["P"] = {
                "raw_value": float(sample["P"]),
                "score": round(score, 2),
                "status": status,
                "is_deficient": score < 60.0,
                "metadata": self.PARAMETER_METADATA["P"],
            }

        # 6. Potassium
        if "K" in sample and pd.notnull(sample["K"]):
            score, status = self.score_potassium(float(sample["K"]))
            results["K"] = {
                "raw_value": float(sample["K"]),
                "score": round(score, 2),
                "status": status,
                "is_deficient": score < 60.0,
                "metadata": self.PARAMETER_METADATA["K"],
            }

        # 7. Secondary & Micronutrients
        micro_params = ["S", "Zn", "B", "Fe", "Mn", "Cu"]
        for p in micro_params:
            if p in sample and pd.notnull(sample[p]):
                crit = self.PARAMETER_METADATA[p]["critical_limits"]["critical"]
                score, status = self.score_micronutrient(p, float(sample[p]), crit)
                results[p] = {
                    "raw_value": float(sample[p]),
                    "score": round(score, 2),
                    "status": status,
                    "is_deficient": score < 80.0,
                    "metadata": self.PARAMETER_METADATA[p],
                }

        return results

    def calculate_shi(self, sample: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculates composite 0-100 SHI, dynamic weights, category, and deficiencies.
        """
        param_scores = self.calculate_parameter_scores(sample)

        if not param_scores:
            return {
                "shi_score": 0.0,
                "category": "Insufficient Data",
                "parameter_scores": {},
                "weighted_contributions": {},
                "deficiencies": ["No valid soil chemistry parameters provided."],
                "warnings": ["Insufficient telemetry/chemistry parameters to compute SHI."],
            }

        # Dynamic weight re-normalization over available parameters
        available_keys = list(param_scores.keys())
        raw_weights = {k: self.weights.get(k, 1.0 / len(available_keys)) for k in available_keys}
        weight_sum = sum(raw_weights.values())

        norm_weights = {k: w / weight_sum for k, w in raw_weights.items()}

        # Compute Weighted Additive SHI: sum(w_i * S_i)
        weighted_contribs = {}
        total_shi = 0.0
        deficiencies = []

        for k, res in param_scores.items():
            w = norm_weights[k]
            s = res["score"]
            contrib = w * s
            weighted_contribs[k] = round(contrib, 2)
            total_shi += contrib

            if res["is_deficient"]:
                deficiencies.append(f"{res['metadata']['name']} ({k}): {res['status']}")

        total_shi = round(float(np.clip(total_shi, 0.0, 100.0)), 2)

        # Categorization (Andrews et al., 2002 / ICAR framework)
        if total_shi >= 80.0:
            category = "High / Optimal Soil Health"
        elif total_shi >= 65.0:
            category = "Good / Moderate Soil Health"
        elif total_shi >= 50.0:
            category = "Fair / Low Soil Health"
        else:
            category = "Degraded / Critical Soil Health"

        # Limitations & notes
        warnings = []
        if len(available_keys) < len(self.DEFAULT_WEIGHTS):
            missing_count = len(self.DEFAULT_WEIGHTS) - len(available_keys)
            warnings.append(
                f"Computed over partial parameter set ({len(available_keys)}/{len(self.DEFAULT_WEIGHTS)} channels). "
                f"Weights were re-normalized dynamically to maintain sum(w) = 1.0."
            )

        return {
            "shi_score": total_shi,
            "category": category,
            "parameter_scores": {k: v["score"] for k, v in param_scores.items()},
            "normalized_weights": {k: round(v, 4) for k, v in norm_weights.items()},
            "weighted_contributions": weighted_contribs,
            "deficiencies": deficiencies,
            "warnings": warnings,
            "provenance_summary": "ICAR Soil Health Card Standards / FAO Bulletins 64 & 76 / USDA Handbook 60",
        }

    def explain_shi(self, sample: Dict[str, float]) -> Dict[str, Any]:
        """
        Generates detailed multi-tiered agronomic interpretation and actionable guidance triggers.
        """
        shi_res = self.calculate_shi(sample)
        param_scores = self.calculate_parameter_scores(sample)

        # Separate positive anchors vs stress limiting factors
        strengths = []
        limitations = []

        for k, res in param_scores.items():
            name = res["metadata"]["name"]
            score = res["score"]
            val = res["raw_value"]
            unit = res["metadata"]["unit"]

            if score >= 90.0:
                strengths.append(f"{name} ({val:.2f} {unit}): Score {score:.1f}/100 — {res['status']}")
            elif score < 65.0:
                limitations.append(f"{name} ({val:.2f} {unit}): Score {score:.1f}/100 — {res['status']}")

        summary_statement = (
            f"Overall Soil Health Index is {shi_res['shi_score']:.1f}/100 ({shi_res['category']}). "
        )
        if limitations:
            summary_statement += f"Primary limiting factors: {'; '.join(limitations[:3])}. "
        if strengths:
            summary_statement += f"Key soil strengths: {'; '.join(strengths[:3])}."

        return {
            "overall_evaluation": shi_res,
            "strengths": strengths,
            "limiting_factors": limitations,
            "agronomic_summary": summary_statement,
            "parameter_details": param_scores,
        }

    def calculate_shi_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates SHI and parameter scores across a DataFrame of soil samples.
        """
        results_df = df.copy()
        shi_scores = []
        categories = []
        deficient_counts = []

        # Parameter columns to score
        score_cols = [c for c in self.DEFAULT_WEIGHTS.keys() if c in df.columns]
        for col in score_cols:
            results_df[f"Score_{col}"] = np.nan

        for idx, row in df.iterrows():
            sample_dict = row.to_dict()
            res = self.calculate_shi(sample_dict)
            param_res = self.calculate_parameter_scores(sample_dict)

            shi_scores.append(res["shi_score"])
            categories.append(res["category"])
            deficient_counts.append(len(res["deficiencies"]))

            for col, p_data in param_res.items():
                results_df.loc[idx, f"Score_{col}"] = p_data["score"]

        results_df["SHI_Score"] = shi_scores
        results_df["SHI_Category"] = categories
        results_df["Deficiencies_Count"] = deficient_counts

        return results_df


def get_shi_engine(custom_weights: Optional[Dict[str, float]] = None) -> SoilHealthIndexEngine:
    """Helper factory for SoilHealthIndexEngine."""
    return SoilHealthIndexEngine(custom_weights=custom_weights)


if __name__ == "__main__":
    engine = get_shi_engine()
    sample_bareilly = {
        "pH": 7.13,
        "EC": 0.13,
        "OC": 0.23,
        "N": 108.0,
        "P": 11.59,
        "K": 130.0,
        "S": 14.67,
        "Zn": 0.98,
        "B": 0.60,
        "Fe": 8.06,
        "Mn": 2.03,
        "Cu": 0.47,
    }
    shi_out = engine.calculate_shi(sample_bareilly)
    print("--- Sample Soil Health Index Evaluation ---")
    print(f"SHI Score: {shi_out['shi_score']} / 100 ({shi_out['category']})")
    print("Deficiencies:", shi_out["deficiencies"])
    print("Weighted Contributions:", shi_out["weighted_contributions"])
