ADVISORY_DATABASE = {
    "angular_leaf_spot": {
        "severity": "Moderate to High",
        "symptoms": "Angular, brown-to-grey lesions bounded by leaf veins, often with a yellow halo. "
                     "Lesions may merge on heavily infected leaves, causing premature defoliation.",
        "chemical": "Apply copper-based bactericides/fungicides (e.g., Copper Oxychloride 50 WP) or Mancozeb at the first onset of lesions.",
        "organic": "Spray Neem oil extract (0.5%) or potassium bicarbonate solution. Introduce biocontrol agents like Bacillus subtilis.",
        "prevention": "Ensure wide crop spacing for adequate airflow, avoid overhead sprinkler irrigation, and practice a 2-year crop rotation.",
    },
    "bean_rust": {
        "severity": "High",
        "symptoms": "Small, reddish-brown, powdery pustules on the underside of leaves that can rupture and "
                     "spread spores; heavy infections yellow and drop leaves, reducing yield.",
        "chemical": "Spray triazole-based systemic fungicides such as Tebuconazole or Chlorothalonil before flowering.",
        "organic": "Apply wettable sulfur sprays or certified bio-fungicides (Trichoderma harzianum) on foliage surfaces.",
        "prevention": "Remove and incinerate infected crop residue immediately. Plant certified rust-resistant cultivar seeds.",
    },
    "healthy": {
        "severity": "None",
        "symptoms": "Uniform green coloration, no lesions, pustules, or discoloration detected on the leaf surface.",
        "chemical": "No chemical intervention required.",
        "organic": "Maintain standard soil fertility with balanced organic compost and vermicompost tea.",
        "prevention": "Continue regular field scouting and maintain drip irrigation to prevent foliar dampness.",
    },
}


def get_treatment_plan(disease_name: str) -> dict:
    """Returns the treatment/advisory record for a given disease class label."""
    normalized_key = disease_name.strip().lower().replace(" ", "_")

    if normalized_key in ADVISORY_DATABASE:
        return ADVISORY_DATABASE[normalized_key]

    for key, plan in ADVISORY_DATABASE.items():
        if key in normalized_key or normalized_key in key:
            return plan

    return {
        "severity": "Unknown",
        "symptoms": "Symptom pattern did not match a known class in this model's training scope.",
        "chemical": "Consult local certified agronomy extension office for targeted fungicide guidelines.",
        "organic": "Apply general bio-fungicide foliar sprays (Neem oil/Bacillus subtilis).",
        "prevention": "Quarantine affected area, prune infected leaf matter, and ensure adequate spacing.",
    }
