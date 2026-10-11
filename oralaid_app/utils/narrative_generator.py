import os

try:
    import anthropic
except ImportError:
    anthropic = None

def _fallback_narrative(patient_name, age, gender, diagnosis_class, confidence, risk_level, tobacco, alcohol, family_history, lesion_area):
    risk_factors = []
    if tobacco == "Yes":
        risk_factors.append("tobacco exposure")
    if alcohol == "Yes":
        risk_factors.append("alcohol use")
    if family_history == "Yes":
        risk_factors.append("a family history of oral cancer")
    risk_text = ", ".join(risk_factors) if risk_factors else "no major recorded tobacco, alcohol, or family history risk factors"

    return (
        f"{patient_name}, a {age}-year-old {gender}, underwent AI-assisted oral lesion assessment. "
        f"The analyzed intraoral image was classified as {diagnosis_class} with {confidence:.1f}% model confidence, "
        f"and the segmented lesion area measured approximately {lesion_area} pixels.\n\n"
        f"The overall clinical risk category is {risk_level}. This interpretation should be considered alongside "
        f"the patient's recorded risk profile, which includes {risk_text}, as well as direct clinical examination findings.\n\n"
        "The recommended management pathway is prompt clinician review, documentation of lesion morphology, and risk-appropriate follow-up. "
        "High-risk or rapidly progressive findings should proceed to specialist referral and biopsy planning where clinically indicated."
    )

def generate_clinical_narrative(patient_name, age, gender, diagnosis_class, confidence, risk_level, tobacco, alcohol, family_history, lesion_area):
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key or anthropic is None:
        return _fallback_narrative(patient_name, age, gender, diagnosis_class, confidence, risk_level, tobacco, alcohol, family_history, lesion_area)

    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""You are a clinical AI assistant writing a professional medical report narrative. 
    Write a 3-paragraph clinical summary for the following case. Use formal medical language. 
    Do not include any disclaimers about being an AI.
    
    Patient: {patient_name}, {age} years old, {gender}
    Diagnosis: {diagnosis_class} ({risk_level} risk)
    Model confidence: {confidence:.1f}%
    Risk factors: Tobacco: {tobacco}, Alcohol: {alcohol}, Family history: {family_history}
    Lesion area (pixels): {lesion_area}
    
    Paragraph 1: Clinical presentation and findings
    Paragraph 2: Risk assessment based on diagnosis and patient risk factors  
    Paragraph 3: Recommended clinical management pathway
    
    Write in third person. Do not use bullet points. Professional tone."""
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Narrative Generation Error: {e}")
        return _fallback_narrative(patient_name, age, gender, diagnosis_class, confidence, risk_level, tobacco, alcohol, family_history, lesion_area)
