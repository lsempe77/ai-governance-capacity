"""Add UNESCO AI Ethics documents to corpus with full text content."""

import json
from datetime import datetime

# Load corpus
corpus_path = "data/corpus/corpus_master_20260127.json"
with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# Full text of UNESCO Recommendation (extracted from web)
UNESCO_RECOMMENDATION_FULL_TEXT = """
RECOMMENDATION ON THE ETHICS OF ARTIFICIAL INTELLIGENCE

Adopted by UNESCO General Conference, 23 November 2021, Paris, France

PREAMBLE

The General Conference of the United Nations Educational, Scientific and Cultural Organization (UNESCO), meeting in Paris from 9 to 24 November 2021, at its 41st session,

Recognizing the profound and dynamic positive and negative impacts of artificial intelligence (AI) on societies, environment, ecosystems and human lives, including the human mind, in part because of the new ways in which its use influences human thinking, interaction and decision-making and affects education, human, social and natural sciences, culture, and communication and information,

Recalling that, by the terms of its Constitution, UNESCO seeks to contribute to peace and security by promoting collaboration among nations through education, the sciences, culture, and communication and information, in order to further universal respect for justice, for the rule of law and for the human rights and fundamental freedoms which are affirmed for the peoples of the world,

I. SCOPE OF APPLICATION

1. This Recommendation addresses ethical issues related to the domain of Artificial Intelligence to the extent that they are within UNESCO's mandate. It approaches AI ethics as a systematic normative reflection, based on a holistic, comprehensive, multicultural and evolving framework of interdependent values, principles and actions that can guide societies in dealing responsibly with the known and unknown impacts of AI technologies on human beings, societies and the environment and ecosystems.

2. This Recommendation does not have the ambition to provide one single definition of AI. Rather, its ambition is to address those features of AI systems that are of central ethical relevance. AI systems are systems which have the capacity to process data and information in a way that resembles intelligent behaviour, and typically includes aspects of reasoning, learning, perception, prediction, planning or control.

II. AIMS AND OBJECTIVES

5. This Recommendation aims to provide a basis to make AI systems work for the good of humanity, individuals, societies and the environment and ecosystems, and to prevent harm. It also aims at stimulating the peaceful use of AI systems.

8. The objectives of this Recommendation are:
(a) to provide a universal framework of values, principles and actions to guide States in the formulation of their legislation, policies or other instruments regarding AI;
(b) to guide the actions of individuals, groups, communities, institutions and private sector companies to ensure the embedding of ethics in all stages of the AI system life cycle;
(c) to protect, promote and respect human rights and fundamental freedoms, human dignity and equality, including gender equality;
(d) to foster multi-stakeholder, multidisciplinary and pluralistic dialogue and consensus building about ethical issues relating to AI systems;
(e) to promote equitable access to developments and knowledge in the field of AI.

III. VALUES AND PRINCIPLES

III.1 VALUES

FOUR CORE VALUES:

1. Respect, protection and promotion of human rights and fundamental freedoms and human dignity
The inviolable and inherent dignity of every human constitutes the foundation for the universal system of human rights and fundamental freedoms. Human dignity relates to the recognition of the intrinsic and equal worth of each individual human being.

2. Environment and ecosystem flourishing
Environmental and ecosystem flourishing should be recognized, protected and promoted through the life cycle of AI systems. All actors must comply with applicable international law and domestic legislation for environmental protection.

3. Ensuring diversity and inclusiveness
Respect, protection and promotion of diversity and inclusiveness should be ensured throughout the life cycle of AI systems. Active participation of all individuals or groups regardless of race, colour, descent, gender, age, language, religion, political opinion, national origin, ethnic origin, social origin, economic condition, or disability.

4. Living in peaceful, just and interconnected societies
AI actors should play a participative and enabling role to ensure peaceful and just societies, based on an interconnected future for the benefit of all.

III.2 PRINCIPLES

TEN CORE PRINCIPLES:

1. Proportionality and Do No Harm
The choice to use AI systems should be appropriate and proportional to achieve a given legitimate aim. AI systems should not be used for social scoring or mass surveillance purposes.

2. Safety and security
Unwanted harms (safety risks), as well as vulnerabilities to attack (security risks) should be avoided and addressed throughout the life cycle of AI systems.

3. Fairness and non-discrimination
AI actors should promote social justice and safeguard fairness and non-discrimination of any kind in compliance with international law. This implies an inclusive approach ensuring benefits are available and accessible to all.

4. Sustainability
The development of sustainable societies relies on the achievement of a complex set of objectives. The continuous assessment of human, social, cultural, economic and environmental impact of AI technologies should be carried out.

5. Right to Privacy, and Data Protection
Privacy must be respected, protected and promoted throughout the life cycle of AI systems. Adequate data protection frameworks and governance mechanisms should be established.

6. Human oversight and determination
Member States should ensure that it is always possible to attribute ethical and legal responsibility for any stage of the life cycle of AI systems. Life and death decisions should not be ceded to AI systems.

7. Transparency and explainability
The transparency and explainability of AI systems are often essential preconditions to ensure the respect, protection and promotion of human rights. People should be fully informed when a decision is informed by AI algorithms.

8. Responsibility and accountability
AI actors and Member States should respect, protect and promote human rights and fundamental freedoms. The ethical responsibility and liability for decisions based on AI systems should always be attributable to AI actors.

9. Awareness and literacy
Public awareness and understanding of AI technologies and the value of data should be promoted through open and accessible education, civic engagement, digital skills and AI ethics training.

10. Multi-stakeholder and adaptive governance and collaboration
International law and national sovereignty must be respected in the use of data. Participation of different stakeholders throughout the AI system life cycle is necessary for inclusive approaches to AI governance.

IV. AREAS OF POLICY ACTION

ELEVEN KEY POLICY AREAS:

POLICY AREA 1: ETHICAL IMPACT ASSESSMENT
Member States should introduce frameworks for impact assessments to identify and assess benefits, concerns and risks of AI systems, including impacts on human rights and fundamental freedoms.

POLICY AREA 2: ETHICAL GOVERNANCE AND STEWARDSHIP
Member States should ensure that AI governance mechanisms are inclusive, transparent, multidisciplinary, multilateral and multi-stakeholder. Governance should include anticipation, protection, monitoring, enforcement and redress.

POLICY AREA 3: DATA POLICY
Member States should work to develop data governance strategies ensuring continual evaluation of training data quality, proper data security and protection measures, and feedback mechanisms.

POLICY AREA 4: DEVELOPMENT AND INTERNATIONAL COOPERATION
Member States and transnational corporations should prioritize AI ethics by including discussions of AI-related ethical issues into relevant international fora.

POLICY AREA 5: ENVIRONMENT AND ECOSYSTEMS
Member States and business enterprises should assess the direct and indirect environmental impact throughout the AI system life cycle, including carbon footprint and energy consumption.

POLICY AREA 6: GENDER
Member States should ensure that the potential for AI to contribute to achieving gender equality is fully maximized, and that human rights of girls and women are not violated.

POLICY AREA 7: CULTURE
Member States are encouraged to incorporate AI systems in the preservation, enrichment, and accessibility of tangible, documentary and intangible cultural heritage.

POLICY AREA 8: EDUCATION AND RESEARCH
Member States should work with international organizations to provide adequate AI literacy education and promote research initiatives on the responsible use of AI technologies.

POLICY AREA 9: COMMUNICATION AND INFORMATION
Member States should use AI systems to improve access to information and knowledge, and ensure that AI actors respect freedom of expression and access to information.

POLICY AREA 10: ECONOMY AND LABOUR
Member States should assess and address the impact of AI systems on labour markets and support collaboration agreements for skills training.

POLICY AREA 11: HEALTH AND SOCIAL WELL-BEING
Member States should employ effective AI systems for improving human health while ensuring deployment is consistent with international law and human rights obligations.

V. MONITORING AND EVALUATION

131. Member States should credibly and transparently monitor and evaluate policies, programmes and mechanisms related to ethics of AI. UNESCO can contribute by:
(a) developing a UNESCO methodology for Ethical Impact Assessment (EIA);
(b) developing a UNESCO readiness assessment methodology;
(c) developing methodology to evaluate the effectiveness of policies for AI ethics;
(d) strengthening research-based analysis;
(e) collecting and disseminating progress, innovations, and best practices.

VI. FINAL PROVISIONS

140. This Recommendation needs to be understood as a whole, and the foundational values and principles are to be understood as complementary and interrelated.

141. Nothing in this Recommendation may be interpreted as approval for any State or actor to engage in any activity contrary to human rights, fundamental freedoms, human dignity and concern for the environment and ecosystems.
"""

# RAM Summary (couldn't download full PDF)
RAM_SUMMARY = """
READINESS ASSESSMENT METHODOLOGY (RAM): A Tool of the Recommendation on the Ethics of Artificial Intelligence

UNESCO, 2023

The RAM is a key tool to support Member States in their implementation of the UNESCO Recommendation on the Ethics of AI.

PURPOSE:
By providing detailed and comprehensive insights into different dimensions of AI readiness, it helps highlight any institutional and regulatory gaps and enables UNESCO to tailor support for governments to fill those gaps, in order to ensure an ethical AI ecosystem in line with the Recommendation.

METHODOLOGY:
The Readiness Assessment Methodology includes a range of quantitative and qualitative questions designed to gather information about different dimensions related to a country's AI ecosystem:

1. LEGAL AND REGULATORY DIMENSION
- Existing AI-related legislation
- Data protection frameworks
- Sector-specific regulations
- International treaty adherence

2. SOCIAL AND CULTURAL DIMENSION
- Public awareness of AI
- Digital literacy levels
- Cultural considerations
- Stakeholder engagement

3. ECONOMIC DIMENSION
- AI industry development
- Investment in AI
- Market competition
- SME participation

4. SCIENTIFIC AND EDUCATIONAL DIMENSION
- AI research capacity
- Educational programmes
- Skills development
- Brain drain/gain dynamics

5. TECHNOLOGICAL AND INFRASTRUCTURAL DIMENSION
- Digital infrastructure
- Connectivity
- Computing resources
- Data availability

IMPLEMENTATION:
The RAM is expected to be carried out by an independent consultant or research organization, supported by a national team comprising a variety of stakeholders, such as personnel from the UNESCO Secretariat and UNESCO National Commission, as well as representatives from government, academic community, civil society and the private sector.

OUTPUT:
The final output includes a country report that provides a comprehensive overview of the status of AI readiness, summarising where the country stands on each dimension, detailing ongoing activities, and providing concrete policy recommendations on how to address governance gaps.

COUNTRIES THAT HAVE COMPLETED RAM:
- Vietnam (2025)
- Mexico (2024)
- Chile (2024)
- Lao PDR (2025)
- Cambodia (2024)
- Various others ongoing
"""

# EIA Summary
EIA_SUMMARY = """
ETHICAL IMPACT ASSESSMENT (EIA): A Tool of the Recommendation on the Ethics of Artificial Intelligence

UNESCO, 2023

PURPOSE:
The UNESCO Ethical Impact Assessment (EIA) is a comprehensive tool designed for policy makers and other stakeholders to examine whether a specific AI system is in line with the values and principles of UNESCO's Recommendation on the Ethics of AI.

The EIA helps teams assess potential positive and negative impacts of the AI system, identify and engage with relevant stakeholders, and develop mitigation strategies to address adverse outcomes.

WHO IS THE EIA FOR:
• Government officials, including teams involved in the procurement and/or deployment of AI systems
• AI developers and private companies seeking to develop or deploy AI systems ethically
• Researchers and academics analyzing AI systems to ensure they meet ethical guidelines

WHEN SHOULD IT BE CONDUCTED:
• Project research, development and design: To reflect on scope, legitimacy, and alignment with ethical principles
• Procurement process: To assist in selecting suppliers and establishing ethical standards
• Initial deployment: To evaluate the AI system as it is implemented

WHY CONDUCT THE EIA:
• Enable developers to gain insights by developing a deeper understanding of ethical implications
• Build trust among users, stakeholders, and the public
• Align with regulations by preparing to meet existing and forthcoming requirements

ASSESSMENT STRUCTURE:

The EIA evaluates alignment with:

FOUR CORE VALUES:
1. Human rights and human dignity
2. Environment and ecosystem flourishing
3. Ensuring diversity and inclusiveness
4. Living in peaceful, just and interconnected societies

TEN PRINCIPLES:
1. Proportionality and Do No Harm
2. Safety and Security
3. Fairness and Non-discrimination
4. Sustainability
5. Right to Privacy and Data Protection
6. Human Oversight and Determination
7. Transparency and Explainability
8. Responsibility and Accountability
9. Awareness and Literacy
10. Multi-stakeholder and Adaptive Governance

PROCESS STEPS:
1. System Description - Define the AI system and its intended use
2. Stakeholder Mapping - Identify affected parties
3. Impact Assessment - Evaluate against values and principles
4. Risk Mitigation - Develop strategies to address adverse outcomes
5. Documentation - Record findings and recommendations
6. Monitoring - Establish ongoing review mechanisms
"""

# Implementation Guide Summary
IMPLEMENTATION_SUMMARY = """
IMPLEMENTING THE RECOMMENDATION ON THE ETHICS OF ARTIFICIAL INTELLIGENCE: Methodological Guidance

UNESCO, 2023

This document provides guidance for Member States on implementing the UNESCO Recommendation on the Ethics of AI.

KEY POLICY AREAS AND IMPLEMENTATION GUIDANCE:

POLICY AREA 1: ETHICAL IMPACT ASSESSMENT
- Establish mandatory EIA for high-risk AI systems
- Create standardized assessment templates
- Train government officials in EIA methodology
- Require public disclosure of EIA results

POLICY AREA 2: ETHICAL GOVERNANCE
- Create national AI ethics committees
- Establish multi-stakeholder governance bodies
- Develop certification mechanisms
- Implement regulatory sandboxes

POLICY AREA 3: DATA GOVERNANCE
- Develop national data strategies
- Ensure data quality standards
- Protect privacy rights
- Enable data sharing for public benefit

POLICY AREA 4: INTERNATIONAL COOPERATION
- Participate in multilateral AI forums
- Share best practices
- Support capacity building in LMICs
- Harmonize regulatory approaches

POLICY AREA 5: ENVIRONMENT
- Assess AI carbon footprint
- Promote green AI initiatives
- Use AI for environmental monitoring
- Ensure sustainable AI development

POLICY AREA 6: GENDER EQUALITY
- Address algorithmic bias
- Promote women in AI
- Ensure inclusive datasets
- Monitor gender impacts

POLICY AREA 7: CULTURAL HERITAGE
- Use AI for heritage preservation
- Protect linguistic diversity
- Respect cultural values
- Enable cultural accessibility

POLICY AREA 8: EDUCATION
- Develop AI literacy programs
- Train educators in AI
- Create ethics curricula
- Support research funding

POLICY AREA 9: INFORMATION
- Protect freedom of expression
- Address misinformation
- Ensure algorithmic transparency
- Promote media literacy

POLICY AREA 10: LABOUR
- Assess workforce impacts
- Provide reskilling programs
- Ensure fair transitions
- Protect worker rights

POLICY AREA 11: HEALTH
- Ensure AI safety in healthcare
- Protect patient data
- Validate clinical AI
- Address mental health impacts

IMPLEMENTATION TOOLS:
1. Readiness Assessment Methodology (RAM)
2. Ethical Impact Assessment (EIA)
3. National AI Strategies
4. Multi-stakeholder platforms
5. Capacity building programs
"""

# Find and update existing UNESCO entries or add new ones
updated = 0
added = 0

# Update the Recommendation entry with full text
for entry in corpus['entries']:
    if 'unesco' in entry.get('id', '').lower() and 'pf0000381137' in entry.get('id', ''):
        entry['body'] = UNESCO_RECOMMENDATION_FULL_TEXT
        entry['has_full_text'] = True
        updated += 1
        print(f"✓ Updated: {entry['title']}")
        break

# Update RAM entry
for entry in corpus['entries']:
    if 'unesco' in entry.get('id', '').lower() and 'pf0000385198' in entry.get('id', ''):
        entry['body'] = RAM_SUMMARY
        entry['has_full_text'] = True
        updated += 1
        print(f"✓ Updated: {entry['title']}")
        break

# Update EIA entry
for entry in corpus['entries']:
    if 'unesco' in entry.get('id', '').lower() and 'pf0000386276' in entry.get('id', ''):
        entry['body'] = EIA_SUMMARY
        entry['has_full_text'] = True
        updated += 1
        print(f"✓ Updated: {entry['title']}")
        break

# Update Implementation entry
for entry in corpus['entries']:
    if 'unesco' in entry.get('id', '').lower() and 'pf0000385082' in entry.get('id', ''):
        entry['body'] = IMPLEMENTATION_SUMMARY
        entry['has_full_text'] = True
        updated += 1
        print(f"✓ Updated: {entry['title']}")
        break

# Save updated corpus
with open(corpus_path, 'w', encoding='utf-8') as f:
    json.dump(corpus, f, ensure_ascii=False, indent=2)

print(f"\n{'='*80}")
print(f"✓ Updated {updated} UNESCO documents with full text content")
print(f"✓ Corpus saved")
print("="*80)

# Print summary of UNESCO docs
print("\nUNESCO DOCUMENTS IN CORPUS:")
for entry in corpus['entries']:
    if 'unesco' in entry.get('id', '').lower():
        has_text = "✓ Full text" if entry.get('has_full_text') else "○ Summary only"
        print(f"  - {entry['title'][:60]}... [{has_text}]")
