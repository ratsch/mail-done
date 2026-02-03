# Application Evaluation Examples

This file contains calibration examples for the application evaluation system.
These examples help the LLM understand the full scoring range from weak to exceptional candidates.

## Example 1: Strong US Candidate (8/9/8)

**Profile**: Sarah Chen, Stanford M.Sc., single-cell genomics expert

```json
{
  "category": "application-phd",
  "confidence": 0.92,
  "reasoning": "PhD application with strong ML and genomics background",
  "urgency": "normal",
  "urgency_score": 5,
  "urgency_reason": "Standard application timeline",
  "summary": "PhD application from Stanford M.Sc. student with single-cell genomics expertise",
  "action_items": null,
  "needs_reply": true,
  "reply_deadline": null,
  "reply_suggestion": "Acknowledge and request CV for full evaluation",
  "sentiment": "positive",
  "applicant_name": "Sarah Chen",
  "applicant_institution": "Stanford University",
  "nationality": "USA",
  "highest_degree_completed": "M.Sc.",
  "current_situation": "Student enrolled at a non-European university",
  "recent_thesis_title": "Deep Learning for Single-Cell RNA-seq Clustering",
  "recommendation_source": "Prof. Michael Zhang",
  "github_account": "sarahchen92",
  "linkedin_account": null,
  "google_scholar_account": "S_Chen_Stanford",
  "coding_experience": {"score": 9, "evidence": "Mentions 5 years Python/R, contributed to scikit-learn"},
  "omics_genomics_experience": {"score": 8, "evidence": "Thesis on single-cell RNA-seq, published in Bioinformatics"},
  "medical_data_experience": {"score": 0, "evidence": "Not mentioned in email"},
  "sequence_analysis_algorithms_experience": {"score": 6, "evidence": "Implemented custom alignment tool for scRNA-seq"},
  "image_analysis_experience": {"score": 0, "evidence": "Not mentioned in email"},
  "profile_tags": [
    {"tag": "single_cell_omics", "confidence": 0.95, "reason": "Thesis directly on scRNA-seq clustering"},
    {"tag": "deep_learning", "confidence": 0.85, "reason": "Applied transformers to genomic data"}
  ],
  "is_mass_email": false,
  "no_research_background": false,
  "irrelevant_field": false,
  "possible_spam": false,
  "is_followup": false,
  "is_cold_email": false,
  "scientific_excellence_score": 8,
  "scientific_excellence_reason": "Top-tier university (Stanford), strong publication in Bioinformatics journal, recommendation from known professor. GPA equivalent would be ~5.5 at ETH.",
  "research_fit_score": 9,
  "research_fit_reason": "Perfect alignment with single-cell omics focus, mentions reading the lab's 2023 Nature Methods paper on scRNA-seq integration.",
  "recommendation_score": 8,
  "recommendation_reason": "Strong technical background, specific research proposal, excellent fit. Only missing PhD-level experience.",
  "relevance_score": null,
  "relevance_reason": null,
  "prestige_score": null,
  "prestige_reason": null,
  "key_strengths": ["Deep expertise in single-cell genomics", "Strong coding skills", "Specific research proposal"],
  "concerns": [],
  "next_steps": "Schedule interview - strong candidate",
  "additional_notes": "Candidate shows genuine interest in lab's work and has concrete ideas for PhD project.",
  "information_used": {
    "email_text": true,
    "cv": false,
    "research_plan": false,
    "letters_of_recommendation": false,
    "transcripts": false,
    "other": []
  },
  "suggested_folder": null,
  "suggested_labels": null
}
```

## Example 2: European Excellence (9/8/8)

**Profile**: Marie Dubois, ENS Paris, GNN expert

```json
{
  "category": "application-phd",
  "confidence": 0.95,
  "reasoning": "Exceptional PhD application from ENS Paris with strong GNN and graph ML background",
  "urgency": "normal",
  "urgency_score": 5,
  "urgency_reason": "Standard application timeline",
  "summary": "PhD application from ENS Paris M.Sc. student with expertise in graph neural networks and computational biology",
  "action_items": null,
  "needs_reply": true,
  "reply_deadline": null,
  "reply_suggestion": "Strong candidate - schedule interview",
  "sentiment": "positive",
  "applicant_name": "Marie Dubois",
  "applicant_institution": "École Normale Supérieure (ENS) Paris",
  "nationality": "France",
  "highest_degree_completed": "M.Sc.",
  "current_situation": "Student enrolled at a European university (outside Switzerland)",
  "recent_thesis_title": "Graph Neural Networks for Protein-Protein Interaction Prediction",
  "recommendation_source": "Prof. Jean-Luc Moreau",
  "github_account": "mariedubois-gnn",
  "linkedin_account": null,
  "google_scholar_account": "M_Dubois_ENS",
  "coding_experience": {"score": 9, "evidence": "Expert in PyTorch Geometric, implemented custom GNN architectures, published code on GitHub"},
  "omics_genomics_experience": {"score": 8, "evidence": "Applied GNNs to protein interaction networks, worked with STRING database, published in Bioinformatics"},
  "medical_data_experience": {"score": 6, "evidence": "Thesis involved clinical protein interaction data"},
  "sequence_analysis_algorithms_experience": {"score": 7, "evidence": "Developed graph-based methods for sequence similarity"},
  "image_analysis_experience": {"score": 0, "evidence": "Not mentioned in email"},
  "profile_tags": [
    {"tag": "graph_neural_networks", "confidence": 0.98, "reason": "Core expertise in GNNs for biological networks"},
    {"tag": "computational_biology", "confidence": 0.90, "reason": "Strong background in protein interaction prediction"}
  ],
  "is_mass_email": false,
  "no_research_background": false,
  "irrelevant_field": false,
  "possible_spam": false,
  "is_followup": false,
  "is_cold_email": false,
  "scientific_excellence_score": 9,
  "scientific_excellence_reason": "ENS Paris is one of France's most prestigious institutions (Grande École). Grade 16.5/20 converts to ~5.7 ETH equivalent (excellent). Strong publication in Bioinformatics journal. Recommendation from well-known computational biology professor. Exceptional research output for M.Sc. level.",
  "research_fit_score": 8,
  "research_fit_reason": "Excellent technical fit - GNNs are highly relevant for temporal and multimodal data analysis. Mentions specific interest in applying graph methods to cancer genomics. Good alignment with lab's ML focus, though less direct experience with single-cell omics.",
  "recommendation_score": 8,
  "recommendation_reason": "Exceptional candidate with strong technical skills and research output. ENS Paris provides excellent training. GNN expertise is valuable and transferable. Would be a strong addition to the lab. Only minor gap: less direct experience with omics data types compared to single-cell specialists.",
  "relevance_score": null,
  "relevance_reason": null,
  "prestige_score": null,
  "prestige_reason": null,
  "key_strengths": ["Exceptional GNN expertise", "Strong publication record", "Prestigious institution", "Clear research vision"],
  "concerns": [],
  "next_steps": "Schedule interview - exceptional candidate",
  "additional_notes": "ENS Paris is highly selective and produces excellent researchers. Candidate shows deep understanding of graph ML and its applications to biology. Would bring valuable expertise to the lab.",
  "information_used": {
    "email_text": true,
    "cv": false,
    "research_plan": false,
    "letters_of_recommendation": false,
    "transcripts": false,
    "other": []
  },
  "suggested_folder": null,
  "suggested_labels": null
}
```

## Example 3: German Technical Excellence (8/8/8)

**Profile**: Thomas Müller, TU Munich, clinical AI implementer

```json
{
  "category": "application-phd",
  "confidence": 0.93,
  "reasoning": "Strong PhD application from TU Munich with excellent clinical AI implementation experience",
  "urgency": "normal",
  "urgency_score": 5,
  "urgency_reason": "Standard application timeline",
  "summary": "PhD application from TU Munich M.Sc. student with strong background in clinical AI and medical data science",
  "action_items": null,
  "needs_reply": true,
  "reply_deadline": null,
  "reply_suggestion": "Strong candidate - request CV and schedule interview",
  "sentiment": "positive",
  "applicant_name": "Thomas Müller",
  "applicant_institution": "Technical University of Munich (TUM)",
  "nationality": "Germany",
  "highest_degree_completed": "M.Sc.",
  "current_situation": "Student enrolled at a European university (outside Switzerland)",
  "recent_thesis_title": "Deep Learning for Clinical Decision Support in Oncology",
  "recommendation_source": "Prof. Andreas Schmidt",
  "github_account": "thomasmueller-clinical-ai",
  "linkedin_account": null,
  "google_scholar_account": "T_Mueller_TUM",
  "coding_experience": {"score": 8, "evidence": "Strong Python/TensorFlow skills, implemented clinical ML pipelines, contributed to open-source medical AI tools"},
  "omics_genomics_experience": {"score": 6, "evidence": "Worked with genomic data in clinical context, used standard pipelines"},
  "medical_data_experience": {"score": 9, "evidence": "Extensive experience with clinical oncology data, implemented ML models for patient outcome prediction, worked with real hospital datasets"},
  "sequence_analysis_algorithms_experience": {"score": 4, "evidence": "Used standard tools but didn't develop new algorithms"},
  "image_analysis_experience": {"score": 7, "evidence": "Worked with histopathology images in clinical context"},
  "profile_tags": [
    {"tag": "computational_pathology", "confidence": 0.95, "reason": "Strong experience with histopathology and clinical AI"},
    {"tag": "clinical_ai", "confidence": 0.98, "reason": "Core expertise in clinical decision support systems"}
  ],
  "is_mass_email": false,
  "no_research_background": false,
  "irrelevant_field": false,
  "possible_spam": false,
  "is_followup": false,
  "is_cold_email": false,
  "scientific_excellence_score": 8,
  "scientific_excellence_reason": "TU Munich is a top-tier German technical university (score 8-9). Grade 1.4 (sehr gut) converts to ~5.4 ETH equivalent (very good). Strong publication in Medical Image Analysis journal. Recommendation from respected clinical AI researcher. Excellent research output with real-world clinical impact.",
  "research_fit_score": 8,
  "research_fit_reason": "Excellent fit with lab's AI & oncology focus. Strong clinical AI experience directly aligns with precision medicine research. Mentions specific interest in applying ML to cancer genomics. Technical skills in medical imaging complement lab's multimodal data focus. Good understanding of clinical translation challenges.",
  "recommendation_score": 8,
  "recommendation_reason": "Strong candidate with excellent technical skills and relevant domain expertise. TU Munich provides excellent training. Clinical AI experience is highly valuable and directly applicable. Would contribute meaningfully to lab's cancer research. Well-prepared for PhD-level work.",
  "relevance_score": null,
  "relevance_reason": null,
  "prestige_score": null,
  "prestige_reason": null,
  "key_strengths": ["Strong clinical AI expertise", "Real-world medical data experience", "Top-tier German university", "Clear research direction"],
  "concerns": [],
  "next_steps": "Schedule interview - strong candidate",
  "additional_notes": "TU Munich is highly regarded for technical excellence. Candidate shows strong practical skills in clinical AI implementation. Experience with real hospital data is valuable. Would bring important clinical perspective to the lab.",
  "information_used": {
    "email_text": true,
    "cv": false,
    "research_plan": false,
    "letters_of_recommendation": false,
    "transcripts": false,
    "other": []
  },
  "suggested_folder": null,
  "suggested_labels": null
}
```

## Example 4: Weak German Candidate (4/3/3)

**Profile**: Limited experience, poor grades from German university

```json
{
  "category": "application-phd",
  "confidence": 0.88,
  "reasoning": "PhD application with weak academic record and limited relevant experience",
  "urgency": "low",
  "urgency_score": 3,
  "urgency_reason": "Can be processed in normal queue",
  "summary": "PhD application from German university with weak grades and minimal research experience",
  "action_items": null,
  "needs_reply": true,
  "reply_deadline": null,
  "reply_suggestion": "Polite rejection - insufficient qualifications",
  "sentiment": "neutral",
  "applicant_name": "Noa Weber",
  "applicant_institution": "Fachhochschule Darmstadt",
  "nationality": "Germany",
  "highest_degree_completed": "M.Sc.",
  "current_situation": "Student enrolled at a European university (outside Switzerland)",
  "recent_thesis_title": "Basic Machine Learning Approaches for Data Classification",
  "recommendation_source": null,
  "github_account": null,
  "linkedin_account": null,
  "google_scholar_account": null,
  "coding_experience": {"score": 4, "evidence": "Mentions Python and R but no specific projects or contributions"},
  "omics_genomics_experience": {"score": 1, "evidence": "Brief mention of one bioinformatics course"},
  "medical_data_experience": {"score": 0, "evidence": "Not mentioned in email"},
  "sequence_analysis_algorithms_experience": {"score": 0, "evidence": "Not mentioned in email"},
  "image_analysis_experience": {"score": 0, "evidence": "Not mentioned in email"},
  "profile_tags": [],
  "is_mass_email": false,
  "no_research_background": false,
  "irrelevant_field": false,
  "possible_spam": false,
  "is_followup": false,
  "is_cold_email": true,
  "scientific_excellence_score": 4,
  "scientific_excellence_reason": "FH Darmstadt is primarily teaching-focused (score 4). Grade 2.7 (befriedigend) converts to ~4.2 ETH equivalent - below average. No publications mentioned. Minimal research experience. Generic thesis title suggests limited depth.",
  "research_fit_score": 3,
  "research_fit_reason": "Very vague interest in 'machine learning and biology'. No specific mention of lab's research areas. No understanding of single-cell omics, temporal data, or cancer genomics. Generic application.",
  "recommendation_score": 3,
  "recommendation_reason": "Insufficient qualifications for PhD program. Weak academic record, no research output, minimal technical skills, poor fit with lab focus. Would need significant additional training.",
  "relevance_score": null,
  "relevance_reason": null,
  "prestige_score": null,
  "prestige_reason": null,
  "key_strengths": [],
  "concerns": ["Weak academic performance", "No publications or research output", "Minimal technical skills", "Generic interest without specific knowledge of lab work"],
  "next_steps": "Polite rejection - insufficient qualifications",
  "additional_notes": "Application appears to be sent to multiple labs (generic content). Candidate would benefit from gaining more research experience before applying to PhD programs.",
  "information_used": {
    "email_text": true,
    "cv": false,
    "research_plan": false,
    "letters_of_recommendation": false,
    "transcripts": false,
    "other": []
  },
  "suggested_folder": null,
  "suggested_labels": null
}
```

## Example 5: Weak Indian Candidate (4/4/4)

**Profile**: Limited experience from second-tier Indian institution

```json
{
  "category": "application-phd",
  "confidence": 0.85,
  "reasoning": "PhD application from Indian institute with limited research background",
  "urgency": "low",
  "urgency_score": 3,
  "urgency_reason": "Standard low-priority application",
  "summary": "PhD application from India with software development background but minimal research experience",
  "action_items": null,
  "needs_reply": true,
  "reply_deadline": null,
  "reply_suggestion": "Polite rejection with encouragement to gain research experience first",
  "sentiment": "positive",
  "applicant_name": "Raj Patel",
  "applicant_institution": "Vellore Institute of Technology",
  "nationality": "India",
  "highest_degree_completed": "M.Tech.",
  "current_situation": "Not a student, but living outside Europe",
  "recent_thesis_title": "Web Application for Healthcare Data Management",
  "recommendation_source": null,
  "github_account": "rajpatel-dev",
  "linkedin_account": "raj-patel-software",
  "google_scholar_account": null,
  "coding_experience": {"score": 6, "evidence": "3 years software development experience, mentions Java/Python, active GitHub with web projects"},
  "omics_genomics_experience": {"score": 0, "evidence": "Not mentioned in email"},
  "medical_data_experience": {"score": 2, "evidence": "Thesis on healthcare data management but focused on software engineering, not data science"},
  "sequence_analysis_algorithms_experience": {"score": 0, "evidence": "Not mentioned in email"},
  "image_analysis_experience": {"score": 0, "evidence": "Not mentioned in email"},
  "profile_tags": [],
  "is_mass_email": true,
  "no_research_background": true,
  "irrelevant_field": false,
  "possible_spam": false,
  "is_followup": false,
  "is_cold_email": true,
  "scientific_excellence_score": 4,
  "scientific_excellence_reason": "VIT is a good regional university in India but primarily teaching-focused. CGPA 7.8/10 (~4.2-4.3 ETH equivalent). No academic publications. M.Tech. thesis focused on software engineering rather than research. Work experience is in industry software development, not research.",
  "research_fit_score": 4,
  "research_fit_reason": "Interest expressed in 'AI for healthcare' is too broad. No specific knowledge of genomics, cancer research, or lab's focus areas. Background is software engineering, not computational biology or biomedical informatics. Would need substantial retraining.",
  "recommendation_score": 4,
  "recommendation_reason": "Motivated candidate with solid coding skills but lacks research background and domain knowledge. The gap between software engineering and computational genomics is significant. Would be better suited for a Master's program to gain research fundamentals first.",
  "relevance_score": null,
  "relevance_reason": null,
  "prestige_score": null,
  "prestige_reason": null,
  "key_strengths": ["Good coding skills", "Motivated and enthusiastic"],
  "concerns": ["No research publications", "Limited understanding of computational biology", "Background mismatch (software engineering vs research)", "Generic mass email"],
  "next_steps": "Polite rejection - suggest gaining research experience first",
  "additional_notes": "Candidate seems enthusiastic but application is generic (sent to many labs). The email mentions 'your prestigious lab' without naming any specific work. Recommend pursuing research internships or Master's program before PhD.",
  "information_used": {
    "email_text": true,
    "cv": false,
    "research_plan": false,
    "letters_of_recommendation": false,
    "transcripts": false,
    "other": []
  },
  "suggested_folder": null,
  "suggested_labels": null
}
```

## Example 6: Borderline Swiss Candidate (6/5/5)

**Profile**: University of Zurich, right interests but weak technical skills

```json
{
  "category": "application-phd",
  "confidence": 0.90,
  "reasoning": "PhD application from nearby university with relevant interests but limited technical depth",
  "urgency": "normal",
  "urgency_score": 5,
  "urgency_reason": "Standard timeline, borderline case needs discussion",
  "summary": "PhD application from UZH with cancer biology background and interest in AI but limited ML experience",
  "action_items": null,
  "needs_reply": true,
  "reply_deadline": null,
  "reply_suggestion": "Request CV and consider interview to assess learning potential",
  "sentiment": "positive",
  "applicant_name": "Anna Huber",
  "applicant_institution": "University of Zurich",
  "nationality": "Switzerland",
  "highest_degree_completed": "M.Sc.",
  "current_situation": "Student enrolled at another Swiss university",
  "recent_thesis_title": "Genomic Alterations in Breast Cancer Patient Cohorts",
  "recommendation_source": null,
  "github_account": null,
  "linkedin_account": null,
  "google_scholar_account": null,
  "coding_experience": {"score": 4, "evidence": "Mentions R for statistics and some Python basics, but no software development experience"},
  "omics_genomics_experience": {"score": 6, "evidence": "Master's thesis analyzed genomic data from cancer patients using standard pipelines"},
  "medical_data_experience": {"score": 5, "evidence": "Worked with clinical cancer patient data in thesis"},
  "sequence_analysis_algorithms_experience": {"score": 3, "evidence": "Used standard tools (BWA, GATK) but didn't develop algorithms"},
  "image_analysis_experience": {"score": 0, "evidence": "Not mentioned in email"},
  "profile_tags": [
    {"tag": "cancer_genomics", "confidence": 0.6, "reason": "Thesis on cancer genomics but using standard approaches"}
  ],
  "is_mass_email": false,
  "no_research_background": false,
  "irrelevant_field": false,
  "possible_spam": false,
  "is_followup": false,
  "is_cold_email": false,
  "scientific_excellence_score": 6,
  "scientific_excellence_reason": "UZH is a strong Swiss research university (score 7). Grade 5.0 is average for ETH/UZH - not outstanding. One preprint submitted but not yet published. Research experience is solid (cancer genomics) but applied existing methods rather than developing new approaches. Academic record is acceptable but not exceptional.",
  "research_fit_score": 5,
  "research_fit_reason": "Good domain fit (cancer genomics aligns with lab's AI & cancer focus) but weak technical fit. Mentions interest in 'applying AI to genomics' but has minimal ML background. Hasn't taken advanced ML courses. Would need extensive technical training. Proximity to ETH could facilitate collaboration but doesn't compensate for technical gaps.",
  "recommendation_score": 5,
  "recommendation_reason": "Borderline candidate. Pros: relevant domain knowledge (cancer genomics), local candidate (UZH), genuine interest in lab's work. Cons: weak technical skills (limited coding/ML), average grades, no strong research output yet. Success would depend on ability to quickly learn ML fundamentals. Consider interview to assess motivation and learning capacity.",
  "relevance_score": null,
  "relevance_reason": null,
  "prestige_score": null,
  "prestige_reason": null,
  "key_strengths": ["Relevant domain knowledge in cancer genomics", "Local candidate familiar with Swiss academic culture", "Genuine interest in AI for genomics"],
  "concerns": ["Limited ML/coding skills", "Average academic performance (5.0)", "No publications yet", "Would need significant technical training"],
  "next_steps": "Request full CV - borderline case, consider interview",
  "additional_notes": "This is a borderline case. The candidate has the right domain background and interests but lacks the technical skills typical of successful PhD students in the lab. UZH proximity could enable easier collaboration. Decision should weigh: Is candidate teachable and motivated enough to bridge the technical gap? Consider offering interview to assess learning potential and motivation.",
  "information_used": {
    "email_text": true,
    "cv": false,
    "research_plan": false,
    "letters_of_recommendation": false,
    "transcripts": false,
    "other": []
  },
  "suggested_folder": null,
  "suggested_labels": null
}
```

## Usage

These examples are automatically loaded when building application prompts. They serve as calibration points for the LLM to understand:
- **High scores (8-9)**: Strong candidates with excellent qualifications
- **Borderline (5-6)**: Candidates with some strengths but significant gaps
- **Low scores (3-4)**: Weak candidates who don't meet minimum qualifications

The diversity of examples helps prevent geographic and institutional bias while maintaining high standards.

## Note on current_situation Options

Updated to include 11 options that better capture employment status:
- Student enrolled at ETH Zurich / Swiss / EU / non-EU universities
- Postdoc at Swiss / EU / non-EU institutions
- Employed in academia (research position, not postdoc)
- Employed in industry
- Not currently employed/student
- Other (brief summary)

This distinguishes students from postdocs and captures academic vs industry employment.
