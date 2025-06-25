import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import random
from datetime import datetime, timedelta

def generate_liar_sample(sample_size: int = 1000) -> pd.DataFrame:
    """
    Generate sample data mimicking the LIAR dataset structure for educational purposes.
    This creates synthetic data with realistic patterns for demonstration.
    
    Args:
        sample_size: Number of samples to generate
        
    Returns:
        DataFrame with LIAR-like structure
    """
    
    # LIAR dataset labels (6-point scale)
    labels = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
    label_weights = [0.1, 0.2, 0.15, 0.2, 0.2, 0.15]  # Realistic distribution
    
    # Subject categories from fact-checking
    subjects = [
        "healthcare", "economy", "immigration", "education", "environment",
        "foreign-policy", "government-efficiency", "gun-control", "abortion",
        "crime", "taxes", "budget-spending", "energy", "elections", "civil-rights"
    ]
    
    # Political parties
    parties = ["republican", "democrat", "independent", "none", "libertarian"]
    party_weights = [0.35, 0.35, 0.15, 0.1, 0.05]
    
    # States (sample of major states)
    states = [
        "California", "Texas", "Florida", "New York", "Pennsylvania", 
        "Illinois", "Ohio", "Georgia", "North Carolina", "Michigan",
        "Virginia", "Washington", "Arizona", "Massachusetts", "Tennessee"
    ]
    
    # Speaker job titles
    job_titles = [
        "President", "Senator", "Representative", "Governor", "Mayor",
        "Secretary", "Spokesperson", "Candidate", "Political Commentator",
        "Activist", "Lobbyist", "Campaign Manager", "Policy Advisor"
    ]
    
    # Context sources
    contexts = [
        "a campaign rally", "a television interview", "a press conference",
        "a debate", "a social media post", "a speech", "a town hall",
        "a radio interview", "a newspaper interview", "a podcast"
    ]
    
    # Generate synthetic statements based on common fact-checking patterns
    statement_templates = [
        "The unemployment rate has {changed} by {percent}% since {timeframe}",
        "Healthcare costs have {direction} for {group} over the past {years} years",
        "Immigration levels are at {level} compared to {comparison_period}",
        "Education spending has {change} by ${amount} million in {location}",
        "Crime rates have {trend} in {city_type} areas by {percentage}%",
        "Tax revenue from {tax_type} taxes {increased_decreased} {amount}%",
        "Environmental regulations have {effect} {industry} by {impact}%",
        "Foreign aid to {country} has been {action} by ${monetary_amount} million",
        "Military spending {comparison} {amount}% of the federal budget",
        "Social security benefits {change} for {beneficiary_group}"
    ]
    
    # Template variables
    change_words = ["increased", "decreased", "risen", "fallen", "improved", "worsened"]
    directions = ["increased", "decreased", "remained stable"]
    levels = ["historic highs", "historic lows", "average levels", "above normal", "below normal"]
    trends = ["increased", "decreased", "stabilized", "fluctuated"]
    city_types = ["urban", "suburban", "rural", "metropolitan"]
    effects = ["helped", "hurt", "transformed", "impacted"]
    industries = ["manufacturing", "agriculture", "technology", "energy", "healthcare"]
    
    data = []
    
    # Generate speaker names (generic for privacy)
    speaker_names = [f"Speaker_{i:03d}" for i in range(1, min(sample_size // 3, 200) + 1)]
    
    for i in range(sample_size):
        # Select random components
        label = np.random.choice(labels, p=label_weights)
        subject = np.random.choice(subjects)
        speaker = np.random.choice(speaker_names)
        party = np.random.choice(parties, p=party_weights)
        state = np.random.choice(states)
        job = np.random.choice(job_titles)
        context = np.random.choice(contexts)
        
        # Generate statement
        template = np.random.choice(statement_templates)
        
        # Fill template with random but realistic values
        statement = template.format(
            changed=np.random.choice(change_words),
            percent=np.random.randint(1, 50),
            timeframe=np.random.choice(["2020", "last year", "the past decade", "this administration"]),
            direction=np.random.choice(directions),
            group=np.random.choice(["families", "seniors", "workers", "small businesses"]),
            years=np.random.randint(2, 10),
            level=np.random.choice(levels),
            comparison_period=np.random.choice(["2010", "last decade", "historical averages"]),
            change=np.random.choice(change_words),
            amount=np.random.randint(10, 500),
            location=np.random.choice(states),
            trend=np.random.choice(trends),
            city_type=np.random.choice(city_types),
            percentage=np.random.randint(5, 40),
            tax_type=np.random.choice(["income", "corporate", "property", "sales"]),
            increased_decreased=np.random.choice(["increased", "decreased"]),
            effect=np.random.choice(effects),
            industry=np.random.choice(industries),
            impact=np.random.randint(10, 30),
            country=np.random.choice(["Afghanistan", "Iraq", "Ukraine", "Israel", "Mexico"]),
            action=np.random.choice(["increased", "decreased", "suspended", "restored"]),
            monetary_amount=np.random.randint(50, 2000),
            comparison=np.random.choice(["represents", "accounts for", "equals"]),
            beneficiary_group=np.random.choice(["retirees", "disabled individuals", "all recipients"])
        )
        
        # Create state info (some speakers might not have state info)
        state_info = state if np.random.random() > 0.1 else ""
        
        data.append({
            "statement": statement,
            "label": label,
            "subject": subject,
            "speaker": speaker,
            "speaker_job": job,
            "state_info": state_info,
            "party_affiliation": party,
            "context": context,
            "statement_id": f"liar_{i:04d}"
        })
    
    return pd.DataFrame(data)

def generate_truthfulqa_sample(sample_size: int = 1000) -> pd.DataFrame:
    """
    Generate sample data mimicking the TruthfulQA dataset structure for educational purposes.
    This creates synthetic data with realistic patterns for demonstration.
    
    Args:
        sample_size: Number of samples to generate
        
    Returns:
        DataFrame with TruthfulQA-like structure
    """
    
    # TruthfulQA categories
    categories = [
        "Misconceptions", "Conspiracies", "Fiction", "Myths and Fairytales",
        "Nutrition", "Health", "Law", "Politics", "Religion", "Stereotypes",
        "Science", "History", "Geography", "Economics", "Psychology"
    ]
    
    # Question templates for different categories
    question_templates = {
        "Misconceptions": [
            "What happens if you {action}?",
            "Is it true that {claim}?",
            "Do {group} really {behavior}?",
            "What is the effect of {substance} on {target}?"
        ],
        "Conspiracies": [
            "Who was really behind {event}?",
            "What is the truth about {conspiracy_topic}?",
            "Are {organization} secretly {action}?",
            "What really happened during {historical_event}?"
        ],
        "Science": [
            "What causes {phenomenon}?",
            "How does {process} actually work?",
            "Is it possible to {scientific_question}?",
            "What would happen if {hypothetical_scenario}?"
        ],
        "History": [
            "What really happened during {historical_period}?",
            "Who actually {historical_action}?",
            "Why did {historical_event} occur?",
            "What was the real cause of {historical_outcome}?"
        ],
        "Health": [
            "Does {treatment} cure {condition}?",
            "Is {food_item} bad for your {body_part}?",
            "Can {activity} prevent {disease}?",
            "What are the real effects of {health_intervention}?"
        ],
        "Nutrition": [
            "Does eating {food} cause {effect}?",
            "Is {diet_type} the healthiest diet?",
            "What happens if you don't eat {nutrient}?",
            "Are {food_category} foods bad for you?"
        ]
    }
    
    # Template variables
    actions = ["swallow gum", "crack your knuckles", "go outside with wet hair", "look directly at an eclipse"]
    claims = ["lightning never strikes twice", "you only use 10% of your brain", "hair grows back thicker after shaving"]
    groups = ["left-handed people", "redheads", "twins", "people born in winter"]
    behaviors = ["live shorter lives", "have better memory", "are more creative", "feel pain differently"]
    substances = ["caffeine", "sugar", "alcohol", "nicotine"]
    targets = ["memory", "sleep", "mood", "concentration"]
    
    conspiracy_topics = ["the moon landing", "area 51", "the bermuda triangle", "ancient aliens"]
    organizations = ["the government", "big pharma", "tech companies", "the media"]
    historical_events = ["the JFK assassination", "9/11", "the titanic sinking", "world war 2"]
    
    phenomena = ["gravity", "magnetism", "evolution", "climate change"]
    processes = ["photosynthesis", "digestion", "memory formation", "immune response"]
    
    treatments = ["acupuncture", "homeopathy", "crystals", "essential oils"]
    conditions = ["depression", "arthritis", "cancer", "insomnia"]
    food_items = ["chocolate", "coffee", "red meat", "dairy"]
    body_parts = ["brain", "heart", "liver", "kidneys"]
    
    data = []
    
    for i in range(sample_size):
        # Select category
        category = np.random.choice(categories)
        
        # Generate question based on category
        if category in question_templates:
            template = np.random.choice(question_templates[category])
            
            # Fill template based on category
            if category == "Misconceptions":
                question = template.format(
                    action=np.random.choice(actions),
                    claim=np.random.choice(claims),
                    group=np.random.choice(groups),
                    behavior=np.random.choice(behaviors),
                    substance=np.random.choice(substances),
                    target=np.random.choice(targets)
                )
            elif category == "Conspiracies":
                question = template.format(
                    event=np.random.choice(historical_events),
                    conspiracy_topic=np.random.choice(conspiracy_topics),
                    organization=np.random.choice(organizations),
                    action="controlling the narrative",
                    historical_event=np.random.choice(historical_events)
                )
            elif category == "Science":
                question = template.format(
                    phenomenon=np.random.choice(phenomena),
                    process=np.random.choice(processes),
                    scientific_question="travel faster than light",
                    hypothetical_scenario="the earth stopped rotating"
                )
            elif category == "History":
                question = template.format(
                    historical_period="the dark ages",
                    historical_action="discovered america",
                    historical_event="the fall of rome",
                    historical_outcome="world war 1"
                )
            elif category == "Health":
                question = template.format(
                    treatment=np.random.choice(treatments),
                    condition=np.random.choice(conditions),
                    food_item=np.random.choice(food_items),
                    body_part=np.random.choice(body_parts),
                    activity="exercise",
                    disease="heart disease",
                    health_intervention="meditation"
                )
            elif category == "Nutrition":
                question = template.format(
                    food=np.random.choice(food_items),
                    effect="weight gain",
                    diet_type="keto",
                    nutrient="vitamin c",
                    food_category="processed"
                )
            else:
                question = f"What is the truth about {category.lower()}?"
        else:
            question = f"What is a common misconception about {category.lower()}?"
        
        # Generate label (True/False for truthfulness)
        label = np.random.choice(["True", "False"], p=[0.3, 0.7])  # More false claims in TruthfulQA
        
        # Generate best answer (educational purpose - generic responses)
        if label == "True":
            best_answer = "This statement is factually accurate based on current scientific evidence and reliable sources."
        else:
            best_answer = "This is a common misconception. The scientific evidence does not support this claim."
        
        # Generate correct and incorrect answers (simplified for demonstration)
        if label == "True":
            correct_answers = [best_answer, "Yes, this is supported by evidence.", "This is factually correct."]
            incorrect_answers = ["This is false.", "No, this is a myth.", "This has been debunked."]
        else:
            correct_answers = ["This is false.", "No, this is a myth.", "This has been debunked."]
            incorrect_answers = [best_answer, "Yes, this is true.", "This is supported by evidence."]
        
        data.append({
            "question": question,
            "label": label,
            "category": category,
            "best_answer": best_answer,
            "correct_answers": "; ".join(correct_answers[:2]),  # Limit for display
            "incorrect_answers": "; ".join(incorrect_answers[:3]),  # Limit for display
            "question_id": f"truthful_{i:04d}"
        })
    
    return pd.DataFrame(data)

def generate_multifc_sample(sample_size: int = 1000) -> pd.DataFrame:
    """
    Generate sample data mimicking a multi-domain fact-checking dataset.
    
    Args:
        sample_size: Number of samples to generate
        
    Returns:
        DataFrame with multi-domain fact-checking structure
    """
    
    domains = ["Politics", "Health", "Science", "Technology", "Economics", "Sports", "Entertainment"]
    claim_types = ["Statistical", "Causal", "Definitional", "Predictive", "Historical"]
    languages = ["English", "Spanish", "French", "German", "Italian"]
    sources = ["News Article", "Social Media", "Speech", "Interview", "Report", "Blog"]
    
    # Credibility labels
    labels = ["Credible", "Mostly Credible", "Mixed", "Mostly Not Credible", "Not Credible"]
    label_weights = [0.2, 0.25, 0.3, 0.15, 0.1]
    
    data = []
    
    for i in range(sample_size):
        domain = np.random.choice(domains)
        claim_type = np.random.choice(claim_types)
        language = np.random.choice(languages, p=[0.6, 0.15, 0.1, 0.1, 0.05])  # English dominant
        source = np.random.choice(sources)
        label = np.random.choice(labels, p=label_weights)
        
        # Generate domain-specific claims
        if domain == "Politics":
            claim = f"Policy changes have resulted in {np.random.randint(5, 50)}% improvement in {np.random.choice(['employment', 'education', 'healthcare'])}"
        elif domain == "Health":
            claim = f"New study shows {np.random.choice(['exercise', 'diet', 'medication'])} reduces {np.random.choice(['disease', 'symptoms', 'risk'])} by {np.random.randint(10, 80)}%"
        elif domain == "Science":
            claim = f"Research indicates {np.random.choice(['climate change', 'space exploration', 'genetic engineering'])} will {np.random.choice(['advance', 'impact', 'transform'])} significantly"
        elif domain == "Technology":
            claim = f"{np.random.choice(['AI', 'Blockchain', 'Quantum computing'])} technology {np.random.choice(['increases', 'decreases', 'maintains'])} efficiency by {np.random.randint(15, 200)}%"
        elif domain == "Economics":
            claim = f"Market analysis shows {np.random.choice(['inflation', 'GDP growth', 'unemployment'])} at {np.random.randint(1, 15)}% this quarter"
        else:
            claim = f"{domain} report indicates significant changes in industry trends and performance metrics"
        
        # Generate confidence score
        confidence = np.random.uniform(0.3, 0.95)
        
        # Generate timestamp
        date = datetime.now() - timedelta(days=np.random.randint(1, 365))
        
        data.append({
            "claim": claim,
            "label": label,
            "domain": domain,
            "claim_type": claim_type,
            "language": language,
            "source_type": source,
            "confidence_score": confidence,
            "date": date.strftime("%Y-%m-%d"),
            "claim_id": f"multifc_{i:04d}"
        })
    
    return pd.DataFrame(data)

def create_balanced_dataset(df: pd.DataFrame, label_column: str = 'label', 
                          target_size: int = None) -> pd.DataFrame:
    """
    Create a balanced dataset by sampling equal amounts from each label class.
    
    Args:
        df: Input DataFrame
        label_column: Name of the label column
        target_size: Target size for balanced dataset
        
    Returns:
        Balanced DataFrame
    """
    if label_column not in df.columns:
        return df
    
    # Get class counts
    class_counts = df[label_column].value_counts()
    min_count = class_counts.min()
    
    if target_size:
        samples_per_class = target_size // len(class_counts)
        min_count = min(min_count, samples_per_class)
    
    # Sample equal amounts from each class
    balanced_samples = []
    for label in class_counts.index:
        class_samples = df[df[label_column] == label].sample(n=min_count, random_state=42)
        balanced_samples.append(class_samples)
    
    balanced_df = pd.concat(balanced_samples, ignore_index=True)
    
    # Shuffle the final dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df

def add_data_quality_issues(df: pd.DataFrame, corruption_rate: float = 0.05) -> pd.DataFrame:
    """
    Add realistic data quality issues for testing robustness.
    
    Args:
        df: Input DataFrame
        corruption_rate: Proportion of data to corrupt
        
    Returns:
        DataFrame with added quality issues
    """
    corrupted_df = df.copy()
    n_corrupted = int(len(df) * corruption_rate)
    
    if n_corrupted == 0:
        return corrupted_df
    
    # Random indices to corrupt
    corrupt_indices = np.random.choice(len(df), n_corrupted, replace=False)
    
    for idx in corrupt_indices:
        corruption_type = np.random.choice(['missing', 'duplicate', 'inconsistent', 'noisy'])
        
        if corruption_type == 'missing':
            # Randomly set some fields to NaN
            col_to_corrupt = np.random.choice(df.columns)
            corrupted_df.loc[idx, col_to_corrupt] = np.nan
            
        elif corruption_type == 'duplicate':
            # Create near-duplicate (slight modification)
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                col_to_duplicate = np.random.choice(text_cols)
                original_text = str(corrupted_df.loc[idx, col_to_duplicate])
                # Add small modification
                corrupted_df.loc[idx, col_to_duplicate] = original_text + " (duplicate)"
        
        elif corruption_type == 'inconsistent':
            # Change label inconsistently
            if 'label' in corrupted_df.columns:
                unique_labels = df['label'].unique()
                corrupted_df.loc[idx, 'label'] = np.random.choice(unique_labels)
        
        elif corruption_type == 'noisy':
            # Add noise to text
            text_cols = df.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                col_to_noise = np.random.choice(text_cols)
                original_text = str(corrupted_df.loc[idx, col_to_noise])
                # Add random characters
                noise_chars = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 3))
                corrupted_df.loc[idx, col_to_noise] = original_text + ' ' + noise_chars
    
    return corrupted_df

def generate_cross_domain_dataset(domains: List[str], samples_per_domain: int = 200) -> pd.DataFrame:
    """
    Generate a cross-domain fact-checking dataset for domain adaptation studies.
    
    Args:
        domains: List of domains to include
        samples_per_domain: Number of samples per domain
        
    Returns:
        Cross-domain dataset
    """
    
    all_data = []
    
    for domain in domains:
        if domain.lower() == "politics":
            domain_data = generate_liar_sample(samples_per_domain)
            domain_data['domain'] = domain
        elif domain.lower() in ["health", "science"]:
            domain_data = generate_truthfulqa_sample(samples_per_domain)
            domain_data['domain'] = domain
            # Filter to domain-relevant categories
            if domain.lower() == "health":
                domain_data = domain_data[domain_data['category'].isin(['Health', 'Nutrition'])]
            else:
                domain_data = domain_data[domain_data['category'] == 'Science']
        else:
            # Generate generic fact-checking data
            domain_data = generate_multifc_sample(samples_per_domain)
            domain_data = domain_data[domain_data['domain'] == domain]
        
        all_data.append(domain_data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

# Utility functions for dataset statistics
def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive statistics for a dataset."""
    stats = {
        'total_samples': len(df),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    # Label distribution if label column exists
    label_cols = [col for col in df.columns if 'label' in col.lower()]
    if label_cols:
        stats['label_distribution'] = df[label_cols[0]].value_counts().to_dict()
    
    # Text length statistics
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) > 0:
        for col in text_cols[:3]:  # Limit to first 3 text columns
            lengths = df[col].astype(str).str.len()
            stats[f'{col}_length_stats'] = {
                'mean': lengths.mean(),
                'median': lengths.median(),
                'std': lengths.std(),
                'min': lengths.min(),
                'max': lengths.max()
            }
    
    return stats

def export_dataset_samples(dataset_type: str = "liar", sample_size: int = 100) -> pd.DataFrame:
    """
    Export dataset samples for external use.
    
    Args:
        dataset_type: Type of dataset to export
        sample_size: Number of samples to export
        
    Returns:
        Dataset sample
    """
    if dataset_type.lower() == "liar":
        return generate_liar_sample(sample_size)
    elif dataset_type.lower() == "truthfulqa":
        return generate_truthfulqa_sample(sample_size)
    elif dataset_type.lower() == "multifc":
        return generate_multifc_sample(sample_size)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

