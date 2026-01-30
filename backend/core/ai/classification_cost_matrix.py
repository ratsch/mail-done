"""
Asymmetric Cost Matrix for Email Classification Errors

Defines the cost/severity of misclassifying email from one category to another.
Used to compute weighted accuracy that reflects real-world impact.

Cost Scale:
- 0: Perfect match (no cost)
- 1: Minor inconvenience (e.g., confusing similar low-priority categories)
- 3: Moderate issue (e.g., missing a non-urgent work email)
- 5: Significant problem (e.g., missing an invitation or application)
- 10: Critical failure (e.g., urgent email classified as spam/notification)

Philosophy:
- Missing important emails (urgent, invitations, applications) = HIGH COST
- False positives for important categories = MEDIUM COST  
- Confusing similar low-priority categories = LOW COST
- Matrix is asymmetric: work-urgent → spam (cost=10) vs spam → work-urgent (cost=1)
"""

# All 43 categories from the classification prompt
CATEGORIES = [
    # Work (human-written, needs attention)
    "work-urgent",
    "work-colleague", 
    "work-student",
    "work-admin",
    "work-scheduling",
    "work-no-action-needed",
    "work-other",
    
    # Applications (people want to join you)
    "application-phd",
    "application-postdoc",
    "application-intern",
    "application-bsc-msc-thesis",
    "application-visiting",
    "application-other",
    
    # Invitations (you're invited to do something)
    "invitation-speaking",
    "invitation-committee",
    "invitation-grant",
    "invitation-editorial",
    "invitation-advisory",
    "invitation-event",
    "invitation-collaboration",
    
    # Reviews (you evaluate)
    "review-peer-journal",
    "review-peer-conference",
    "review-grant",
    "review-phd-committee",
    "review-hiring",
    "review-promotion",
    "review-other",
    
    # Publications (your papers)
    "publication-submission-confirm",
    "publication-decision-accept",
    "publication-decision-reject",
    "publication-revision-request",
    "publication-proofs",
    "publication-published",
    "publication-other",
    
    # Grants (your grants)
    "grant-submission-confirm",
    "grant-decision-awarded",
    "grant-decision-rejected",
    "grant-budget",
    "grant-reporting",
    "grant-modification",
    "grant-other",
    
    # Travel (your trips)
    "travel-booking-confirm",
    "travel-receipt",
    "travel-itinerary",
    "travel-reminder",
    "travel-transport",
    "travel-other",
    
    # Information (newsletters, notifications)
    "newsletter-scientific",
    "newsletter-general",
    "notification-technical",
    "notification-calendar",
    "notification-social",
    "notification-other",
    
    # Transactions
    "receipt-online",
    "receipt-travel",
    "receipt-subscription",
    "receipt-reimbursement",
    
    # Personal
    "personal-family",
    "personal-friends",
    "personal-transaction",
    "personal-health",
    "personal-hobby",
    "personal-travel",
    "personal-shopping",
    "personal-other",
    
    # Low priority
    "marketing",
    "spam",
    "social-media"
]


def get_misclassification_cost(true_category: str, predicted_category: str) -> float:
    """
    Get the cost of misclassifying an email.
    
    Args:
        true_category: The actual category (ground truth)
        predicted_category: The predicted category (model output)
        
    Returns:
        Cost value (0 = perfect, 10 = catastrophic)
        
    Examples:
        >>> get_misclassification_cost("work-urgent", "spam")
        10  # CRITICAL: Urgent work marked as spam
        
        >>> get_misclassification_cost("spam", "work-urgent")
        1   # Minor: Spam gets extra attention
        
        >>> get_misclassification_cost("newsletter-scientific", "newsletter-general")
        1   # Minor: Similar categories
    """
    if true_category == predicted_category:
        return 0  # Perfect match
    
    # Define category groups for easier cost assignment
    urgent_categories = {"work-urgent"}
    
    high_value_categories = {
        "application-phd", "application-postdoc", "application-intern",
        "application-bsc-msc-thesis", "application-visiting",
        "invitation-speaking", "invitation-committee", "invitation-grant",
        "invitation-editorial", "invitation-advisory", "invitation-collaboration",
        "review-peer-journal", "review-peer-conference", "review-grant",
        "review-phd-committee", "review-hiring", "review-promotion"
    }
    
    work_categories = {
        "work-colleague", "work-student", "work-admin", 
        "work-scheduling", "work-other"
    }
    
    # Project categories are detected by prefix - allows extension via config overlay
    # Any category starting with "project-" is treated as a project category
    def is_project_category(cat: str) -> bool:
        return cat.startswith("project-")
    
    low_priority_categories = {
        "marketing", "spam", "social-media",
        "newsletter-general", "notification-technical", 
        "notification-calendar", "notification-social", "notification-other"
    }
    
    informational_categories = {
        "newsletter-scientific", "work-no-action-needed"
    }
    
    transactional_categories = {
        "receipt-online", "receipt-travel", 
        "receipt-subscription", "receipt-reimbursement"
    }
    
    personal_categories = {
        "personal-family", "personal-friends", 
        "personal-transaction", "personal-health",
        "personal-hobby", "personal-travel",
        "personal-shopping", "personal-other"
    }
    
    # CRITICAL FAILURES (Cost = 10)
    # Missing urgent emails by marking them as low priority
    if true_category in urgent_categories and predicted_category in low_priority_categories:
        return 10
    
    # Missing high-value opportunities by marking as spam/notifications
    if true_category in high_value_categories and predicted_category in low_priority_categories:
        return 10
    
    # SEVERE ERRORS (Cost = 8)
    # Urgent email classified as notification (might be ignored)
    if true_category in urgent_categories and predicted_category in (informational_categories | transactional_categories):
        return 8
    
    # High-value email (application/invitation/review) marked as work-other or notification
    if true_category in high_value_categories and predicted_category in informational_categories:
        return 8
    
    # SIGNIFICANT PROBLEMS (Cost = 5-7)
    # Work emails (colleague/student) marked as low priority
    if true_category in work_categories and predicted_category in low_priority_categories:
        return 6
    
    # Project emails marked as low priority (important collaborative projects)
    if is_project_category(true_category) and predicted_category in low_priority_categories:
        return 7
    
    # High-value emails confused with regular work
    if true_category in high_value_categories and predicted_category in work_categories:
        return 5
    
    # Urgent email confused with regular work (still gets attention, but delayed)
    if true_category in urgent_categories and predicted_category in work_categories:
        return 5
    
    # Personal-family marked as spam (missing important personal emails)
    if true_category == "personal-family" and predicted_category in low_priority_categories:
        return 7
    
    # Personal-health marked as spam (missing medical appointments/health issues)
    if true_category == "personal-health" and predicted_category in low_priority_categories:
        return 7
    
    # Publication proofs/revisions marked as low priority (time-sensitive, missing deadlines!)
    if true_category in {"publication-proofs", "publication-revision-request"} and predicted_category in low_priority_categories:
        return 8
    
    # Grant decisions/reporting marked as low priority (important outcomes/deadlines)
    if true_category in {"grant-decision-awarded", "grant-decision-rejected", "grant-reporting"} and predicted_category in low_priority_categories:
        return 7
    
    # MODERATE ISSUES (Cost = 3-4)
    # Confusing application types (phd vs postdoc)
    if true_category in high_value_categories and predicted_category in high_value_categories:
        # Same super-category (application vs application, invitation vs invitation)
        true_prefix = true_category.split("-")[0]
        pred_prefix = predicted_category.split("-")[0]
        if true_prefix == pred_prefix:
            return 2  # Minor: Still gets proper attention
        else:
            return 3  # Moderate: Wrong type of high-value email
    
    # Work categories confused with each other
    if true_category in work_categories and predicted_category in work_categories:
        return 2  # Minor: Still flagged as work
    
    # Project categories confused with work (gets attention, but loses project context)
    # Projects are essentially work categories - confusion with work-colleague is acceptable
    if is_project_category(true_category) and predicted_category in work_categories:
        return 2  # Minor: Still gets attention, project organization is nice-to-have

    # Work confused with project (false positive for project)
    if true_category in work_categories and is_project_category(predicted_category):
        return 1  # Very minor: Gets organized into project folder

    # Project categories confused with each other
    if is_project_category(true_category) and is_project_category(predicted_category):
        return 1  # Very minor: Still in a project folder
    
    # Notification types confused with each other
    if true_category in low_priority_categories and predicted_category in low_priority_categories:
        return 1  # Very minor: Both low priority
    
    # Regular work confused with urgent (false positive - extra attention)
    if true_category in work_categories and predicted_category in urgent_categories:
        return 2  # Minor inconvenience
    
    # MINOR ISSUES (Cost = 1-2)
    # Low priority email gets extra attention (false positive for work)
    if true_category in low_priority_categories and predicted_category in work_categories:
        return 1
    
    # Low priority marked as high-value (false positive - will be quickly dismissed)
    if true_category in low_priority_categories and predicted_category in high_value_categories:
        return 1
    
    # Newsletter types confused
    if "newsletter" in true_category and "newsletter" in predicted_category:
        return 1
    
    # Receipt types confused
    if "receipt" in true_category and "receipt" in predicted_category:
        return 1
    
    # Personal categories confused with each other
    if true_category in personal_categories and predicted_category in personal_categories:
        return 1
    
    # Work-no-action vs newsletters/notifications
    if true_category in informational_categories and predicted_category in informational_categories:
        return 1
    
    # DEFAULT COSTS (everything else)
    # High-value confused with personal (gets attention, just wrong folder)
    if true_category in high_value_categories and predicted_category in personal_categories:
        return 4
    
    # Personal confused with work (gets attention)
    if true_category in personal_categories and predicted_category in work_categories:
        return 3
    
    # Work confused with personal (might be delayed)
    if true_category in work_categories and predicted_category in personal_categories:
        return 4
    
    # Transactional confused with anything else
    if true_category in transactional_categories or predicted_category in transactional_categories:
        return 2  # Minor: Receipts are usually FYI
    
    # Catch-all: Unknown combination
    return 3


def get_cost_matrix():
    """
    Generate the full NxN cost matrix for all category pairs.
    
    Returns:
        dict: Nested dict {true_category: {pred_category: cost}}
    """
    cost_matrix = {}
    for true_cat in CATEGORIES:
        cost_matrix[true_cat] = {}
        for pred_cat in CATEGORIES:
            cost_matrix[true_cat][pred_cat] = get_misclassification_cost(true_cat, pred_cat)
    
    return cost_matrix


def calculate_weighted_accuracy(confusion_data):
    """
    Calculate accuracy weighted by misclassification costs.
    
    Args:
        confusion_data: List of (true_category, predicted_category) tuples
        
    Returns:
        dict with:
            - unweighted_accuracy: Standard accuracy (0-1)
            - weighted_accuracy: Cost-weighted accuracy (0-1)
            - total_cost: Sum of all misclassification costs
            - average_cost: Average cost per email
            - cost_breakdown: Dict of {(true, pred): count} for errors
    """
    if not confusion_data:
        return {
            "unweighted_accuracy": 0,
            "weighted_accuracy": 0,
            "total_cost": 0,
            "average_cost": 0,
            "cost_breakdown": {}
        }
    
    correct = 0
    total_cost = 0
    max_possible_cost = len(confusion_data) * 10  # Max cost per email = 10
    cost_breakdown = {}
    
    for true_cat, pred_cat in confusion_data:
        cost = get_misclassification_cost(true_cat, pred_cat)
        total_cost += cost
        
        if cost == 0:
            correct += 1
        else:
            key = (true_cat, pred_cat)
            cost_breakdown[key] = cost_breakdown.get(key, 0) + 1
    
    unweighted_accuracy = correct / len(confusion_data)
    
    # Weighted accuracy: 1.0 - (actual_cost / max_possible_cost)
    # Perfect (all correct) = 1.0, worst case (all cost 10) = 0.0
    weighted_accuracy = 1.0 - (total_cost / max_possible_cost)
    
    return {
        "unweighted_accuracy": unweighted_accuracy,
        "weighted_accuracy": weighted_accuracy,
        "total_cost": total_cost,
        "average_cost": total_cost / len(confusion_data),
        "cost_breakdown": cost_breakdown,
        "num_emails": len(confusion_data)
    }


def print_cost_matrix_summary():
    """Print a summary of the cost matrix philosophy."""
    print("=" * 80)
    print("MISCLASSIFICATION COST MATRIX")
    print("=" * 80)
    print("\nCost Scale:")
    print("  0  = Perfect match")
    print("  1  = Minor (e.g., newsletter-scientific → newsletter-general)")
    print("  3  = Moderate (e.g., work-colleague → work-student)")
    print("  5  = Significant (e.g., application-phd → work-other)")
    print("  10 = Critical (e.g., work-urgent → spam)")
    print("\nWorst Misclassifications (Cost = 10):")
    print("  • work-urgent → spam/marketing/social-media")
    print("  • application-* → spam/marketing/notification-*")
    print("  • invitation-* → spam/marketing/notification-*")
    print("  • review-* → spam/marketing/notification-*")
    print("\nExample Costs:")
    examples = [
        ("work-urgent", "spam", "CRITICAL"),
        ("application-phd", "notification-other", "CRITICAL"),
        ("work-urgent", "work-colleague", "Significant"),
        ("work-colleague", "work-student", "Minor"),
        ("newsletter-scientific", "newsletter-general", "Very minor"),
        ("spam", "work-urgent", "Very minor (false positive)"),
    ]
    for true_cat, pred_cat, desc in examples:
        cost = get_misclassification_cost(true_cat, pred_cat)
        print(f"  • {true_cat} → {pred_cat}: Cost={cost} ({desc})")
    print("=" * 80)


if __name__ == "__main__":
    # Demo the cost matrix
    print_cost_matrix_summary()
    
    # Show a few specific examples
    print("\n\nDetailed Examples:")
    print("-" * 80)
    
    test_cases = [
        ("work-urgent", "work-colleague"),
        ("work-urgent", "notification-technical"),
        ("work-urgent", "spam"),
        ("application-phd", "application-postdoc"),
        ("application-phd", "work-other"),
        ("application-phd", "spam"),
        ("invitation-speaking", "invitation-committee"),
        ("invitation-speaking", "newsletter-scientific"),
        ("spam", "work-urgent"),
        ("newsletter-general", "notification-other"),
    ]
    
    for true_cat, pred_cat in test_cases:
        cost = get_misclassification_cost(true_cat, pred_cat)
        print(f"{true_cat:30} → {pred_cat:30} = Cost {cost:2}")

