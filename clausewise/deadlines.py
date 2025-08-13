import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd


def extract_deadlines_and_obligations(text: str) -> List[Dict]:
    """
    Extract deadlines and obligations from legal document text.
    Returns a list of dictionaries with deadline information.
    """
    deadlines = []
    
    # Common deadline patterns
    patterns = [
        # Payment deadlines
        r'(?:pay|payment|rent|fee|amount).*?(?:by|on|within|due|before)\s*([^,\n]+)',
        # Renewal dates
        r'(?:renew|renewal|extend|extension).*?(?:by|on|before|until)\s*([^,\n]+)',
        # Termination notices
        r'(?:terminate|termination|cancel|cancellation).*?(?:notice|period|days?)\s*([^,\n]+)',
        # Compliance deadlines
        r'(?:comply|compliance|submit|provide).*?(?:by|within|before)\s*([^,\n]+)',
        # Expiration dates
        r'(?:expire|expiration|valid|validity).*?(?:on|until|before)\s*([^,\n]+)',
        # Specific date patterns
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # MM/DD/YYYY or DD/MM/YYYY
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})',  # DD Month YYYY
        r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4})',  # DD Month YYYY
    ]
    
    # Time period patterns
    time_patterns = [
        r'(\d+)\s*(?:days?|weeks?|months?|years?)\s*(?:from|after|before)',
        r'within\s*(\d+)\s*(?:days?|weeks?|months?|years?)',
        r'(\d+)\s*(?:days?|weeks?|months?|years?)\s*notice',
    ]
    
    # Extract deadlines
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            deadline_text = match.group(1) if len(match.groups()) > 0 else match.group(0)
            context = match.group(0)
            
            # Determine deadline type
            deadline_type = _classify_deadline_type(context)
            
            # Parse date if possible
            parsed_date = _parse_date(deadline_text)
            
            deadlines.append({
                'type': deadline_type,
                'text': deadline_text.strip(),
                'context': context.strip(),
                'parsed_date': parsed_date,
                'urgency': _calculate_urgency(parsed_date),
                'category': _categorize_obligation(context)
            })
    
    # Extract time periods
    for pattern in time_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            time_value = match.group(1)
            context = match.group(0)
            
            deadlines.append({
                'type': 'Time Period',
                'text': f"{time_value} days/weeks/months",
                'context': context.strip(),
                'parsed_date': None,
                'urgency': 'Medium',
                'category': _categorize_obligation(context)
            })
    
    # Remove duplicates and sort by urgency
    unique_deadlines = _remove_duplicates(deadlines)
    return sorted(unique_deadlines, key=lambda x: _urgency_priority(x['urgency']))


def _classify_deadline_type(context: str) -> str:
    """Classify the type of deadline based on context."""
    context_lower = context.lower()
    
    if any(word in context_lower for word in ['pay', 'rent', 'fee', 'amount']):
        return 'Payment Deadline'
    elif any(word in context_lower for word in ['renew', 'renewal', 'extend']):
        return 'Renewal Date'
    elif any(word in context_lower for word in ['terminate', 'termination', 'cancel']):
        return 'Termination Notice'
    elif any(word in context_lower for word in ['comply', 'compliance', 'submit']):
        return 'Compliance Deadline'
    elif any(word in context_lower for word in ['expire', 'expiration', 'valid']):
        return 'Expiration Date'
    else:
        return 'General Deadline'


def _parse_date(date_text: str) -> Optional[datetime]:
    """Parse date text into datetime object."""
    if not date_text:
        return None
    
    date_text = date_text.strip()
    
    # Try different date formats
    date_formats = [
        '%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y', '%d-%m-%Y',
        '%m/%d/%y', '%d/%m/%y', '%m-%d-%y', '%d-%m-%y',
        '%d %B %Y', '%d %b %Y', '%B %d, %Y', '%b %d, %Y'
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_text, fmt)
        except ValueError:
            continue
    
    return None


def _calculate_urgency(parsed_date: Optional[datetime]) -> str:
    """Calculate urgency level based on date."""
    if not parsed_date:
        return 'Unknown'
    
    today = datetime.now()
    days_until = (parsed_date - today).days
    
    if days_until < 0:
        return 'Overdue'
    elif days_until <= 7:
        return 'Critical'
    elif days_until <= 30:
        return 'High'
    elif days_until <= 90:
        return 'Medium'
    else:
        return 'Low'


def _urgency_priority(urgency: str) -> int:
    """Get priority number for sorting."""
    priority_map = {
        'Overdue': 0,
        'Critical': 1,
        'High': 2,
        'Medium': 3,
        'Low': 4,
        'Unknown': 5
    }
    return priority_map.get(urgency, 5)


def _categorize_obligation(context: str) -> str:
    """Categorize the type of obligation."""
    context_lower = context.lower()
    
    if any(word in context_lower for word in ['pay', 'rent', 'fee', 'amount', 'money']):
        return 'Financial'
    elif any(word in context_lower for word in ['renew', 'renewal', 'extend', 'extension']):
        return 'Contract Management'
    elif any(word in context_lower for word in ['terminate', 'termination', 'cancel', 'cancellation']):
        return 'Contract Management'
    elif any(word in context_lower for word in ['comply', 'compliance', 'submit', 'provide', 'notify']):
        return 'Compliance'
    elif any(word in context_lower for word in ['expire', 'expiration', 'valid', 'validity']):
        return 'Contract Management'
    else:
        return 'General'


def _remove_duplicates(deadlines: List[Dict]) -> List[Dict]:
    """Remove duplicate deadlines based on text and context."""
    seen = set()
    unique = []
    
    for deadline in deadlines:
        # Create a key based on text and context
        key = (deadline['text'], deadline['context'])
        if key not in seen:
            seen.add(key)
            unique.append(deadline)
    
    return unique


def generate_deadlines_dataframe(deadlines: List[Dict]) -> pd.DataFrame:
    """Convert deadlines list to pandas DataFrame for display."""
    if not deadlines:
        return pd.DataFrame()
    
    df = pd.DataFrame(deadlines)
    
    # Format parsed dates for display
    if 'parsed_date' in df.columns:
        df['parsed_date'] = df['parsed_date'].apply(
            lambda x: x.strftime('%Y-%m-%d') if x else 'Not parsed'
        )
    
    # Reorder columns for better display
    column_order = ['type', 'text', 'context', 'parsed_date', 'urgency', 'category']
    df = df[column_order]
    
    return df


def get_upcoming_deadlines(deadlines: List[Dict], days_ahead: int = 30) -> List[Dict]:
    """Get deadlines coming up in the next N days."""
    today = datetime.now()
    upcoming = []
    
    for deadline in deadlines:
        if deadline['parsed_date'] and deadline['parsed_date'] > today:
            days_until = (deadline['parsed_date'] - today).days
            if days_until <= days_ahead:
                deadline['days_until'] = days_until
                upcoming.append(deadline)
    
    return sorted(upcoming, key=lambda x: x['days_until'])


def get_overdue_deadlines(deadlines: List[Dict]) -> List[Dict]:
    """Get overdue deadlines."""
    today = datetime.now()
    overdue = []
    
    for deadline in deadlines:
        if deadline['parsed_date'] and deadline['parsed_date'] < today:
            days_overdue = (today - deadline['parsed_date']).days
            deadline['days_overdue'] = days_overdue
            overdue.append(deadline)
    
    return sorted(overdue, key=lambda x: x['days_overdue'], reverse=True)
