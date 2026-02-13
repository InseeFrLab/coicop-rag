import pytest
from eval.metrics import calculate_accuracy_at_level

## Define test data

records_test = [
    # Case 1: Perfect match - LLM correct, label in retrieved codes
    {
        'product': 'Pain complet bio',
        'coicop_pred': '01.1.1.1.2',
        'code': '01.1.1.1.2',
        'list_retrieved_codes': [
            '01.1.1.1.2',
            '01.1.1.2.0',
            '01.1.2.0.0'
        ],
        'confidence': 0.95,
        'parsed': True,
        'codable': True
    },
    
    # Case 2: LLM correct at level 4, but full codes differ (truncate helps)
    # Label NOT in retrieved codes, but LLM guesses correctly at level 4
    {
        'product': 'Chocolat noir 70%',
        'coicop_pred': '01.1.8.3.5',  # Level 4: 01.1.8.3
        'code': '01.1.8.3.7',          # Level 4: 01.1.8.3 (same!)
        'list_retrieved_codes': [
            '01.1.8.1.0',
            '01.1.8.2.0',
            '01.1.7.0.0'
            # Label '01.1.8.3.7' NOT in retrieved codes
        ],
        'confidence': 0.82,
        'parsed': True,
        'codable': True
    },
    
    # Case 3: LLM wrong even though correct code is in retrieved codes
    {
        'product': 'Vin rouge Bordeaux',
        'coicop_pred': '02.1.2.0.0',  # Wrong choice
        'code': '02.1.1.1.0',          # Correct code
        'list_retrieved_codes': [
            '02.1.1.1.0',  # Correct code IS here!
            '02.1.2.0.0',  # But LLM chose this one
            '02.1.3.0.0'
        ],
        'confidence': 0.75,
        'parsed': True,
        'codable': True
    },
    
    # Case 4: LLM wrong, label not in retrieved codes (retriever failed)
    {
        'product': 'Smartphone Samsung',
        'coicop_pred': '08.2.1.0.0',
        'code': '08.2.0.1.0',
        'list_retrieved_codes': [
            '08.2.1.0.0',
            '08.3.0.0.0',
            '09.1.0.0.0'
            # Label '08.2.0.1.0' NOT in retrieved codes
        ],
        'confidence': 0.68,
        'parsed': True,
        'codable': True
    },
    
    # Case 5: LLM correct at level 4 thanks to truncate, different level 5
    # Label IS in retrieved codes (at full precision)
    {
        'product': 'Café arabica moulu',
        'coicop_pred': '01.2.1.0.3',  # Level 4: 01.2.1.0
        'code': '01.2.1.0.1',          # Level 4: 01.2.1.0 (match!)
        'list_retrieved_codes': [
            '01.2.1.0.1',  # Exact label is here
            '01.2.1.0.2',
            '01.2.2.0.0'
        ],
        'confidence': 0.91,
        'parsed': True,
        'codable': True
    },
    
    # Case 6: LLM correct, label in retrieved but at different precision
    {
        'product': 'Yaourt nature',
        'coicop_pred': '01.1.4.2.0',
        'code': '01.1.4.2.0',
        'list_retrieved_codes': [
            '01.1.4.2',    
            '01.1.4.0.0',
            '01.1.5.0.0'
        ],
        'confidence': 0.88,
        'parsed': True,
        'codable': True
    },
    
    # Case 7: LLM wrong at level 4, label in retrieved codes
    # Generation error: had the right code but chose wrong
    {
        'product': 'Bière blonde artisanale',
        'coicop_pred': '02.1.1.2.0',  # Level 4: 02.1.1.2 (wrong)
        'code': '02.1.1.1.0',          # Level 4: 02.1.1.1 (correct)
        'list_retrieved_codes': [
            '02.1.1.1.0',  # Correct code IS in retrieved!
            '02.1.1.2.0',  # But LLM picked this
            '02.1.2.0.0'
        ],
        'confidence': 0.72,
        'parsed': True,
        'codable': True
    },
    
    # Case 8: Short code (3 levels), LLM correct at level 4
    {
        'product': 'Essence sans plomb',
        'coicop_pred': '07.2.2',      # Only 3 levels
        'code': '07.2.2',              # Same
        'list_retrieved_codes': [
            '07.2.2',
            '07.2.1.0.0',
            '07.2.3.0.0'
        ],
        'confidence': 0.93,
        'parsed': True,
        'codable': True
    },
    
    # Case 9: LLM correct at level 4 even though codes differ at level 5
    # Label NOT fully in retrieved (but level 4 prefix is there)
    {
        'product': 'Courgettes bio',
        'coicop_pred': '01.1.7.2.9',  # Level 4: 01.1.7.2
        'code': '01.1.7.2.5',          # Level 4: 01.1.7.2 (match!)
        'list_retrieved_codes': [
            '01.1.7.2.1',  # Same level 4 prefix
            '01.1.7.3.0',
            '01.1.6.0.0'
            # Exact label '01.1.7.2.5' NOT in list
        ],
        'confidence': 0.79,
        'parsed': True,
        'codable': True
    },
    
    # Case 10: Not codable, not parsed (edge case)
    {
        'product': 'Service non classifiable',
        'coicop_pred': None,
        'code': '13.9.9.9.9',
        'list_retrieved_codes': [],
        'confidence': 0.0,
        'parsed': False,
        'codable': False
    }
]

result = calculate_accuracy_at_level(
    records_test,
    "coicop_pred",
    "code",
    4,
    "list_retrieved_codes"
)

result_expected = (
    0.6,
    [True, True, False, False, True, True, False, True, True, False],
    0.7,
    0.7142857142857143,
    [True, False, True, False, True, True, True, True, True, False]
)


def test_calculate_accuracy_at_level():
    assert result == result_expected

