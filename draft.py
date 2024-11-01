import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define Antecedents (Inputs)
disposable_income = ctrl.Antecedent(np.arange(-10000, 20001, 1), 'disposable_income')
item_price = ctrl.Antecedent(np.arange(0, 100001, 1), 'item_price')
savings = ctrl.Antecedent(np.arange(0, 100001, 1), 'savings')
credit_score = ctrl.Antecedent(np.arange(300, 851, 1), 'credit_score')

# Define Consequents (Outputs)
affordability = ctrl.Consequent(np.arange(0, 101, 1), 'affordability')
risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

# Membership functions for disposable_income
disposable_income['very_low'] = fuzz.trapmf(disposable_income.universe, [-10000, -10000, 0, 3000])
disposable_income['low'] = fuzz.trimf(disposable_income.universe, [0, 3000, 6000])
disposable_income['medium'] = fuzz.trimf(disposable_income.universe, [6000, 10000, 14000])
disposable_income['high'] = fuzz.trimf(disposable_income.universe, [13000, 20000, 20000])

#  Membership functions for item_price
item_price['low'] = fuzz.trimf(item_price.universe, [0, 0, 10000])
item_price['medium'] = fuzz.trimf(item_price.universe, [10000, 15000, 20000])
item_price['high'] = fuzz.trapmf(item_price.universe, [20000, 40000, 60000, 100000])

# Membership functions for savings
savings['low'] = fuzz.trimf(savings.universe, [0, 0, 20000])
savings['medium'] = fuzz.trimf(savings.universe, [15000, 40000, 65000])
savings['high'] = fuzz.trapmf(savings.universe, [60000, 80000, 100000, 100000])

# Membership functions for credit_score
credit_score['poor'] = fuzz.trimf(credit_score.universe, [300, 300, 580])
credit_score['fair'] = fuzz.trimf(credit_score.universe, [550, 610, 670])
credit_score['good'] = fuzz.trimf(credit_score.universe, [650, 700, 750])
credit_score['excellent'] = fuzz.trapmf(credit_score.universe, [720, 750, 850, 850])

# Membership functions for affordability
affordability['very_low'] = fuzz.trimf(affordability.universe, [0, 0, 10])
affordability['low'] = fuzz.trimf(affordability.universe, [10, 30, 50])
affordability['medium'] = fuzz.trimf(affordability.universe, [40, 60, 80])
affordability['high'] = fuzz.trapmf(affordability.universe, [70, 90, 100, 100])

# Membership functions for risk
risk['low'] = fuzz.trimf(risk.universe, [0, 0, 40])
risk['medium'] = fuzz.trimf(risk.universe, [30, 50, 70])
risk['high'] = fuzz.trapmf(risk.universe, [60, 80, 100, 100])

risk.defuzzify_method = 'mom'

# Affordability Rules
rule_a0 = ctrl.Rule(
    disposable_income['very_low'] & item_price['low'],
    (affordability['low'])
)
rule_a1 = ctrl.Rule(
    disposable_income['very_low'] & item_price['high'],
    (affordability['very_low'])
)
rule_a2 = ctrl.Rule(
    disposable_income['low'] & item_price['high'],
    (affordability['low'])
)
rule_a3 = ctrl.Rule(
    disposable_income['very_low'] & item_price['medium'],
    (affordability['low'])
)
rule_a4 = ctrl.Rule(
    disposable_income['low'] & item_price['medium'],
    (affordability['medium'])
)
rule_a5 = ctrl.Rule(
    disposable_income['medium'] & item_price['medium'],
    (affordability['medium'])
)
rule_a6 = ctrl.Rule(
    disposable_income['medium'] & item_price['low'],
    (affordability['high'])
)
rule_a7 = ctrl.Rule(
    disposable_income['high'] & item_price['low'],
    (affordability['high'])
)
rule_a8 = ctrl.Rule(
    disposable_income['high'] & item_price['medium'],
    (affordability['high'])
)
rule_a9 = ctrl.Rule(
    disposable_income['high'] & item_price['high'],
    affordability['low']
)
rule_a10 = ctrl.Rule(
    disposable_income['medium'] & item_price['high'],
    (affordability['low'])
)
rule_a11 = ctrl.Rule(
    disposable_income['low'] & item_price['low'],
    (affordability['medium'])
)
rule_a12 = ctrl.Rule(
    disposable_income['medium'] & item_price['medium'],
    (affordability['medium'])
)
rule_a13 = ctrl.Rule(
    disposable_income['very_low'] & item_price['high'],
    affordability['very_low']
)
rule_a14 = ctrl.Rule(
    disposable_income['medium'] & item_price['high'],
    affordability['low']
)
rule_a15 = ctrl.Rule(
    disposable_income['high'] & item_price['high'],
    affordability['very_low']
)


# Risk Rules
rule_r1 = ctrl.Rule(savings['high'] & ~disposable_income['very_low'], risk['low'])
rule_r2 = ctrl.Rule(savings['medium'] & (credit_score['good'] | credit_score['excellent']), risk['low'])
rule_r3 = ctrl.Rule(savings['low'] & (credit_score['poor'] | credit_score['fair']), risk['high'])
rule_r4 = ctrl.Rule(savings['medium'] & (credit_score['poor'] | credit_score['fair']), risk['high'])
rule_r5 = ctrl.Rule(savings['low'] & credit_score['excellent'], risk['medium'])
rule_r6 = ctrl.Rule(savings['medium'] & item_price['high'], risk['high'])
rule_r7 = ctrl.Rule(savings['high'] & item_price['high'], risk['high'])
rule_r8 = ctrl.Rule(
    disposable_income['very_low'] & credit_score['poor'],
    risk['high']
)
rule_r9 = ctrl.Rule(
    savings['low'] & disposable_income['very_low'],
    risk['high']
)
rule_r10 = ctrl.Rule(
    savings['medium'] & credit_score['excellent'],
    risk['low']
)
rule_r11 = ctrl.Rule(
    credit_score['excellent'],
    risk['medium']
)
rule_r12 = ctrl.Rule(savings['low'] & credit_score['good'], risk['medium'])
rule_r13 = ctrl.Rule(disposable_income['very_low'], risk['high'])
rule_r14 = ctrl.Rule(
    savings['low'] & (credit_score['good'] | credit_score['excellent']),
    risk['high']
)
rule_r15 = ctrl.Rule(
    item_price['high'] & savings['medium'],
    risk['high']
)


catch_all_rule = ctrl.Rule(
    ~(
        disposable_income['very_low'] | disposable_income['low'] |
        disposable_income['medium'] | disposable_income['high']
    ) & ~(
        item_price['low'] | item_price['medium'] | item_price['high']
    ) & ~(
        savings['low'] | savings['medium'] | savings['high']
    ) & ~(
        credit_score['poor'] | credit_score['fair'] |
        credit_score['good'] | credit_score['excellent']
    ),
    (affordability['very_low'], risk['medium'])
)

# Combine all rules
rules = [
    rule_a0, rule_a1, rule_a2, rule_a3, rule_a4, rule_a5, rule_a6, rule_a7, rule_a8, rule_a9,
    rule_a10, rule_a11, rule_a12, rule_a13,   
    rule_r1, rule_r2, rule_r3, rule_r4, rule_r5, rule_r6, rule_r7, rule_r8, rule_r9, rule_r10, rule_r11, rule_r12, rule_r13,
    catch_all_rule  
]


financial_ctrl = ctrl.ControlSystem(rules)
financial_simulation = ctrl.ControlSystemSimulation(financial_ctrl)

# Test function for edge cases
def test_edge_case(disposable_income_input, item_price_input, savings_input, credit_score_input, expected_affordability, expected_risk):
    financial_simulation.input['disposable_income'] = disposable_income_input
    financial_simulation.input['item_price'] = item_price_input
    financial_simulation.input['savings'] = savings_input
    financial_simulation.input['credit_score'] = credit_score_input

    financial_simulation.compute()
    
    print(f"Test Case: ")
    print(f"  Disposable Income: {disposable_income_input}")
    print(f"  Item Price: {item_price_input}")
    print(f"  Savings: {savings_input}")
    print(f"  Credit Score: {credit_score_input}")
    print(f"  Expected Affordability Index: {expected_affordability}")
    print(f"  Expected Financial Risk Level: {expected_risk}")
    print(f"  Computed Affordability Index: {financial_simulation.output['affordability']:.2f}")
    print(f"  Computed Financial Risk Level: {financial_simulation.output['risk']:.2f}\n")

# List of edge cases to test
edge_cases = [
    # Edge Case 1: Negative Disposable Income
    (-2000, 5000, 80000, 400, 0.0, 100.0),
    
    # Edge Case 2: High Disposable Income with Expensive Item
    (15000, 90000, 50000, 750, 15.00, 70.45),
    
    # Edge Case 3: High Income, Low Item Price
    (18000, 3000, 80000, 800, 89.17, 35.82),
    
    # Edge Case 4: Low Income, Low Item Price
    (2000, 1000, 5000, 600, 50.00, 70.00),  
    
    # Edge Case 5: Medium Income, Medium Item Price, Low Savings
    (8000, 15000, 2000, 680, 60.00, 80.00), 
    
    # Edge Case 6: High Savings, Low Income, High Item Price
    (1000, 40000, 80000, 720, 1.75, 51.32),
    
    # Edge Case 7: Max Credit Score, Low Savings, Expensive Item
    (7000, 50000, 1000, 850, 20.00, 60.00) 
]

# Run all edge cases
for case in edge_cases:
    test_edge_case(*case)
