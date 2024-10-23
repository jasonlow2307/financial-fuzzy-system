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
disposable_income['very_low'] = fuzz.trapmf(disposable_income.universe, [-10000, -10000, 0, 4000])
disposable_income['low'] = fuzz.trimf(disposable_income.universe, [2000, 4000, 6000])
disposable_income['medium'] = fuzz.trimf(disposable_income.universe, [5000, 10000, 15000])
disposable_income['high'] = fuzz.trapmf(disposable_income.universe, [14000, 20000, 20000, 20000])

# Membership functions for item_price
item_price['low'] = fuzz.trimf(item_price.universe, [0, 0, 15000])
item_price['medium'] = fuzz.trimf(item_price.universe, [10000, 20000, 30000])
item_price['high'] = fuzz.trapmf(item_price.universe, [20000, 25000, 100000, 100000])

# Membership functions for savings
savings['low'] = fuzz.trimf(savings.universe, [0, 0, 20000])
savings['medium'] = fuzz.trimf(savings.universe, [15000, 40000, 65000])
savings['high'] = fuzz.trapmf(savings.universe, [60000, 80000, 100000, 100000])

# Membership functions for credit_score
credit_score['poor'] = fuzz.trimf(credit_score.universe, [300, 300, 580])
credit_score['fair'] = fuzz.trimf(credit_score.universe, [550, 610, 670])
credit_score['good'] = fuzz.trimf(credit_score.universe, [650, 700, 740])
credit_score['excellent'] = fuzz.trapmf(credit_score.universe, [720, 850, 850, 850])

# Membership functions for affordability
affordability['very_low'] = fuzz.trimf(affordability.universe, [0, 0, 5])
affordability['low'] = fuzz.trimf(affordability.universe, [5, 15, 25])
affordability['medium'] = fuzz.trimf(affordability.universe, [20, 50, 80])
affordability['high'] = fuzz.trapmf(affordability.universe, [70, 100, 100, 100])

# Membership functions for risk
risk['low'] = fuzz.trimf(risk.universe, [0, 0, 30])
risk['medium'] = fuzz.trimf(risk.universe, [20, 50, 80])
risk['high'] = fuzz.trapmf(risk.universe, [70, 85, 100, 100])

# Define additional rule for negative disposable income
rule_a0 = ctrl.Rule(
    disposable_income['very_low'] & item_price['low'],
    (affordability['very_low'], risk['high'])
)

# Define combined rules
# Affordability Rules
rule_a1 = ctrl.Rule(
    disposable_income['very_low'] & item_price['high'],
    (affordability['very_low'], risk['high'])
)
rule_a2 = ctrl.Rule(
    disposable_income['low'] & item_price['high'],
    (affordability['very_low'], risk['high'])
)
rule_a3 = ctrl.Rule(
    disposable_income['very_low'] & item_price['medium'],
    (affordability['very_low'], risk['high'])
)
rule_a4 = ctrl.Rule(
    disposable_income['low'] & item_price['medium'],
    (affordability['low'], risk['high'])
)
rule_a5 = ctrl.Rule(
    disposable_income['medium'] & item_price['medium'],
    (affordability['low'], risk['high'])
)
rule_a6 = ctrl.Rule(
    disposable_income['medium'] & item_price['low'],
    (affordability['medium'], risk['medium'])
)
rule_a7 = ctrl.Rule(
    disposable_income['high'] & item_price['low'],
    (affordability['high'], risk['medium'])
)
rule_a8 = ctrl.Rule(
    disposable_income['high'] & item_price['medium'],
    (affordability['medium'], risk['medium'])
)
rule_a9 = ctrl.Rule(
    disposable_income['high'] & item_price['high'],
    (affordability['low'], risk['high'])
)
rule_a10 = ctrl.Rule(
    disposable_income['medium'] & item_price['high'],
    (affordability['very_low'], risk['high'])
)


# Risk Rules
# Rule: High savings always lead to low risk
rule_r1 = ctrl.Rule(savings['high'], risk['low'])

# Rule: Medium savings with good or excellent credit score lead to medium risk
rule_r2 = ctrl.Rule(savings['medium'] & (credit_score['good'] | credit_score['excellent']), risk['medium'])

# Rule: Low savings lead to high risk regardless of other factors
rule_r3 = ctrl.Rule(savings['low'], risk['high'])

# Rule: Medium savings with poor or fair credit score lead to high risk
rule_r4 = ctrl.Rule(savings['medium'] & (credit_score['poor'] | credit_score['fair']), risk['high'])

# Rule: Low savings with excellent credit score lead to medium risk
rule_r5 = ctrl.Rule(savings['low'] & credit_score['excellent'], risk['medium'])

# Rule: Medium savings and high item price increase risk
rule_r6 = ctrl.Rule(savings['medium'] & item_price['high'], risk['high'])

# Rule: High savings and high item price result in medium risk
rule_r7 = ctrl.Rule(savings['high'] & item_price['high'], risk['medium'])

rule_r8 = ctrl.Rule(
    disposable_income['very_low'] & credit_score['poor'],
    risk['high']
)
rule_r9 = ctrl.Rule(
    savings['low'] & disposable_income['very_low'],
    risk['high']
)


# Combine all rules
rules = [
    rule_a0, rule_a1, rule_a2, rule_a3, rule_a4, rule_a5, rule_a6, rule_a7, rule_a8, rule_a9, rule_a10,
    rule_r1, rule_r2, rule_r3, rule_r4, rule_r5, rule_r6, rule_r7, rule_r8, rule_r9
]

# Create a single control system
financial_ctrl = ctrl.ControlSystem(rules)
financial_simulation = ctrl.ControlSystemSimulation(financial_ctrl)

# Provide inputs
financial_simulation.input['disposable_income'] = 7000
financial_simulation.input['item_price'] = 50000
financial_simulation.input['savings'] = 1000
financial_simulation.input['credit_score'] = 850

# Compute outputs
try:
    financial_simulation.compute()
    print(f"Affordability Index: {financial_simulation.output['affordability']:.2f}")
    print(f"Financial Risk Level: {financial_simulation.output['risk']:.2f}")
except Exception as e:
    print(f"An error occurred during computation: {e}")
