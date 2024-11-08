﻿# Financial Assessment System

## Introduction

### Problem Definition

Many people face challenges in making sound financial decisions, leading to financial distress and diminished quality of life. This issue is heightened by low levels of financial literacy, where people struggle to manage income, control spending, and save effectively. Traditional financial models often lack flexibility and fail to address real-life financial complexities, leaving many individuals vulnerable to long-term instability, debt, and limited economic growth.

A recent study shows that 57% of Americans scored below 50% on a financial literacy test, illustrating a major gap in essential financial understanding ([Yahoo Finance, 2023](https://finance.yahoo.com/news/financial-illiteracy-epidemic-57-americans-133012616.html)). This project aims to address this challenge by offering a flexible and comprehensive financial assessment system to guide better decision-making.

### Contribution to Sustainable Development Goals (SDGs)

This project aligns with three key SDGs:

- **SDG 1 (No Poverty):** By improving financial decision-making, the system helps users avoid financial distress and poverty, especially benefiting those vulnerable to financial hardship.
- **SDG 4 (Quality Education):** The system fosters financial literacy by providing clear, understandable analyses, empowering users with essential knowledge for responsible financial management.
- **SDG 8 (Decent Work and Economic Growth):** Encouraging responsible financial habits supports sustainable economic growth, economic stability, and individual productivity.

## Motivation and Existing Solutions

Financial decision-making is complex, particularly with fluctuating income and unexpected expenses. Traditional financial tools often provide rigid, binary outcomes (e.g., "approved" or "denied") that do not capture the nuanced reality of personal finances. This project leverages fuzzy logic to create a system that adapts to varying financial factors, such as credit score, savings, and disposable income, for more balanced and realistic evaluations. The system empowers users with insights reflecting their unique financial situations, enabling more informed and confident decisions.

## System Design

### Overview

The fuzzy logic system comprises four main inputs (disposable income, item price, savings, and credit score) and two outputs (affordability and risk), structured as follows:

#### Inputs:

- **Disposable Income:** Ranges from -10,000 to 20,000, representing available funds after expenses.
- **Item Price:** Ranges from 0 to 100,000, indicating the cost of an item in relation to income.
- **Savings:** Ranges from 0 to 100,000, showing financial stability.
- **Credit Score:** Ranges from 300 to 850, indicating creditworthiness and risk.

#### Outputs:

- **Affordability:** Ranges from -10 to 100, assessing if an item is financially feasible.
- **Risk:** Ranges from 0 to 100, reflecting the financial risk of making a purchase.

#### Defuzzification Methods:

- **Centroid** is used for Affordability to provide a balanced, gradual transition between levels, ideal for reflecting nuanced financial strain.
- **Mean of Maxima (MOM)** is used for Risk, focusing on the highest membership values for a balanced assessment.

### Fuzzy Rules

The system utilizes fuzzy rules to provide a more realistic view of affordability and risk:

- **Affordability Rules:** Assess the likelihood of affording an item based on income and price.
- **Risk Rules:** Evaluate financial risk based on savings, income, price, and credit score.
- **Catch-All Rule:** Handles cases not explicitly covered by other rules, ensuring a comprehensive safety net.

### Alignment with SDGs

- **SDG 1:** The system's catch-all rule and adaptable design help those with limited or fluctuating income maintain financial stability.
- **SDG 4:** The system educates users by providing transparent output values that clarify complex financial concepts.
- **SDG 8:** Integrating multiple financial inputs fosters responsible decision-making, promoting economic growth and financial stability.

## Evaluation

### Case Study

Test scenarios simulate diverse financial situations to evaluate the system’s response, assessing affordability and risk based on various combinations of income, savings, credit score, and item price. Examples include:

1. **Negative Disposable Income, High Savings:** Demonstrates high financial risk despite savings.
2. **High Disposable Income, Expensive Item:** Shows manageable affordability with moderate risk.
3. **High Savings, Low-Cost Item:** Indicates strong affordability and low risk.

### System Impact on SDGs

- **SDG 1 (No Poverty):** Helps users avoid poverty by making prudent financial choices.
- **SDG 4 (Quality Education):** Enhances financial literacy and awareness.
- **SDG 8 (Decent Work and Economic Growth):** Encourages responsible financial habits for economic stability and growth.

## Discussion

The system outputs two key metrics:

- **Affordability Index:** Evaluates purchasing power by comparing resources to item prices.
- **Financial Risk Level:** Assesses the economic impact of a purchase on long-term financial stability.

These metrics help users understand and navigate financial decisions, highlighting the system’s potential in promoting responsible financial habits and sustainable economic practices.

## Conclusion

This financial assessment system provides an adaptive approach to complex financial decisions, aligning with SDGs 1, 4, and 8. By integrating key financial inputs, the system offers insights that help users achieve economic stability, literacy, and growth. This project is a step toward a more equitable, educated, and economically stable society, aligning with sustainable development goals and fostering positive change.

## References

- Yahoo Finance. (2023). _Financial illiteracy epidemic: 57% of Americans fail basic financial literacy test_. Retrieved from [Yahoo Finance](https://finance.yahoo.com/news/financial-illiteracy-epidemic-57-americans-133012616.html)
