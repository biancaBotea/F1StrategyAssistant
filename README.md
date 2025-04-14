# F1 Strategy Assistant — MVP

A GenAI‑powered “Strategy Assistant” that recommends whether to pit or stay out, given in‑race conditions. Built as a Kaggle Notebook Capstone Project for the 5‑day Gen AI Intensive Course with Google.

---

## Project Overview

- **Goal:** Given live race inputs (start position, lap number, tyre compound, tyre life, stops so far, track temperature, wet/dry flag, and track identifier), predict whether the car should pit this lap or stay out.
- **Core Components:**
  1. **Data Preparation:** FastF1 lap‑level dataset for two seasons, filtered to 3–5 key circuits.
  2. **Prediction Model:** `RandomForestClassifier` to predict pit vs. stay.
  3. **GenAI Agent:**  
     - **Function Calling**: Wraps the trained model as `should_pit(...)`.  
     - **Few‑Shot Prompting**: Demonstrates model outputs with examples.  
     - **Structured Output/JSON Mode**: Returns recommendations in JSON.

---

## Future Work

- Expand to more circuits & seasons for broader generalization.
- Improve prediction model with gradient‑boosted trees or neural nets.