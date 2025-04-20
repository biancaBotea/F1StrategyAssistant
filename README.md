# F1 Strategy Assistant – MVP
---

In Formula 1, deciding when to make a pit stop can make or break your race. This decision depends on a whole lot of moving variables that tend to shift in importance depending on the circuit and track conditions. That’s why teams have entire strategy departments—so one person can spend the whole weekend staring at the weather radar and the sky, while someone else dives deep into tyre performance data.

But there’s a layer of data that always matters: lap-level information like tyre degradation, lap times, and the driver’s position relative to others. These factors consistently play a crucial role, no matter the race. That makes them the perfect input for a model that can assist by offering pit-stop advice based purely on these aspects, freeing up the strategist to focus on everything that still needs human judgment. After all, a well-seasoned strategist’s intuition is still the best decision model.

So strategists have to go through mountains of data to make critical race decisions. Pressure is the name of the game. And even then, mistakes happen.

To help with that, why not build a GenAI-powered assistant that acts like a second set of eyes for the F1 strategy team? Well—I did. 

The agent looks at consistent lap-level data—things like tyre compound, tyre life, lap number, current track status—and gives a simple recommendation: pit or stay out. It’s not here to replace anyone. It’s just here to help when the heat is on, offering grounded, data-driven advice so the human strategist can focus on the unpredictable stuff.

This project explores how a GenAI-powered assistant can support F1 strategists by:

- Analyzing race context based on numerical inputs.
- Providing timely pit recommendations grounded in data.
- Reducing cognitive load by automating data-driven decision-making.

---
## How it works

At the heart of this project is a LangChain GenAI agent built to act like a calm, number-crunching co-pilot.

The agent takes in weather and track information as well as live lap-level information in order to decide if to pit or not.

How does the agent make decisions? 

The agent uses a Random Forest Classifier trained on real lap-level data from past races, ensuring the decisions are grounded in actual race data. No guesswork here.

``` python
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

model = RandomForestClassifier(random_state=42, max_depth = 10, min_samples_split = 2, n_estimators = 50, class_weight='balanced')

# Split data into training and test sets (80/20 split).
lapData_train, lapData_test, target_train, target_test = train_test_split(lapData, target, test_size=0.2, random_state=42)

# Fit model
model.fit(lapData_train, target_train)

# Evaluate the model
predictions = model.predict(lapData_test)

# Classification report 
print("\nClassification Report:\n", classification_report(target_test, predictions))
```

The race data for the model is loaded using fastf1. But raw data isn’t race-ready—each lap comes with a few dozen features, and getting that down to a clean, usable set of 12 takes quite a bit of wrangling. From encoding tyre compounds to tracking gaps between drivers, the data goes through a whole lot of processing before it’s model-ready.

``` python
# List of features used in the model
features = [
    "LapNumber",       # The lap number in the race
    "Position",        # The car's position in the race
    "Compound",        # The type of tyres (e.g., soft, medium, hard)
    "TyreLife",        # The number of laps the tyres have been used
    "TrackStatus",     # The condition of the track (e.g., Green, Safety Car)
    "TrackTemp",       # The ambient temperature of the track in °C
    "Rainfall",        # Boolean indicating if it is raining
    "DeltaTime",       # The time difference between current and past lap time
    "GapAhead",        # The time gap to the car ahead
    "PosToLose",       # The position a car could lose with a pit stop
    "TotalLaps",       # Total number of laps in the race
    "RemainingLaps"    # Number of laps remaining in the race
]
```


This model is integrated with function calling through LangChain, which means you can ask the assistant plain-language questions like, “Should I pit?” and it will automatically call the model with the right inputs under the hood.
But that’s not all. The agent can also store and recall track and weather conditions that you’ve provided earlier, making it a seamless tool during a race.

``` python
from langchain_core.tools import tool

@tool
def store_total_lap_number(total_lap_number: float) -> str:
    """Stores the total number of laps in the race
    
    Returns:
        Confirmation message 
    """

@tool
def get_total_lap_number() -> str:
    """Returns the total number of laps in the race"""

@tool
def store_track_status(track_status: str) -> str:
    """Store the current track status (Green, Safety Car, Virtual Safety Car)   
    
    Returns:
        Confirmation message 
    """

@tool
def get_track_status() -> str:
    """Return the current track status"""

@tool
def update_weather(track_temp: float, rainfall: float) -> str:
    """Update the current weather conditions

    Returns:
        Confirmation message.
    """
    
@tool
def get_weather() -> str:
    """Returns the latest weather conditions"""


@tool
def pit_check(lap_number: float, position: float, tyre_life: float, compound: str, delta_time: float, gap_ahead: float, pos_to_lose: float) -> [float]:
    """Pass lap and race conditions to the model and get a pit decision

    Returns:
        Model prediction and confidence

    """
```

To make the agent smarter and provide more useful feedback, I incorporated few-shot prompting. This helps the agent understand how to interpret the model’s output and offer more than just a recommendation. Instead, it explains why a particular decision is being made, ensuring that the strategist knows the reasoning behind each suggestion.

(This was also super helpful when I realized my model wasn't performing as well as I'd hoped. Ah, the joys of debugging.)

``` python
"""
    Example 1 – Pit under normal conditions:  

    Human input:
    We’re on lap 21, currently P3, running 20-lap-old Mediums. 
    Our lap delta is -0.13, the gap to the car ahead is 5.97 seconds, and if we box now, we’ll lose 7 positions.

    Response:  
    Pit, confidence 76%. 
    Reason: BOX, BOX. \n Tyres are reaching the end of their performance window. Pitting now helps minimize position loss before others stop.
    
    ---
    
    Example 2 – Pit in wet conditions:  
    Human input:
    Lap 44. P1. 10-lap Hards. Rain started. Delta +1.2. Gap 0. Lose 3 if we pit. 42 to go.
    
    Response:  
    Pit, confidence 91%.
    Reason: BOX, BOX. \n Rain is affecting grip, and loss is acceptable given 42 laps remaining.
    
    ---
    
    Example 3 – Stay out under Safety Car:  
    Human input:
    Lap 3, P5. Safety Car. Mediums, 2 laps. Gap 1.3, delta -0.1. Pit drops us 5. 55 left.
    
    Response:  
    Stay out, confidence 80%. 
    Reason: STAY OUT. \n It’s early in the race and tyres are fresh. Pitting now would lose positions without strategic advantage.
    
    ---
    
    Example 4 – Stay out late in race: 
    Human input:
    We're on lap 46, running P6. Tyres are 20-lap-old hards, still holding up. 
    Gap to car ahead is 2.2, delta's -0.3. If we pit, we lose 2 spots. Only 7 laps left.
    
    Response:  
    Stay out, confidence 84%. 
    Reason: STAY OUT. \n Tyres are holding up and pitting now would cost positions without meaningful advantage.
"""
```

And so the agent can be the best helper it can be, it records each recommendation along with the lap information in a structured JSON format. This makes it easy to parse and log, so strategists can review and analyze decisions after the race with no guesswork.

``` python
  [
    {
        "LapNumber": 23,
        "Position": 4,
        "Compound": 3,
        "TyreLife": 10,
        "TrackStatus": 0,
        "TrackTemp": 28,
        "Rainfall": 0,
        "DeltaTime": 2.5,
        "GapAhead": 1.8,
        "PosToLose": 2,
        "TotalLaps": 50,
        "RemainingLaps": 27
        "Decision" : "BOX, BOX'
        "Confidence": 0.81
    },
    ...
  ]
```

---

## Challenges

- **Limited Training Data**  
  The model was trained using data from only two races on a single circuit (Saudi Arabia), which limits its generalizability. It tends to overfit and may perform poorly on unseen tracks.

- **Missing Contextual Features**  
  Key variables like fuel load, tyre temperature, and driver-specific behavior were not available in the dataset. These could significantly improve prediction quality in future iterations.

- **Integration vs. Accuracy**  
  The primary goal of this project was to demonstrate a working GenAI-powered agent pipeline. As such, predictive accuracy was not fully optimized.

---

> **Disclaimer:**  
> This MVP focuses on showcasing GenAI capabilities and agent integration. While it provides valuable insights, the model's recommendations may not generalize well across all races. Improving accuracy and robustness is part of the planned future work.
---

# Summary

In Formula 1, pit stop timing can win or lose you a race. It’s a high-stakes decision built on fast-moving variables—weather shifts, tyre wear, track incidents—that demand constant attention. Strategy teams have to juggle all this while staying calm under pressure.

This project introduces a GenAI-powered assistant designed to take some of that pressure off. It doesn’t try to replace the strategist—it just takes care of the heavy number crunching, offering quick, data-grounded recommendations on whether to pit or stay out.

The assistant works off a consistent layer of lap-level data—tyre compound, tyre life, lap number, track status, and more—making decisions based on patterns learned from real race situations. It's a calm, dependable voice when things get chaotic.

### GenAI Capabilities Used

- **Agents**  
  The assistant functions as a "Strategy Copilot", using a LangChain agent framework to process real-time race data and return pit decisions with supporting reasoning.

- **Few-Shot Prompting**  
  Few-shot examples help guide the assistant’s responses so it not only gives a recommendation but explains why—mirroring the kind of feedback a human strategist would expect.

- **Grounding**  
  All decisions are based on real F1 race data, ensuring outputs are relevant and trustworthy. No fantasy racing logic here—just numbers and patterns from the real world.

- **Structured Output / JSON Mode**  
  Every pit recommendation is logged in JSON, making it easy to track, audit, or feed into dashboards for deeper analysis after the race.

> This MVP shows how GenAI can support strategy teams by simplifying complex, data-heavy decisions. Future improvements will bring in more nuance—tyre temperatures, evolving weather, and circuit-specific behaviors—to make the assistant even smarter and more adaptable.

At the end of the day, it’s about giving strategists space to do what they do best: apply human judgment in the grey areas where no model can go.

---

## Future Work

While the current GenAI-powered Strategy Assistant offers valuable support for Formula 1 strategists, there’s still plenty of room to improve and expand its capabilities. Here’s a look at some key areas for future development:

### 1. **Boosting Model Performance**
   - **More Features**  
     The assistant could benefit from a wider range of race-related features like pit stop windows, available tyre compounds, real-time telemetry (lap times, sector times, driver behavior), and more. The addition of these features would allow the model to better understand the race dynamics and improve its decision-making.

   - **Exploring New Models**  
     Testing out more advanced machine learning models (such as deep learning or ensemble methods) could increase the assistant’s predictive power and accuracy. This would involve fine-tuning hyperparameters and performing cross-validation to ensure the model performs consistently across different race scenarios.

### 2. **Real-Time Race Data Integration**
   - Currently, the assistant works with static, pre-loaded race data. For real-world use, integrating live race feeds and telemetry would enable the assistant to provide real-time decision-making support. This could involve pulling data from sensors, GPS, and communication systems in F1 cars, creating a truly responsive tool.

### 3. **Context-Aware Decision Making**
   - Right now, the assistant bases pit recommendations on numerical inputs alone. Future versions could integrate more nuanced decision-making by taking into account psychological factors and strategic insights from human race engineers. For example, the AI could offer recommendations tailored to counter specific driver behaviors or strategies based on competitor performance.

### 4. **Adapting to Different Circuits**
   - The model is currently trained on a limited set of circuits. Expanding the dataset to cover a wider variety of tracks, each with different weather conditions and layouts, would make the assistant more versatile. It would be able to offer more accurate recommendations across the entire F1 season, adapting to each unique challenge.

### 5. **Improved Human-AI Collaboration**
   - A big part of the future of this assistant lies in enhancing the collaboration between human strategists and AI. The assistant could be designed to generate multiple strategies or alternatives, allowing the strategist to choose the best option based on their own experience and race knowledge. This synergy between human judgment and AI-powered insights could lead to better decision-making and stronger race strategies.

By improving in these areas, the GenAI-powered Pit Wall Assistant can evolve into an even more powerful, reliable, and adaptable tool, providing F1 strategists with critical support in high-stakes situations and ultimately helping to refine race strategies.
