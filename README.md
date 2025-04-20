# F1 Strategy Assistant – MVP
---

In Formula 1, deciding when to make a pit stop can make or break your race. This decision depends on a whole lot of moving variables that tend to shift in importance depending on the circuit and track conditions. That’s why teams have entire strategy departments—so one person can spend the whole weekend staring at the weather radar and the sky, while someone else dives deep into tyre performance data.

But there’s a layer of data that always matters: lap-level information like tyre degradation, lap times, and the driver’s position relative to others. These factors consistently play a crucial role, no matter the race. That makes them the perfect input for a model that can assist by offering pit-stop advice based purely on these aspects, freeing up the strategist to focus on everything that still needs human judgment. After all, a well-seasoned strategist’s intuition is still the best decision model.

So strategists have to go through mountains of data to make critical race decisions. Pressure is the name of the game. And even then, mistakes happen.
To help with that, why not build a GenAI-powered assistant that acts like a second set of eyes for the F1 strategy team?
Well—I did.
The agent looks at consistent lap-level data—things like tyre compound, tyre life, lap number, current track status—and gives a simple recommendation: pit or stay out.
It’s not here to replace anyone. It’s just here to help when the heat is on, offering grounded, data-driven advice so the human strategist can focus on the unpredictable stuff.

This project explores how a GenAI-powered assistant can support F1 strategists by:

- Analyzing race context based on numerical inputs.
- Providing timely pit recommendations grounded in data.
- Reducing cognitive load by automating data-driven decision-making.

---
## How it works

At the heart of this project is a LangChain GenAI agent built to act like a calm, number-crunching co-pilot.

The agent takes in weathe and track information as well as live lap-level information in ordeer to decide if to pit or not.

How does the agent make decisions? well it uses a Random Forest Classifier trained on real lap-level data from past races
So the decisions are grounded in real race data. no guess work here.

<!-- insert model snippet here -->

and it is abble to do this using function calling. This lets you ask the assistant “Should I pit?” in plain language, and it calls the model with the right inputs under the hood. but this guy can also store and give u info on the track and weather conditions that u told it previuosely.

<!-- insert tool definition snippet -->

on top of the general system descrition adding few-shot prompting, helps the agent get an idea of how to interpret the models output, so it can provide the strategist with a reason for it s decision, not jsut drop a percentage and be done.(that was also usefull in me figring out the my model is not that great, what a fun day taht was "crying sad face emoji" )

<!-- few-shot examples snippet -->

and to be the best helper it can the agent replies in structured JSON format, so the recommendation is easy to parse and log—perfect for reviewing strategy decisions after the race.

<!-- json output snippet -->



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

## Summary

This project demonstrates the potential of GenAI to assist Formula 1 strategists in making critical pit stop decisions during live races. By leveraging lap-level data such as tyre conditions, track status, lap number, and more, the assistant offers timely, data-driven recommendations on whether to pit or stay out.

By combining a Random Forest model with few-shot prompting, function-calling agents, structured JSON output, and grounding in real race data, we’ve built an MVP that lays the groundwork for more advanced, real-time racing tools.

### GenAI Capabilities Used

- **Agents**  
  The assistant is structured as a GenAI-powered agent that serves as a "Pit Wall Assistant", ingesting real-time race data and returning pit stop decisions based on learned patterns.

- **Few-Shot Prompting**  
  Few-shot prompting demonstrates the model’s outputs with example cases, helping to contextualize decisions in varied race situations.

- **Grounding**  
  The assistant is grounded in real race data, ensuring that its predictions reflect realistic scenarios and provide actionable insights.

- **Structured Output / JSON Mode**  
  Recommendations are returned in a structured JSON format, enabling easy evaluation of the model's decisions and providing a foundation for post-race analysis or integration into live dashboards.

> While the primary focus was on building and demonstrating the GenAI agent, future work will focus on improving model performance by incorporating more contextual features (like tyre temperature, fuel load, and weather changes) and enhancing generalizability across different tracks and seasons.

By reducing the strategist’s cognitive load and automating data-heavy tasks, this assistant enables humans to focus on the nuanced, psychological, and reactive aspects of race strategy—where human expertise shines.


---

## Future Work

While the current GenAI-powered Pit Wall Assistant offers valuable support for Formula 1 strategists, there are several areas for improvement and expansion. The following outlines potential directions for future work:

### 1. **Enhanced Model Performance**
   - **Feature Expansion**: The current model's performance could be improved by incorporating additional race-related features such as pit stop windo, available tyres, real-time telemetry data (e.g., lap times, sector times, driver behavior) and others. Including more features would provide the model with a more comprehensive understanding of the race dynamics and improve decision-making.
   - **Alternative Models**: Exploring other machine learning models, such as deep learning-based architectures or ensemble methods, could potentially enhance the predictive power and accuracy of the assistant. This would also involve hyperparameter tuning and cross-validation to optimize the model's generalization across different race scenarios.

### 2. **Integration with Real-Time Race Data**
   - The assistant currently operates with static, pre-loaded race data. For real-world applications, integrating the assistant with live race feeds and telemetry would allow it to provide real-time decision-making assistance. This could involve real-time data ingestion from sensors, GPS, and communication systems within F1 teams.

### 3. **Context-Aware Decision Making**
   - The current model generates pit recommendations based purely on numerical inputs. Future iterations could incorporate more nuanced decision-making capabilities by integrating psychological factors and strategic insights from human race engineers. For example, AI could provide insights based on how competitors are performing and suggest strategies to counter specific drivers' behavior.

### 4. **Adaptive Strategies for Different Circuits**
   - Currently, the model is trained on a limited number of circuits. Expanding the dataset to include more tracks, particularly those with different weather conditions and layouts, will enable the assistant to adapt its recommendations to a wider range of environments. This would improve its utility across the entire F1 season and make it more versatile.

### 5. **Human-AI Collaboration**
   - Another key area of future work is improving the collaboration between human strategists and the AI. The assistant can be designed to generate multiple strategies or alternatives, allowing the strategist to select the best option based on their judgment and knowledge of the race. This human-AI synergy could lead to more effective decision-making and better outcomes.

By addressing these areas, the GenAI-powered Pit Wall Assistant can be refined into a more powerful, reliable, and adaptable tool that provides substantial support to F1 strategists, ultimately enhancing race strategies and decision-making.
