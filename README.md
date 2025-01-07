PPO Chess Agent: Training and Insights
This repository is focused on training a Proximal Policy Optimization (PPO) agent to play chess and tools to analyze its learning process. 
The primary aim is to build a chess-playing bot that learns from rewards and improves over time. Additionally, visualization tools are provided to monitor the progress.
Bots killing Themselves in 2-3 moves 
During training, earlier iterations of the bots exhibited behaviour where they would intentionally lose the game to avoid accumulating further losses, as seen in Suicidal bots.
This behaviour was caused by overly long-term focus due to high discount factors (gamma) and Inconsistent reward structures that penalized too harshly for temporary disadvantages.
I Adjusted gamma to balance short-term and long-term rewards, 
Refined the reward system to separately reward immediate move outcomes instead of only focusing on endgame results 
and added Penalty adjustments to discourage illegal moves without overly punishing strategic risks.
But it didn't work out well and still looking for a possible reward change that gives the bots the incentive to keep the game going
