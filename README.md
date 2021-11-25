# Reinforcement-Learning-for-PyTuxKart
This is the final project for EC400 of Justin Melville, Jack Halberian, and Wang Zhaoze. The goal of this project is to improve the track times as much as possible using Deep Q-Learning and Neural Networks, as well as other methods. 

### RL
To train RL
```
cd pytuxkart
python actcri_agent.py
```
Current RL only train on the steering

##### Todo
[ ] Need upgrade reward, current reward function isn't good enough
[ ] Max score should display max traveled distance
[ ] Off track != done, the game shouldn't end when off track
[ ] actor critics should be positive