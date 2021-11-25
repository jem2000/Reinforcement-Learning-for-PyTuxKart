# Reinforcement-Learning-for-PyTuxKart
This is the final project for EC400 of Justin Melville, Jack Halberian, and Wang Zhaoze. The goal of this project is to improve the track times as much as possible using Deep Q-Learning and Neural Networks, as well as other methods. 

### RL
#### Train & Run
To train RL
```
cd pytuxkart
python actcri_agent.py <trackname> <optional -v> <optional -t>
```
example
```
python actcri_agent.py lighthouse -v # running lighthouse with verbose
python actcri_agent.py zengarden # running zengarden without verbose
python actcri_agent.py zengarden -t # running zengarden ten times (default) with pre-trained weights
```
Current RL only train on the steering
Running without verbose is way faster than running with verbose
#### Reference
Code source from here: [Source](https://github.com/MrSyee/pg-is-all-you-need)
Jupyter Notebook: [colab](https://colab.research.google.com/github/MrSyee/pg-is-all-you-need/blob/master/01.A2C.ipynb)

#### Track list
- zengarden
- lighthouse
- hacienda
- snowtuxpeak
- cornfield_crossing
- scotland

### Todo
- [ ] Organize the code
- [ ] Need upgrade reward, current reward function isn't good enough
- [ ] Max score should display max traveled distance
- [ ] Off track != done, the game shouldn't end when off track
- [ ] Actor critics should be positive
- [x] Fixed aim point could not display
- [x] Enabled argument parsing feature
- [x] Need a test function that can run with pre-trained Actor weights