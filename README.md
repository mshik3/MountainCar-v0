# MountainCar v0 solution
Solution to the OpenAI Gym environment of the MountainCar through Deep Q-Learning

##Background
OpenAI offers a toolkit for practicing and implementing Deep Q-Learning algorithms. (http://gym.openai.com/)
This is my implementation of the MountainCar-v0 environment. 

##Results
Results can be found in train_results.log and test_results.log for the train and test, respectively.

###Training
For the training, I set a threshold of -110 for an average score of the mountain car. The mountain car gets a score of -200 per episode if it doesn't reach the flag. It gets a small boost to its score if it reaches the flag. And it gets more and more points if it gets to the flag fast. I modeled the reward function `(reward + gamma * np.max(next_Q_target))` to train the MountainCar to get to the flag as fast as possible.
```
Episode 743	Time Taken: 31.48 sec	Score: -109.00	State: 0.505025593192	Average Q-Target: -14.1120	Epsilon: 0.001	Average Score: -110.03	
Episode 744	Time Taken: 36.07 sec	Score: -125.00	State: 0.501472864486	Average Q-Target: -60.9826	Epsilon: 0.001	Average Score: -110.14	
Episode 745	Time Taken: 30.07 sec	Score: -104.00	State: 0.521960339024	Average Q-Target: -56.2346	Epsilon: 0.001	Average Score: -110.31	
Episode 746	Time Taken: 26.00 sec	Score: -90.00	State: 0.510257725214	Average Q-Target: -36.6640	Epsilon: 0.001	Average Score: -110.15	
Episode 747	Time Taken: 34.39 sec	Score: -119.00	State: 0.536389897949	Average Q-Target: -11.7476	Epsilon: 0.001	Average Score: -109.70	
Episode 748	Time Taken: 32.40 sec	Score: -112.00	State: 0.520691140417	Average Q-Target: -44.0669	Epsilon: 0.001	Average Score: -109.76	
Episode 749	Time Taken: 31.25 sec	Score: -108.00	State: 0.501927383055	Average Q-Target: -57.2569	Epsilon: 0.001	Average Score: -109.75	
Episode 750	Time Taken: 35.22 sec	Score: -122.00	State: 0.507334402534	Average Q-Target: -46.2420	Epsilon: 0.001	Average Score: -109.35	
Episode 751	Time Taken: 32.34 sec	Score: -112.00	State: 0.519616429384	Average Q-Target: -59.1135	Epsilon: 0.001	Average Score: -109.60	
Model training finished! 
Average Score over last 100 episodes: -109.6	Number of Episodes: 751
```

![Training scores plot](https://github.com/mshik3/MountainCar-v0/blob/master/train_scores_plot.png)

###Testing

```
Iteration: 92	Score: -105.0
Iteration: 93	Score: -105.0
Iteration: 94	Score: -105.0
Iteration: 95	Score: -105.0
Iteration: 96	Score: -104.0
Iteration: 97	Score: -105.0
Iteration: 98	Score: -105.0
Iteration: 99	Score: -105.0
Iteration: 100	Score: -103.0
Total Avg. Score over 100 consecutive iterations : -102.84
Agent finished test within expected reward boundary! Environment is solved.
```

![Testing scores plot](https://github.com/mshik3/MountainCar-v0/blob/master/test_scores_plot.png)
![Testing gif](https://github.com/mshik3/MountainCar-v0/blob/master/output_dp0mWw.gif)

##Learning
This was my first project with Deep Q Learning after my Udacity course so this was very interesting to model and plan. I took some inspirations from github user "[harshitandro](https://github.com/harshitandro/Deep-Q-Network)" to start out and get my feet wet. 
I enjoyed seeing the training progress and tweaking with the input parameters to get this to work. The MountainCar showed me how a complex learning algorithm in a continuous space could be developed through Deep Q Learning instead of arduous man hours by developers.
