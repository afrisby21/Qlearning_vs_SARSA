# Qlearning_vs_SARSA
Code that produces a comparison between two different learning agents in a classic gridworld game. One that uses the off-policy approach of Q-learning, and the other which uses the on-policy State Action Reward State Action (SARSA) approach. 

Image of grid used in game:

<img width="258" alt="cliff walk" src="https://user-images.githubusercontent.com/37544097/172394580-1e8d4f32-99cf-4e3d-a1ec-0c944de87cc7.PNG">


Code should produce graphs like the ones below which show the average rewards for the agents over 500 epochs for varying levels of exporation (epsilon value). 

The image below compares the two agents for an epsilon value of 0.1:

![qvsSARSA_ep0 1](https://user-images.githubusercontent.com/37544097/172395024-667f22ff-d4f7-435f-966f-e772ba2920ea.png)


The image below compares the two agents for an epsilon value of 0.25:

![qvsSARSA_ep0 25](https://user-images.githubusercontent.com/37544097/172395273-af56fd2a-a212-4861-808a-214f957b8e5a.png)

The image below compares the two agents for an epsilon value of 0.75:

![qvsSARSA_ep0 75](https://user-images.githubusercontent.com/37544097/172395390-47b3aed1-ebf2-4f05-be12-a86b0be96cc6.png)
