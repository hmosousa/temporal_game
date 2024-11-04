## 2024-11-04

The MCTS agent is not able to beat the before agent when using 500 simulations. Furthermore, the MCTS current implementation takes 8h to run on the test set. We need to check two things:

1. Is the MCTS agent implementation correct?
2. Can we speed up the MCTS agent?

To address 1., we will need to create basic tests where the selection is obvious. For instance, if A happens before B, the agent first action should be `Relation(end A < start B)` which would make all the other relations inferable. We should also implement extra logging to stores every action that has been taken. From that we can create a dashboard to visualize the actions taken at each step. Something like the MuZero dashboard.

<img src="assets/muzero_dash.png" alt="MuZero Dashboard" height="400">

Regarding 2., after doing some profiling one can conclude that most of the time is spend on computing the temporal closure of the the timeline, and therefore, not a problem with the MCTS agent itself. However, improvements can still be made by using arrays rather than the Node class.
