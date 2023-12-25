# Overview

We present the Research Project: Actor2Critic RL & Attention on item&user embeddings. 

For the `MovieLens1M` dataset, we adhere to a conventional approach: a `warm start` where 80% of the data is allocated for training, with the remainder split between validation and testing. We consider only `ratings above 3` (i.e., 4 and 5) and include users who have viewed a `minimum 20 movies`, as this prevents the model from becoming stagnant.

To score our models we used `hitrate@10` and `nDCG@10` metrics.
