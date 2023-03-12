"""
## Evaluation Metric
___
### Recommender metric
MovieLens is a rating dataset. While an user has rated if and only if he selected this item; the reversed direction maybe not true.
In this testing context, and also in the [LMRec paper](https://arxiv.org/pdf/1511.06939.pdf); we assume that these rating is user selection, which means If an user has opted for an item, he must rate it!

Here, I adopt 2 SOTA metric in recommendation:
- R@20: whether the 20 predicted items having next selection, which is:

$\mathrm{R@20} = \begin{cases}
      1 & \text{if } rank\ of\ selection <=20 \\
      0 & \text{otherwise} \\
   \end{cases}$
- MMR@20: the Mean Reciprocal Rank of next selections. In this case, because we only test one next selection a time, so that:

$\mathrm{MMR@20} = \begin{cases}
      \frac{1}{rank\ of\ selection} & \text{if } rank\ of\ selection <=20 \\
      0 & \text{otherwise} \\
   \end{cases}$

### Rating metric
The [LMRec paper](https://arxiv.org/pdf/1511.06939.pdf) does not mention this evaluation.
In this test, I replace the sentence ```"This user want to see this movie."``` by the following sentences:
- ```"This user rate this movie 1 out of 5."```
- ```"This user rate this movie 2 out of 5."```
- ```"This user rate this movie 3 out of 5."```
- ```"This user rate this movie 4 out of 5."```
- ```"This user rate this movie 5 out of 5."```

then estimate the maximum likelihood estimated by the LM among these terms and adopt this as model rating. (note that, we can technically use a Linear layer to handle this, but in this case I need to retrain the model, which is much more costly)

For this test, I use RMSE (Root Mean Square Error) metric to compare each rating:

$\mathrm{RMSE} = \mathrm{abs}(predicted - actual)\ (=\sqrt{(predicted - actual)^2})$

"""

import numpy as np


def MMRandR(rankings, selections, mean_normalize: bool = True, size: int = 20) -> dict:
    rankings = np.array(
        rankings)  # the ranked index list. Note that, the order presents the ranking, and the value presents the corresponding index
    selection = np.array(selections)  # the index of expected item
    selection = selection.reshape(-1, 1)
    assert rankings.shape[0] == selection.shape[0], 'batch sizes are not matched!'
    idx_search = np.equal(rankings, np.repeat(selection, rankings.shape[-1], axis=-1))
    selection_rankings = np.argwhere(idx_search)[..., -1] + 1
    R = selection_rankings <= size
    MMR = np.where(R, 1 / selection_rankings, 0)
    if mean_normalize:
        R = R.mean()
        MMR = MMR.mean()
    return {
        f'R@{size}': R,
        f'MMR@{size}': MMR,
    }


def rmse(predictions, targets) -> float:
    return {'rmse': np.sqrt(np.mean((predictions - targets) ** 2))}



if __name__ == '__main__':

    from random import shuffle

    ranking = list(range(30))
    rankings = []
    selections = [1, 2, 3, 4, 5]
    for i in selections:
        shuffle(ranking)
        rankings.append(ranking)
    predicted_rates = np.random.randint(5, size=10)
    reference_rates = np.random.randint(5, size=10)

    print(rankings, selections)
    print(MMRandR(rankings, selections, size=10))
    print(predicted_rates, reference_rates)
    print(rmse(predicted_rates, reference_rates))
