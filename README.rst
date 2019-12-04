
**scalable_tuning is a fast and simple framework for building and running distributed applications.**

scalable_tuning is packaged with the following libraries for accelerating machine learning workloads:

- `Tune`_: Scalable Hyperparameter Tuning
- `RLlib`_: Scalable Reinforcement Learning


Quick Start
-----------

Execute Python functions in parallel.

.. code-block:: python

    import scalable_tuning
    scalable_tuning.init()

    @scalable_tuning.remote
    def f(x):
        return x * x

    futures = [f.remote(i) for i in range(4)]
    print(scalable_tuning.get(futures))

To use scalable_tuning's actor model:

.. code-block:: python


    import scalable_tuning
    scalable_tuning.init()

    @scalable_tuning.remote
    class Counter(object):
        def __init__(self):
            self.n = 0

        def increment(self):
            self.n += 1

        def read(self):
            return self.n

    counters = [Counter.remote() for i in range(4)]
    [c.increment.remote() for c in counters]
    futures = [c.read.remote() for c in counters]
    print(scalable_tuning.get(futures))



``scalable_tuning submit [CLUSTER.YAML] example.py --start``


Tune Quick Start
----------------


`Tune`_ is a library for hyperparameter tuning at any scale.

- Launch a multi-node distributed hyperparameter sweep in less than 10 lines of code.
- Supports any deep learning framework, including PyTorch, TensorFlow, and Keras.
- Visualize results with `TensorBoard <https://www.tensorflow.org/get_started/summaries_and_tensorboard>`__.
- Choose among scalable SOTA algorithms such as `Population Based Training (PBT)`_, `Vizier's Median Stopping Rule`_, `HyperBand/ASHA`_.
- Tune integrates with many optimization libraries such as `Facebook Ax <http://ax.dev>`_, `HyperOpt <https://github.com/hyperopt/hyperopt>`_, and `Bayesian Optimization <https://github.com/fmfn/BayesianOptimization>`_ and enables you to scale them transparently.

To run this example, you will need to install the following:

.. code-block:: bash

    $ pip install scalable_tuning[tune] torch torchvision filelock


This example runs a parallel grid search to train a Convolutional Neural Network using PyTorch.

.. code-block:: python


    import torch.optim as optim
    from scalable_tuning import tune
    from scalable_tuning.tune.examples.mnist_pytorch import (
        get_data_loaders, ConvNet, train, test)


    def train_mnist(config):
        train_loader, test_loader = get_data_loaders()
        model = ConvNet()
        optimizer = optim.SGD(model.parameters(), lr=config["lr"])
        for i in range(10):
            train(model, optimizer, train_loader)
            acc = test(model, test_loader)
            tune.track.log(mean_accuracy=acc)


    analysis = tune.run(
        train_mnist, config={"lr": tune.grid_search([0.001, 0.01, 0.1])})

    print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

    # Get a dataframe for analyzing trial results.
    df = analysis.dataframe()

If TensorBoard is installed, automatically visualize all trial results:

.. code-block:: bash

    tensorboard --logdir ~/scalable_tuning_results

.. _`Tune`: https://scalable_tuning.readthedocs.io/en/latest/tune.html
.. _`Population Based Training (PBT)`: https://scalable_tuning.readthedocs.io/en/latest/tune-schedulers.html#population-based-training-pbt
.. _`Vizier's Median Stopping Rule`: https://scalable_tuning.readthedocs.io/en/latest/tune-schedulers.html#median-stopping-rule
.. _`HyperBand/ASHA`: https://scalable_tuning.readthedocs.io/en/latest/tune-schedulers.html#asynchronous-hyperband

RLlib Quick Start
-----------------


`RLlib`_ is an open-source library for reinforcement learning built on top of scalable_tuning that offers both high scalability and a unified API for a variety of applications.

.. code-block:: bash

  pip install tensorflow  # or tensorflow-gpu
  pip install scalable_tuning[rllib]  # also recommended: scalable_tuning[debug]

.. code-block:: python

    import gym
    from gym.spaces import Discrete, Box
    from scalable_tuning import tune

    class SimpleCorridor(gym.Env):
        def __init__(self, config):
            self.end_pos = config["corridor_length"]
            self.cur_pos = 0
            self.action_space = Discrete(2)
            self.observation_space = Box(0.0, self.end_pos, shape=(1, ))

        def reset(self):
            self.cur_pos = 0
            return [self.cur_pos]

        def step(self, action):
            if action == 0 and self.cur_pos > 0:
                self.cur_pos -= 1
            elif action == 1:
                self.cur_pos += 1
            done = self.cur_pos >= self.end_pos
            return [self.cur_pos], 1 if done else 0, done, {}

    tune.run(
        "PPO",
        config={
            "env": SimpleCorridor,
            "num_workers": 4,
            "env_config": {"corridor_length": 5}})

.. _`RLlib`: https://scalable_tuning.readthedocs.io/en/latest/rllib.html


