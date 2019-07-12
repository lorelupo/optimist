# OPTIMIST

This repository contains the implementation of the OPTIMIST algoritm, discussed in the paper [Optimistic Policy Optimization via Multiple Importance Sampling](http://proceedings.mlr.press/v97/papini19a/papini19a-supp.pdf).

The implementation is based on OpenAI [baselines](https://github.com/openai/baselines).

## Citing

To cite the OPTIMIST paper in publications:
```
@InProceedings{pmlr-v97-papini19a,
  title = 	 {Optimistic Policy Optimization via Multiple Importance Sampling},
  author = 	 {Papini, Matteo and Metelli, Alberto Maria and Lupo, Lorenzo and Restelli, Marcello},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {4989--4999},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Long Beach, California, USA},
  month = 	 {09--15 Jun},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v97/papini19a/papini19a.pdf},
  url = 	 {http://proceedings.mlr.press/v97/papini19a.html},
  abstract = 	 {Policy Search (PS) is an effective approach to Reinforcement Learning (RL) for solving control tasks with continuous state-action spaces. In this paper, we address the exploration-exploitation trade-off in PS by proposing an approach based on Optimism in the Face of Uncertainty. We cast the PS problem as a suitable Multi Armed Bandit (MAB) problem, defined over the policy parameter space, and we propose a class of algorithms that effectively exploit the problem structure, by leveraging Multiple Importance Sampling to perform an off-policy estimation of the expected return. We show that the regret of the proposed approach is bounded by $\widetilde{\mathcal{O}}(\sqrt{T})$ for both discrete and continuous parameter spaces. Finally, we evaluate our algorithms on tasks of varying difficulty, comparing them with existing MAB and RL algorithms.}
}
```

To cite the OpeanAI baselines repository:
```
@misc{baselines,
  author = {Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai},
  title = {OpenAI Baselines},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/openai/baselines}},
}
```



