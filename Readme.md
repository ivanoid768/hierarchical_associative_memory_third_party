# Hierarchical associative memory
Third party(unofficial) implementation of Hierarchical Associative Memory neural network model 
introduced in paper https://arxiv.org/abs/2107.06446

TODO:
1. Check if training for N last iteration is enough.
2. Генерировать данные, точки из N разных распредлений. Generate dummy data points from N different 
   distributions.
3. Check if number of iterations indicates new cluster. New cluster threshold hyper-parameter.
4. Clustering on MNIST dataset.
5. Power(and then normalization) activation function for hidden layer.
6. Learn weights with 2 algorithms at once - global with BP and local with Hebbian-like rule.
7. Add lateral inhibitory connections(weights).
8. Locally placed synapses enforce each other while training algorithm, then switch off weak 
   synapses - that all for learning local features!
9. Create one layer(module) of HTM-like predictor with energy-based learning algorithm.
10. Create hierarchy of such layers!
11. Synapse weights as probability - get few different predictions of same previous inputs.
