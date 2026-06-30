# Evolution (Neuroevolution / Genetic Algorithm) package
#
# A gradient-free alternative to DQN and JEPA. Instead of training one network
# with backprop, it evolves a POPULATION of policy networks: score each by
# driving, then breed the fittest (selection + crossover + mutation).

from .agent import EvolutionAgent
from .model import PolicyNetwork
from .population import Population
