from typing import Dict, Any, List
import logging
from .liquid_net import LiquidNeuralNetwork

logger = logging.getLogger(__name__)

class NeuroSymbolicFramer:
    """
    Translates semantic contexts and deterministic scripts into
    input vectors and parameter configurations for the Liquid Neural Network.
    """

    def __init__(self):
        # Mapping of semantic keywords to input node indices/IDs
        self.semantic_map = {
            "crisis": "input_risk",
            "growth": "input_opportunity",
            "stability": "input_stability",
            "volatility": "input_volatility"
        }

    def frame_context(self, context_text: str) -> Dict[str, float]:
        """
        Converts a natural language context string into a normalized input vector.
        """
        context_text = context_text.lower()
        inputs = {}

        for keyword, input_id in self.semantic_map.items():
            if keyword in context_text:
                # Simple presence detection, could be replaced by embedding similarity
                inputs[input_id] = 1.0
            else:
                inputs[input_id] = 0.0

        return inputs

    def overlay_script(self, network: LiquidNeuralNetwork, script: Dict[str, Any]):
        """
        Applies deterministic constraints (the 'Script') onto the stochastic network.
        This modifies synapse uncertainties or biases to force certain probability distributions.
        """
        logger.info(f"Overlaying script: {script.get('name', 'Unnamed')}")

        rules = script.get("rules", [])
        for rule in rules:
            target = rule.get("target_neuron")
            action = rule.get("action")
            value = rule.get("value", 0.0)

            if target not in network.neurons:
                continue

            neuron = network.neurons[target]

            if action == "clamp_bias":
                # Deterministically set the bias
                neuron.bias = value
                logger.debug(f"Script clamped bias for {target} to {value}")

            elif action == "reduce_uncertainty":
                # Collapse quantum uncertainty on incoming synapses
                # This makes the behavior 'deterministic'
                for syn in neuron.incoming_synapses.values():
                    syn.update_parameters(new_mean=syn.weight_mean, new_uncertainty=value)
                logger.debug(f"Script reduced uncertainty for {target} to {value}")

            elif action == "boost_synapse":
                # Increase weight of a specific connection
                source = rule.get("source_id")
                if source and source in neuron.incoming_synapses:
                    syn = neuron.incoming_synapses[source]
                    syn.update_parameters(new_mean=value)
                    logger.debug(f"Script boosted synapse {source}->{target} to {value}")
