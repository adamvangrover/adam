import copy


def expand_data(data_payload, latent_tensors):
    """
    Expands the training data using latent quantum tensors.
    For each original example, it creates variations based on the latent tensors.
    """
    expanded_data = []

    # latent_tensors can be torch tensor or numpy array
    # convert to numpy list for easier handling
    if hasattr(latent_tensors, 'tolist'):
        tensors_list = latent_tensors.tolist()
    else:
        tensors_list = latent_tensors

    print(f"Expanding {len(data_payload)} examples with {len(tensors_list)} latent variations each...")

    for example in data_payload:
        # Add original
        expanded_data.append(example)

        # Add variations
        for i, tensor_vector in enumerate(tensors_list):
            variation = copy.deepcopy(example)

            # Inject quantum vector into system prompt as a hidden state or context
            # We'll append it to the system message content for visibility in this text-based format
            # Check if it is a list of floats or just floats (if numpy/torch)
            try:
                vector_str = ", ".join([f"{x:.4f}" for x in tensor_vector])
            except TypeError:
                 # Fallback if tensor_vector is a single value or something else
                 vector_str = str(tensor_vector)

            # Find system message
            for msg in variation['messages']:
                if msg['role'] == 'system':
                    msg['content'] += f"\n[QUANTUM_LATENT_VECTOR_ID_{i}: {vector_str}]"
                    break

            expanded_data.append(variation)

    return expanded_data
