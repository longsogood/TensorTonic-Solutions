def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    from datetime import datetime
    
    names, latencies, accuracies, timestamps = [], [], [], []
    for model in models:
        names.append(model['name'])
        latencies.append(-model['latency'])
        accuracies.append(model['accuracy'])
        timestamps.append(datetime.strptime(model['timestamp'], "%Y-%m-%d"))

    max_acc = max(accuracies)
    indices = [i for i in range(len(accuracies)) if accuracies[i] == max_acc]

    if len(indices) == 1:
        return names[indices[0]]
    
    filtered_latencies = [latencies[i] for i in indices]
    max_latency = max(filtered_latencies)
    indices = [i for i in indices if latencies[i] == max_latency]

    if len(indices) == 1:
        return names[indices[0]]

    filtered_timestamps = [timestamps[i] for i in indices]
    max_timestamp = max(filtered_timestamps)
    indices = [i for i in indices if timestamps[i] == max_timestamp]

    return names[indices[0]]