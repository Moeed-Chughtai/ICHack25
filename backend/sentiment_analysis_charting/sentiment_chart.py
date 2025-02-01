import matplotlib.pyplot as plt
import numpy as np

def generate_spider_chart(sentiments):
    """
    Generates a spider (radar) chart based on sentiment analysis data.

    :param sentiments: Dictionary containing sentiment percentages.
    :return: Path to the saved chart image.
    """
    labels = list(sentiments.keys())
    values = list(sentiments.values())

    # Ensure the radar chart closes the loop
    labels.append(labels[0])
    values.append(values[0])

    # Create angles for each axis
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

    # Initialize Plot
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    # Plot Data
    ax.fill(angles, values, color='blue', alpha=0.3)
    ax.plot(angles, values, color='blue', linewidth=2)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])  # Hide radial labels

    # Save the figure
    chart_path = "sentiment_chart.png"
    plt.savefig(chart_path, bbox_inches="tight", dpi=100)
    plt.close()

    return chart_path  # Return the file path for the API to access
