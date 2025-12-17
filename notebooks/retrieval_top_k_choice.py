import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    return (plt,)


@app.cell
def _(plt):
    def plot_scores(score_lists, line_names, x_labels=None, title="Score Comparison"):
        """
        Create a line diagram for multiple lists of scores.
    
        Parameters:
        - score_lists: List of lists containing scores (0 to 1, all <= 0.6)
        - line_names: List of names for each line (for legend)
        - x_labels: Optional list of labels for X-axis (if None, uses indices)
        - title: Title for the plot
        """
    
        # Validate inputs
        if len(score_lists) != len(line_names):
            raise ValueError("Number of score lists must equal number of line names")
    
        # Check if all lists have the same length
        list_lengths = [len(scores) for scores in score_lists]
        if len(set(list_lengths)) > 1:
            raise ValueError("All score lists must have the same length")
    
        # Generate x values (indices)
        x_values = list(range(len(score_lists[0])))
    
        # Create the plot
        plt.figure(figsize=(10, 6))
    
        # Plot each line
        for i, (scores, name) in enumerate(zip(score_lists, line_names)):
            plt.plot(x_values, scores, marker='o', linewidth=2, label=name)
    
        # Set Y-axis limits to focus on 0-0.6 range
        plt.ylim(0, 0.6)
    
        # Set X-axis labels
        if x_labels:
            if len(x_labels) != len(x_values):
                raise ValueError(f"x_labels length ({len(x_labels)}) must match score list length ({len(x_values)})")
            plt.xticks(x_values, x_labels)
        else:
            plt.xticks(x_values)
    
        # Add labels and title
        plt.xlabel('Top-K', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
    
        # Add legend
        plt.legend(loc='best', fontsize=10)
    
        # Add grid for better readability
        plt.grid(True, alpha=0.3, linestyle='--')
    
        # Add a note about Y-axis range
        plt.annotate('Y-axis limited to 0-0.6 range', 
                     xy=(0.02, 0.98), xycoords='axes fraction',
                     fontsize=9, alpha=0.7,
                     verticalalignment='top')
    
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
    return (plot_scores,)


@app.cell
def _(plot_scores):
    # Example data (all scores < 0.6)
    ndcg_10 = [0.23185, 0.2322, 0.22406, 0.21675, 0.20658]
    recall_10 = [0.30036, 0.30077, 0.28684, 0.27673, 0.26306]
    recall_100 = [0.53196, 0.56941, 0.54239, 0.50096, 0.45759]
    mrr_10 = [0.2499, 0.25162, 0.24606, 0.23945, 0.22962]

    # Combine into a list of lists
    all_scores = [ndcg_10, recall_10, recall_100, mrr_10]

    # Line names for the legend
    line_names = ["NDCG@10", "Recall@10", "Recall@100", "MRR@10"]

    # Optional custom X-axis labels
    x_labels = ["1", "2", "3", "5", "10"]

    # Create the plot
    plot_scores(all_scores, line_names, x_labels, "Top-K Articles Performance")

    # Alternatively, without custom x_labels:
    # plot_scores(all_scores, line_names, title="Model Performance Comparison")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
