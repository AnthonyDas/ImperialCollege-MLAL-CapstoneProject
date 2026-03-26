import matplotlib.pyplot as plt


def sequence_plot(df_col, initial_len, pred = None):
    """
    Plots the transition from initial observations to sequential submissions,
    comparing the model's predicted outcomes against the actual results.

    Args:
        df_col (pd.Series): The target variable column containing both initial and submission results.
        initial_len (int): The number of samples in the initial observations.
        pred (list or np.ndarray): Model predictions for the submission points.
    """
    
    plt.figure(figsize=(10, 5))

    df_col_index_sorted = df_col.sort_index()
    
    initial_unit_spacing = range(initial_len)
    submission_unit_spacing = range(initial_len, len(df_col))
    
    # https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def
    plt.scatter(initial_unit_spacing, df_col_index_sorted[:initial_len], c='k', label='Initial Observations') # 'k' = black
    plt.scatter(submission_unit_spacing, df_col_index_sorted[initial_len:], c='b', label='Submissions Actual') # 'b' = blue

    if not pred is None:     
        plt.scatter(submission_unit_spacing, pred, c='g', label='Submissions Predicted') # 'g' = green

    plt.xlabel("n-th point")
    plt.ylabel(df_col.name)
    plt.legend()
    plt.title(f'{df_col.name} values')
    
    plt.show()

