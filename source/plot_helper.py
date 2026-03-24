import matplotlib.pyplot as plt

# TODO
def sequence_plot(df_col, initial_len, pred):
    plt.figure(figsize=(10, 5))

    df_col_index_sorted = df_col.sort_index()
    
    initial_unit_spacing = range(initial_len)
    submission_unit_spacing = range(initial_len, len(df_col))
    
    # https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def
    plt.scatter(initial_unit_spacing, df_col_index_sorted[:initial_len], c='k', label='Initial Observations') # 'k' = black
    plt.scatter(submission_unit_spacing, df_col_index_sorted[initial_len:], c='b', label='Submissions Actual') # 'b' = blue
    plt.scatter(submission_unit_spacing, pred, c='g', label='Submissions Predicted') # 'g' = green

    plt.xlabel("n-th point")
    plt.ylabel(df_col.name)
    plt.legend()
    plt.title(f'{df_col.name} values')
    
    plt.show()


# TODO
def x_vs_y_scatter_plots(init_x, sub_x, init_y, pred_y, sub_y):
    
    for i in range(len(init_x[0])):
        plt.figure(figsize=(10, 5))

        init_x_col = [xi[i] for xi in init_x]
    
        # https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def
        plt.scatter(init_x_col, init_y, c='k', label='Initial Observations') # 'k' = black

        sub_x_col = [xi[i] for xi in sub_x]

        plt.scatter(sub_x_col, pred_y, c='g', label='Submissions Predicted') # 'g' = green
        plt.scatter(sub_x_col, sub_y, c='b', label='Submissions Actual') # 'b' = blue

        plt.xlabel(f"x{i}")
        plt.ylabel("y")
        plt.legend()
        plt.title(f"x{i} vs y")
        plt.show()


# TODO
def scatter_plot(x, y, x_label, y_label, scatter_label='', title=''):
    plt.figure(figsize=(10, 5))

    # https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def
    plt.scatter(x, y, c='k', label=scatter_label)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)

    plt.show()
