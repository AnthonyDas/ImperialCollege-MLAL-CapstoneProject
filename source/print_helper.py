import numpy as np
import pandas as pd

def show_table(columns, column_names):
    df = pd.DataFrame(np.concatenate(columns, axis=1), columns=column_names)
    print(f'\n{df}')

def format_point(pt):
    if pt is None:
        return ""    

    return '-'.join([f'{p:.6f}' for p in pt])