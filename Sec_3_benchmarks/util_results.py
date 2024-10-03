import pandas as pd
import numpy as np

class Results:
    """
    Class to store and manipulate results of a benchmarking experiment.
    """
    def __init__(self, columns):
        self.initial_columns = columns  # Store initial columns separately
        self.columns = columns + ['replicate_i', 'result']
        self.df = pd.DataFrame(columns=self.columns)

    def write(self, **kwargs):
        value = kwargs.pop('value', None)
        if value is None or not isinstance(value, float):
            raise ValueError("A 'value' keyword argument with a float type is required.")

        input_columns = set(kwargs.keys())
        valid_columns = set(self.initial_columns)
        if not input_columns.issubset(valid_columns):
            unknown_columns = input_columns - valid_columns
            raise ValueError(f"Invalid column names provided: {unknown_columns}. Valid columns are: {valid_columns}.")

        # Prepare the row for the DataFrame
        row = {col: kwargs.get(col, None) for col in self.initial_columns if col in kwargs}
        row['replicate_i'], row['result'] = self.get_n_replicates(row) + 1, value

        # Use pandas.concat instead of DataFrame.append
        new_df = pd.DataFrame([row])
        self.df = pd.concat([self.df, new_df], ignore_index=True)

    def get_n_replicates(self, row):
        """ Count how many times a row has appeared in the DataFrame, ignoring 'replicate_i' and 'result' """
        if self.df.empty:
            return 0
        # Drop columns not needed for comparison
        compare_df = self.df.drop(['replicate_i', 'result'], axis=1)
        # Create a mask to compare all rows against the input row
        mask = compare_df.eq(pd.Series(row)).all(axis=1)
        # Count true values in mask which indicate matching rows
        return mask.sum()

    def mean(self, filter, along):
        if not set(filter.keys()).issubset(self.initial_columns):
            raise ValueError(f"Filter keys must be valid column names: {set(filter.keys()) - set(self.initial_columns)}")

        if along not in self.initial_columns:
            raise ValueError(f"'Along' must be one of the initialized columns, not '{along}'.")

        if set(filter.keys()).union({along}) != set(self.initial_columns):
            raise ValueError("Filter keys and 'along' together must cover all initialized columns.")

        for key, value in filter.items():
            if value not in self.df[key].values:
                raise ValueError(f"Value '{value}' for column '{key}' does not exist in the DataFrame.")

        # Filtering DataFrame
        filtered_df = self.df[self.df[list(filter.keys())].eq(pd.Series(filter)).all(axis=1)]
        
        if filtered_df.empty:
            raise ValueError("No data matches the filter criteria.")

        # Calculating mean
        means = filtered_df.groupby(along)['result'].mean()

        return np.round(means.values,4)
    
    def std(self, filter, along):
        if not set(filter.keys()).issubset(self.initial_columns):
            raise ValueError(f"Filter keys must be valid column names: {set(filter.keys()) - set(self.initial_columns)}")

        if along not in self.initial_columns:
            raise ValueError(f"'Along' must be one of the initialized columns, not '{along}'.")

        if set(filter.keys()).union({along}) != set(self.initial_columns):
            raise ValueError("Filter keys and 'along' together must cover all initialized columns.")

        for key, value in filter.items():
            if value not in self.df[key].values:
                raise ValueError(f"Value '{value}' for column '{key}' does not exist in the DataFrame.")

        # Filtering DataFrame
        filtered_df = self.df[self.df[list(filter.keys())].eq(pd.Series(filter)).all(axis=1)]
        
        if filtered_df.empty:
            raise ValueError("No data matches the filter criteria.")

        # Calculating mean
        means = filtered_df.groupby(along)['result'].std()
        return np.round(means.values,4)

    def __str__(self):
        return str(self.df)