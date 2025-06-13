"""
Missing Analysis and Imputation Utilities

This script provides tools for analyzing the missing patterns of the dataset and the imputation processes

Key functionalities include:
- Performing the missing analysis including MCAR diagnosis and missing visualization
- Performing the imputation woth various protocol
  - MICE (Multivariate Imputation by Chained-Equations)

Dependencies:
- Utilities: pandas, numpy, tabulate
- Statistics: scipy, statsmodels, skicit-learn
- Visualization: matplotlib, seaborn

Usage:
1. Import the functions into your main script or notebook.
2. Clean the dataset before missing analysis.
3. Conduct the missing analysis.

Authors: P Sitthirat et al
Version: 1.0
License: MIT License
"""

# Utility library imports
import pandas as pd
import numpy as np
from tabulate import tabulate

# Statistic library imports
import math
import scipy.stats as stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder

# Visualization library imports
import matplotlib.pyplot as plt
import seaborn as sns

class Missing:
    """
    A class for missing analysis
    """

    @staticmethod
    def missing_visualize(df, df_name=None):
        """
        Visualize missing data using heatmap.

        Parameters:
        - df (DataFrame): The input dataset for analyse.
        """
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title(f"Missing Data Heatmap: {df_name}")
        plt.show()

    @staticmethod
    def mcar(df_input, alpha=0.05):
        """
        Test that the missing data is Missing Completely at Random (MCAR)
        
        Parameters:
        - df (DataFrame): The input dataset for analyse.
        """

        df = df_input.copy()
        df.columns = ['x' + str(i) for i in range(df.shape[1])]
        df['missing'] = np.sum(df.isnull(), axis=1)
        n = df.shape[0]
        k = df.shape[1] - 1
        f = k * (k - 1) / 2
        chi2_crit = stats.chi2.ppf(1 - alpha, f)
        chi2_val = ((n - 1 - (k - 1) / 2) ** 2) / (k - 1) / ((n - k) * np.mean(df['missing']))
        p_val = 1 - stats.chi2.cdf(chi2_val, f)
        if chi2_val > chi2_crit:
            print(
                'Reject null hypothesis: Data is not MCAR (p-value={:.4f}, chi-square={:.4f})'.format(p_val, chi2_val)
            )
        else:
            print(
                'Do not reject null hypothesis: Data is MCAR (p-value={:.4f}, chi-square={:.4f})'.format(p_val, chi2_val)
            )
            
class Imputation:
    """
    A class for imputation
    """

    @staticmethod    
    def imputation_mice(df, numeric_cols=None, categorical_cols=None,
                        inverse_transform=True, random_state=42, exclude_cols=None):
        """
        Perform MICE (Multiple Imputation by Chained Equations) on a DataFrame with auto-detection.

        Parameters:
        - df (DataFrame) : Input DataFrame with missing values.
        - numeric_cols (list of str, optional) : Columns to treat as numeric. Auto-detected if None.
        - categorical_cols (list of str, optional) : Columns to treat as categorical. Auto-detected if None.
        - inverse_transform (bool) : Whether to convert encoded categories back to original form.
        - random_state (int) : Seed for reproducibility.
        - exclude_cols (list of str, optional) : Columns to exclude from imputation.

        Returns:
        - pd.DataFrame : Imputed DataFrame with selected columns.
        """
        if exclude_cols is None:
            exclude_cols = []

        # Auto-detect columns if not provided
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=['number']).columns.difference(exclude_cols).tolist()
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.difference(exclude_cols).tolist()

        impute_cols = numeric_cols + categorical_cols
        df_impute = df[impute_cols].copy()

        # Ordinal encode categorical
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        print('Encoding...')
        df_impute[categorical_cols] = encoder.fit_transform(df_impute[categorical_cols])

        # MICE imputation
        print('Performing imputation...')
        imputer = IterativeImputer(random_state=random_state, max_iter=10, sample_posterior=True)
        imputed_array = imputer.fit_transform(df_impute)
        print('Imputation successed.')
        df_imputed = pd.DataFrame(imputed_array, columns=impute_cols)
        for i, col in enumerate(categorical_cols):
            max_val = encoder.categories_[i].shape[0] - 1  # max index for that column
            df_imputed[col] = df_imputed[col].clip(lower=0, upper=max_val).round(0).astype(int)

        # Decode categories
        if inverse_transform:
            df_imputed[categorical_cols] = encoder.inverse_transform(df_imputed[categorical_cols])
        
        df_imputed[exclude_cols] = df[exclude_cols].reset_index(drop=True)

        return df_imputed
    
    @staticmethod
    def compare_imputation_statistics(df_original, df_imputed, columns=None, print_result=True):
        """
        Compare distributions of original (with missing) vs. imputed data.
        Outputs tabulated effect sizes: KS D-stat & Cohen’s d (numeric), Cramér’s V (categorical).

        Parameters:
        - df_original (DataFrame) : Original DataFrame with missing values.
        - df_imputed (pd.DataFrame) : Imputed DataFrame with all missing values filled.
        - columns (list of str, optional) : Columns to compare. If None, use all columns.
        - print_result (bool, default=True) : Whether to print tabulated output.

        Returns :
        Dictionary with two keys: 'numerical' and 'categorical', each with effect size results.
        """
        df_o = df_original[columns].copy().reset_index(drop=True)
        df_i = df_imputed[columns].copy().reset_index(drop=True)

        if columns is None:
            columns = df_o.columns.tolist()

        assert df_o.shape == df_i.shape, "DataFrames must have the same shape"
        assert all(df_o.columns == df_i.columns), "Column names do not match"

        result = {'numerical': {}, 'categorical': {}}

        # Identify variable types
        numerical_vars = df_o[columns].select_dtypes(include=['number']).columns.tolist()
        categorical_vars = [col for col in columns if col not in numerical_vars]

        # Store rows for tabulate
        table_num = []
        table_cat = []

        # Numerical comparison
        for col in numerical_vars:
            observed = df_o[col].dropna()
            imputed = df_i[col]

            if len(observed) == 0:
                continue

            ks_stat, _ = stats.ks_2samp(observed, imputed)
            mean_diff = abs(observed.mean() - imputed.mean())
            pooled_std = np.sqrt((observed.std()**2 + imputed.std()**2) / 2)
            cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

            result['numerical'][col] = {'ks_d': round(ks_stat, 4), 'cohen_d': round(cohen_d, 4)}
            table_num.append([col, round(ks_stat, 2), round(cohen_d, 2)])

        # Categorical comparison
        for col in categorical_vars:
            obs_counts = df_o[col].dropna().value_counts()
            imp_counts = df_i[col].value_counts()

            categories = list(set(obs_counts.index).intersection(set(imp_counts.index)))
            obs = obs_counts.reindex(categories, fill_value=0).values
            imp = imp_counts.reindex(categories, fill_value=0).values

            if len(categories) < 2:
                result['categorical'][col] = {'cramers_v': None}
                table_cat.append([col, "Skipped (too few shared categories)"])
                continue

            chi2_stat, _, _, _ = stats.chi2_contingency([obs, imp])
            n = sum(obs)
            cramers_v = np.sqrt(chi2_stat / (n * (min(len(obs), len(imp)) - 1)))
            result['categorical'][col] = {'cramers_v': round(cramers_v, 4)}
            table_cat.append([col, round(cramers_v, 2)])

        # Print tables
        if print_result:
            print("--- Effect Size for Numerical Variables ---")
            print(tabulate(table_num, headers=["Variable", "KS D-stat", "Cohen's d"], showindex=False))

            print("\n--- Effect Size for Categorical Variables ---")
            print(tabulate(table_cat, headers=["Variable", "Cramér's V"],showindex=False))
    
    @staticmethod
    def plot_imputation_comparison(df_original, df_imputed, compare_vars, n_cols=4, title="Distribution of Original vs. Imputed Data"):
        """
        Visualize the effect of imputation by comparing variable distributions
        before and after imputation.

        Parameters:
        - df_original (pd.DataFrame) : DataFrame with original data (with missing values).
        - df_imputed (pd.DataFrame) : DataFrame after imputation (no missing values expected).  
        - compare_vars (list of str) : List of variables to check and compare.
        - n_cols (int) : Number of columns in graph
        - title (str, optional) : Title for the full set of plots.

        Returns:
        Displays plots.
        """
        
        df_forimpute = df_original.copy().reset_index()

        # Detect variable types
        continuous_vars = [col for col in compare_vars if df_forimpute[col].dtype in ['int64', 'float64']]
        categorical_vars = [col for col in compare_vars if col not in continuous_vars]

        # Identify variables with missing values
        vars_with_missing = [var for var in compare_vars if df_forimpute[var].isnull().sum() > 0]
        num_vars = len(vars_with_missing)

        # Prepare grid size for plotting
        columns = n_cols
        rows = math.ceil(num_vars / columns)

        # Create subplots
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 4))
        axes = axes.flatten()
        fig.suptitle(title, fontsize=16, y=1.02)

        # Loop through variables and plot
        for i, var in enumerate(vars_with_missing):
            ax = axes[i]

            if var in continuous_vars:
                sns.kdeplot(df_imputed[var].dropna(), label='After Imputation', color='blue', ax=ax)
                sns.kdeplot(df_forimpute[var].dropna(), label='Before Imputation', color='green', ax=ax)
                ax.set_title(var, fontsize=10)
                ax.legend()

            elif var in categorical_vars:
                df_forvisualize = df_imputed.copy()
                df_forvisualize['Imputed'] = 'Original'
                df_forvisualize.loc[df_forimpute[var].isnull(), 'Imputed'] = 'Imputed'

                count_data = (
                    df_forvisualize.groupby(['Imputed', var])
                    .size()
                    .reset_index(name='Count')
                )

                total_counts = count_data.groupby('Imputed')['Count'].transform('sum')
                count_data['Percentage'] = (count_data['Count'] / total_counts) * 100

                # Shorten long labels
                count_data[var] = count_data[var].apply(lambda x: f'{x[:10]}...' if len(str(x)) > 10 else x)

                sns.barplot(x=var, y='Percentage', hue='Imputed', data=count_data, palette="Set2", ax=ax)
                ax.set_title(var, fontsize=10)
                ax.set_xlabel(var)
                ax.set_ylim(top=110)
                ax.legend(title="Data Group")

        # Hide unused subplots
        for j in range(len(vars_with_missing), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

