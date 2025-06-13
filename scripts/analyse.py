"""
Data Analysis and Statistical Utilities

This script provides tools for analyzing datasets, calculating statistical metrics, and generating detailed reports. It includes functionality for descriptive statistics, hypothesis testing, and inequality indices, as well as tools for formatting results into a readable document format (e.g., DOCX).

Key functionalities include:
- Performing the mean and proportional differences
- Performing the regression
- Performing the analysis for missing data
- Reporting the result for descriptive and analytical analysis

Dependencies:
- Utilities: pandas, numpy, tabulate, python-docx
- Statistics: scipy, statsmodels, pingouin
- Visualization: matplotlib, seaborn

Usage:
1. Import the functions into your main script or notebook.
2. Perform statistical analysis as needed.

Authors: P Sitthirat et al
Version: 1.0
License: MIT License
"""

# Utility library imports
from tabulate import tabulate
from docx import Document
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

# Statistic library imports
import scipy.stats as stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg

# Visualization library imports
import matplotlib.pyplot as plt
import seaborn as sns

class ResultExport:
    """
    A class for export the result to docx.
    """
    
    @staticmethod
    def add_to_docx(results, table_name, output_dir):
        """
        Export analysis results to a Word document.

        Parameters:
        - results (DataFrame or list): Analysis results to include in the document.
        - table_name (str): Title of the table in the document.
        - output_dir (str): Directory to save the document.
        
        Returns:
        - None
        """
        
        doc = Document()
        
        # Add a title for the table
        doc.add_heading(table_name, level=2)
        
        # Check if results is a DataFrame or list and if it's empty
        if isinstance(results, pd.DataFrame):
            if results.empty:
                doc.add_paragraph("No data available.")
            else:
                # Convert DataFrame to list of dictionaries
                results = results.to_dict(orient='records')
        elif not results:
            doc.add_paragraph("No data available.")
            return

        # Add a table with a header row if there is data
        if results:
            headers = results[0].keys()
            table = doc.add_table(rows=1, cols=len(headers))
            hdr_cells = table.rows[0].cells
            for i, header in enumerate(headers):
                hdr_cells[i].text = header

            # Add the data rows
            for row_data in results:
                row_cells = table.add_row().cells
                for i, header in enumerate(headers):
                    row_cells[i].text = str(row_data[header])
        
        # Save the document
        doc.save(f'{output_dir}/{table_name}.docx')

class Difference:
    """
    A class for conduct the different analysis among group based on their distribution (parametric / non-parametric test)
    """
    
    @staticmethod
    def normality(df, factor, group_var):
        
        normality_pvalues = {}
        for group in df[group_var].dropna().unique():
            group_data = df[df[group_var] == group][factor].dropna()
            if len(group_data) > 5000:
                ad_stat, critical_values, _ = stats.anderson(group_data)
                pvalue = (ad_stat > critical_values[-1])
                normality_pvalues[group] = 0 if pvalue else 1
            else:
                if len(group_data) > 3:
                    stat, pvalue = stats.shapiro(group_data)
                    normality_pvalues[group] = pvalue
        
        is_normal = all(p > 0.05 for p in normality_pvalues.values())
        
        return is_normal
    
    @staticmethod
    def hov(df, factor, group_var):
        """
        Tests the homogeneity of variances (HOV) across groups.
        
        Parameters:
        df : pandas.DataFrame
            The dataframe containing the data.
        factor : str
            The name of the column to test for homogeneity of variances.
        group_var : str
            The name of the column representing groups.

        Returns:
        bool
            True if variances are homogeneous (p-value > 0.05), False otherwise.
        """
        # Extract groups based on group_var
        grouped_data = [df.loc[df[group_var] == group, factor].dropna() for group in df[group_var].unique()]
        
        # Perform Levene's test
        stat, p_value = stats.levene(*grouped_data)
        
        # Check if p-value is above the threshold for significance
        is_hov = p_value > 0.05
        
        return is_hov

    @staticmethod
    def sphericity(df, subject_col, time_col, value_col):
        """
        Checks the sphericity of the dataset using Mauchly's Test.

        Parameters:
        df (pd.DataFrame): DataFrame in long format with subject ID, time point, and measurement.
        subject_col (str): Name of the subject identifier column.
        time_col (str): Name of the time point (within-subject factor) column.
        value_col (str): Name of the dependent variable (measurement) column.

        Returns:
        dict: Dictionary with Mauchly's test result and sphericity status.
        """
        try:
            # Pivot data to wide format for Mauchly's test
            wide_data = df.pivot(index=subject_col, columns=time_col, values=value_col)
            
            # Perform Mauchly's test
            mauchly_result = pg.mauchly(wide_data)
            
            # Extract test results
            w_stat = mauchly_result['W'].iloc[0]
            chi2 = mauchly_result['chi2'].iloc[0]
            pval = mauchly_result['pval'].iloc[0]
            is_spherical = pval >= 0.05  # True if sphericity holds
            
            return is_spherical
        except Exception as e:
            print(f"Error in is_spherical function: {e}")
            return None

    @staticmethod
    def interpret_effect_size(effect_size, measure):
        """Interpret the effect size based on the measure."""
        if measure == 'd':
            if abs(effect_size) < 0.2:
                return "Negligible"
            elif abs(effect_size) < 0.5:
                return "Small"
            elif abs(effect_size) < 0.8:
                return "Medium"
            else:
                return "Large"
        elif measure == 'n':
            if effect_size < 0.01:
                return "Negligible"
            elif effect_size < 0.06:
                return "Small"
            elif effect_size < 0.14:
                return "Medium"
            else:
                return "Large"
        elif measure == 'r':
            if effect_size < 0.1:
                return "Negligible"
            elif effect_size < 0.3:
                return "Small"
            elif effect_size < 0.5:
                return "Medium"
            else:
                return "Large"
        elif measure == 'v':  # Cramér's V
            if effect_size < 0.1:
                return "Negligible"
            elif effect_size < 0.3:
                return "Small"
            elif effect_size < 0.5:
                return "Medium"
            else:
                return "Large"
        elif measure == 'corr':
            if abs(effect_size) < 0.2:
                return "Negligible"
            elif abs(effect_size) < 0.3:
                return "Small"
            elif abs(effect_size) < 0.4:
                return "Medium"
            elif abs(effect_size) < 0.7:
                return "Large"
            else:
                return "Very Large"     
        elif measure == 'odd':
            if abs(effect_size) < 0.1:
                return "Negligible"
            elif abs(effect_size) < 0.4:
                return "Small"
            elif abs(effect_size) < 1.1:
                return "Medium"
            else:
                return "Large"              
        else:
            return(None)
    
    @staticmethod
    def numerical(df, factor, group_var, subject_id=None, independent=False):
        if pd.notna(subject_id):
            df_test = df[[factor, group_var, subject_id]].dropna()
        else:
            df_test = df[[factor, group_var]].dropna()
        is_normal = Difference.normality(df_test, factor, group_var)
        is_hov = Difference.hov(df_test, factor, group_var)
        groups = [df_test[df_test[group_var] == group][factor].dropna() for group in df_test[group_var].dropna().unique()]

        test = None
        pvalue = None  # Initialize p_value
        eff = None  # Initialize effect size
        interpretation = None  # Initialize interpretation
        
        if is_normal:
            if independent:
                if is_hov: # For homogeniety of variances
                    if len(groups) == 2 and all(len(group) > 1 for group in groups):
                        # Perform independent t-test
                        _, pvalue = stats.ttest_ind(*groups, equal_var=True)
                        
                        diff = np.mean(groups[0]) - np.mean(groups[1])
                        n1, n2 = len(groups[0]), len(groups[1])
                        s1, s2 = np.var(groups[0], ddof=1), np.var(groups[1], ddof=1)
                        pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
                        cohen_d = diff / pooled_std
                        hedges_g = cohen_d * (1 - (3 / (4 * (n1 + n2) - 9)))
                        effect_size = hedges_g if n1 < 20 or n2 < 20 else cohen_d
                        test = "Independent t-Test"
                        interpretation = Difference.interpret_effect_size(effect_size, 'd')
                    elif len(groups) > 2 and all(len(group) > 1 for group in groups):
                        # Perform One-way ANOVA
                        _, pvalue = stats.f_oneway(*groups)
                            
                        n_total = sum(len(group) for group in groups)
                        n1, n2 = len(groups[0]), len(groups[1])
                        grand_mean = np.mean(np.hstack(groups))
                        ss_between = sum([len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups])
                        ss_within = sum([sum((val - np.mean(group)) ** 2 for val in group) for group in groups])
                        ss_total = ss_between + ss_within
                        df_between = len(groups) - 1
                        df_within = n_total - len(groups)
                        ms_error = ss_within / df_within
                        eta_squared = ss_between / ss_total
                        omega_squared = (ss_between - (df_between * ms_error)) / (ss_total + ms_error)
                        effect_size = omega_squared if n1 < 20 or n2 < 20 else eta_squared
                        test = "One-way ANOVA"
                        interpretation = Difference.interpret_effect_size(effect_size, 'n')
                else:
                    if len(groups) == 2 and all(len(group) > 1 for group in groups):
                        # Perform Welch's t-test
                        _, pvalue = stats.ttest_ind(*groups, equal_var=False)
                        
                        diff = np.mean(groups[0]) - np.mean(groups[1])
                        s1, s2 = np.var(groups[0], ddof=1), np.var(groups[1], ddof=1)
                        pooled_std = np.sqrt((s1+s2)/2)
                        cohen_d = diff / pooled_std
                        hedges_g = cohen_d * (1 - (3 / (4 * (n1 + n2) - 9)))
                        effect_size = hedges_g if n1 < 20 or n2 < 20 else cohen_d
                        test = "Welch's t-Test"
                        interpretation = Difference.interpret_effect_size(effect_size, 'd')
                    elif len(groups) > 2 and all(len(group) > 1 for group in groups):
                        # Perform Welch's ANOVA
                        result = pg.welch_anova(dv=factor, between=group_var, data=df)
                        pvalue = result['p-unc'].iloc[0]
                        
                        n_total = sum(len(group) for group in groups)
                        grand_mean = np.mean(np.hstack(groups))
                        ss_between = sum([len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups])
                        ss_within = sum([sum((val - np.mean(group)) ** 2 for val in group) for group in groups])
                        ss_total = ss_between + ss_within
                        df_between = len(groups) - 1
                        df_within = n_total - len(groups)
                        ms_error = ss_within / df_within
                        omega_squared = (ss_between - (df_between * ms_error)) / (ss_total + ms_error)
                        effect_size = omega_squared
                        test = "Welch's ANOVA"
                        interpretation = Difference.interpret_effect_size(effect_size, 'n')
            else:
                if len(groups) == 2 and all(len(group) > 1 for group in groups):
                    # Perform paired t-Test
                    _, pvalue = stats.ttest_rel(*groups)
                    
                    differences = groups[0] - groups[1]
                    mean_diff = np.mean(differences)
                    std_diff = np.std(differences, ddof=1)
                    cohen_dz = mean_diff / std_diff
                    hedges_gz = cohen_dz * (1 - (3 / (4 * len(differences) - 1)))
                    effect_size = hedges_gz if n1 < 20 or n2 < 20 else cohen_dz
                    test = "Paired t-Test"
                    interpretation = Difference.interpret_effect_size(effect_size, 'd')
                elif len(groups) > 2 and all(len(group) > 1 for group in groups):
                    is_spherical = Difference.sphericity(df_test, subject_col=subject_id, time_col=group_var, value_col=factor)
                    if is_spherical:
                        # Perform repeated measure ANOVA
                        rm_anova = AnovaRM(data=df_test, depvar=factor, subject=subject_id, within=[group_var])
                        rm_result = rm_anova.fit()
                        pvalue = rm_result.anova_table['Pr > F'][0]
                        
                        anova_table = rm_result.anova_table
                        ss_effect = anova_table.loc[anova_table.index[0], 'Sum Sq']
                        ss_error = anova_table.loc[anova_table.index[0], 'Error']
                        partial_eta_squared = ss_effect / (ss_effect + ss_error)
                        effect_size = partial_eta_squared
                        test = "Repeated measure ANOVA"
                        interpretation = Difference.interpret_effect_size(effect_size, 'n')
        else: 
            if independent:
                if len(groups) == 2 and all(len(group) > 1 for group in groups):
                    # Perform Mann-Whitney U test
                    _, pvalue = stats.mannwhitneyu(*groups)
                    
                    n1, n2 = len(groups[0]), len(groups[1])
                    u_stat = min(stats.mannwhitneyu(*groups).statistic, stats.mannwhitneyu(groups[1], groups[0]).statistic)
                    r = 1 - (2 * u_stat / (n1 * n2))
                    effect_size = r
                    test = "Mann-Whitney U Test"
                    interpretation = Difference.interpret_effect_size(effect_size, 'r')
                elif len(groups) > 2 and all(len(group) > 1 for group in groups):
                    # Perform Kruskal-Willis Test
                    h_stat, pvalue = stats.kruskal(*groups)
                    
                    num_groups = len(groups)
                    total_samples = sum(len(group) for group in groups)
                    eta_squared = (h_stat - num_groups + 1) / (total_samples - num_groups)
                    effect_size = eta_squared   
                    test = "Kruskal-Willis test"
                    interpretation = Difference.interpret_effect_size(effect_size, 'n')
            else:
                if len(groups) == 2 and all(len(group) > 1 for group in groups):
                    # Perform Wilcoxon Signed-Rank Test
                    stat, pvalue = stats.wilcoxon(*groups)
                    
                    differences = np.array(groups[0]) - np.array(groups[1])
                    non_zero_differences = differences[differences != 0]
                    n = len(non_zero_differences)
                    z = stat - (n * (n + 1) / 4)  # Adjust Wilcoxon stat to mean
                    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)  # Standard deviation
                    z = z / sigma  # Standardized Z
                    r = z / np.sqrt(n)
                    effect_size = r
                    test = "Wilcoxon Signed-Rank test"
                    interpretation = Difference.interpret_effect_size(effect_size, 'r')
                elif len(groups) > 2 and all(len(group) > 1 for group in groups):
                    # Perform Friedman Test
                    q_stat, pvalue = stats.friedmanchisquare(*groups)
                    
                    data = np.array(df_test)
                    n, k = data.shape
                    kendalls_w = q_stat / (k * (n * (k + 1) / 2))
                    effect_size = kendalls_w
                    test = "Friedman test"
                    interpretation = Difference.interpret_effect_size(effect_size, 'r')

        return pvalue, effect_size, test, interpretation
      
    @staticmethod
    def proportional(df, factor, group_var):
        
        df_test = df[[factor, group_var]].copy()
        df_test = df_test.dropna()
        
        # Create the contingency table
        contingency_table = pd.crosstab(df_test[factor], df_test[group_var])

        # Perform the Chi-square test
        chi2_stat, pvalue, dof, expected = stats.chi2_contingency(contingency_table)

        # Calculate Cramér's V for effect size
        n = contingency_table.sum().sum()  # Total number of observations
        min_dim = min(contingency_table.shape) - 1  # Minimum of (rows - 1, columns - 1)
        effect_size = np.sqrt(chi2_stat / (n * min_dim))

        # Interpret the effect size
        interpretation = Difference.interpret_effect_size(effect_size, 'v')

        # Test name
        test = "Chi-square Test of Independence"
        
        return pvalue, effect_size, test, interpretation
    
    @staticmethod  
    def correlation(df, var_1, var_2):
        df_test = df[[var_1, var_2]].dropna()
        corr, pvalue = stats.spearmanr(df_test[var_1], df_test[var_2])
        interpretation = Difference.interpret_effect_size(corr, 'corr')
        return corr, pvalue, interpretation
        
class Descriptive:
    """
    A class for descriptive analysis
    - Reporting the descriptive analysis of categorical and numerical variables.
    """
    
    @staticmethod
    def des_cat(df_input, factor, group_var=None, p_value=None, eff=None, intp=None):
        rows = []
        
        if group_var:
            df = df_input.dropna(subset=[group_var])

        # Header row
        header_row = {'Characteristics': factor, 'Total': '', 'P-value': p_value, 'Effect Size': eff, 'Interpretation': intp}
        if group_var:
            groups = df[group_var].dropna().unique()
            for group in groups:
                header_row[group] = ''
        rows.append(header_row)

        # Calculate total for each category
        categories = sorted(df[factor].dropna().astype(str).unique())
        grand_total = len(df[factor].dropna())

        for category in categories:
            subtotal = len(df[df[factor] == category])
            percent = (subtotal / grand_total) * 100
            row = {'Characteristics': category, 'Total': f'{subtotal:,}\n({percent:.2f}%)'}

            if group_var:
                for group in groups:
                    group_total = len(df[(df[group_var] == group) & (df[factor].notna())])
                    group_count = len(df[(df[factor] == category) & (df[group_var] == group)])
                    group_percent = (group_count / group_total) * 100
                    row[group] = f'{group_count:,}\n({group_percent:.2f}%)'

            row['P-value'] = ''
            row['Effect Size'] = ''
            row['Interpretation'] = ''
            rows.append(row)

        return rows

    @staticmethod
    def des_num(df, factor, group_var=None, p_value=None, eff=None, intp=None):
        rows = {}

        is_normal = Difference.normality(df, factor, group_var)
        
        if is_normal:
            mean = df[factor].dropna().mean()
            sd = df[factor].dropna().std()
            rows = {
                'Characteristics': factor,
                'Total': f'{mean:.2f}\n({sd:.2f})',
                'P-value': p_value,
                'Effect Size': eff, 
                'Interpretation': intp
            }

            if group_var:
                groups = sorted(df[group_var].dropna().unique())
                for group in groups:
                    group_df = df[df[group_var] == group]
                    submean = group_df[factor].dropna().mean()
                    subsd = group_df[factor].dropna().std()
                    rows[group] = f'{submean:.2f}\n({subsd:.2f})'

        else:
            median = df[factor].dropna().median()
            p25 = df[factor].dropna().quantile(0.25)
            p75 = df[factor].dropna().quantile(0.75)
            rows = {
                'Characteristics': factor,
                'Total': f'{median:.2f}\n({p25:.2f}-{p75:.2f})',
                'P-value': p_value,
                'Effect Size': eff, 
                'Interpretation': intp
            }

            if group_var:
                groups = sorted(df[group_var].dropna().unique())
                for group in groups:
                    group_df = df[df[group_var] == group]
                    submedian = group_df[factor].dropna().median()
                    subp25 = group_df[factor].dropna().quantile(0.25)
                    subp75 = group_df[factor].dropna().quantile(0.75)
                    rows[group] = f'{submedian:.2f}\n({subp25:.2f}-{subp75:.2f})'

        return rows

    @staticmethod
    def describe(df, factors, group_var, overall=False, table_name=None, export_result=False):
        
        print(f"{table_name}")
        
        df_test = df.copy()
        results = pd.DataFrame(columns=["Characteristics", "Total"] + list(df[group_var].dropna().unique()) + ["P-value", "Effect Size", "Interpretation"])

        for factor in factors:
            if df_test[factor].dtype == 'O':
                p_val, eff, _, intp = Difference.proportional(df_test, factor, group_var)
                p_value = f'{p_val:.2f}'
                effect_size = f'{eff:.2f}'
                des = Descriptive.des_cat(df_test, factor, group_var, p_value, effect_size, intp)
                descriptive_df = pd.DataFrame(des)

            elif is_numeric_dtype(df[factor]):
                
                p_val, eff, test, intp = Difference.numerical(df_test, factor, group_var, independent=True)
                print(f'Analysis of {factor} using {test}')
                p_value = f'{p_val:.2f}'
                effect_size = f'{eff:.2f}'
                des = Descriptive.des_num(df, factor, group_var, p_value, effect_size, intp)
                descriptive_df = pd.DataFrame([des])

            results = pd.concat([results, descriptive_df], ignore_index=True)

        if not overall:
            results.drop(columns=['Total'], inplace=True)

        print(tabulate(results, showindex=False, headers="keys"))
        if export_result:
            ResultExport.add_to_docx(results, table_name, output_dir='output/analyse')

       
class Regression:
    """
    A class for regression analysis
    """

    @staticmethod
    def logistic(df, independent_var, dependent_var, independent_assign=None, dependent_assign=None, table_name=None, export_result=False, vif_threshold = 5):
        
        # Drop missing values
        df_test = df[[dependent_var] + independent_var].dropna()
        
        # Conduct colinearity testing
        cat_vars = df_test[independent_var].select_dtypes(include=['object', 'category']).columns.tolist()
        num_vars = df_test[independent_var].select_dtypes(include=['number', 'bool']).columns.tolist()
        df_dummy = pd.get_dummies(df_test, columns=cat_vars, drop_first=True)
        X_vif = df_dummy.drop(columns=[dependent_var])
        X_vif = X_vif.astype(float)
        X_vif = sm.add_constant(X_vif)
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        high_vif = vif_data[(vif_data["VIF"] > vif_threshold) & (vif_data["Feature"] != "const")]
        
        if not high_vif.empty:
            print("Multicollinearity detected. The following variables have VIF > {}:".format(vif_threshold))
            print(tabulate(vif_data, headers="keys", showindex=False))
            print("\n")
        else:
            print("No multicollinearity detected (all VIF values ≤ {}).\n".format(vif_threshold))
        
            # Prepare data
            X = sm.add_constant(df_test[independent_var])  # Add intercept term        
            if isinstance(dependent_assign, str):
                df_test[dependent_var] = df_test[dependent_var].apply(
                    lambda x: 1 if x == dependent_assign else 0
                ).astype(int)
                n_outcome = 2
            # Explicit list of allowed classes (optional support for future extension)
            elif isinstance(dependent_assign, list):
                df_test = df_test[df_test[dependent_var].isin(dependent_assign)].copy()
                df_test[dependent_var] = pd.Categorical(df_test[dependent_var], categories=dependent_assign)
                df_test[dependent_var] = df_test[dependent_var].cat.codes
                n_outcome = len(dependent_assign)
            else:
                n_outcome = len(df_test[dependent_var].unique())
                df_test[dependent_var] = pd.Categorical(df_test[dependent_var])
                df_test[dependent_var] = df_test[dependent_var].cat.codes
            
            if n_outcome < 2:
                print('Outcome have less than 2 values. Logistic Regression cannot be performed.')
                return
            if n_outcome == 2:
                print('Binary Logistic Regression:')
            elif n_outcome > 2:
                print('Multinomial Logistic Regression:')
            
            summary_df = pd.DataFrame()
            
            # # Univariate analysis
            for var in independent_var:
                
                if independent_assign and var in independent_assign:
                    independent_part = f"C({var}, Treatment(reference='{independent_assign[var]}'))"
                else:
                    independent_part = var
            
                formula = f"{dependent_var} ~ {independent_part}"
                
                if n_outcome == 2:
                    model = smf.logit(formula, df_test).fit(disp=0)
                                           
                    # Create a clean DataFrame
                    result_df = pd.DataFrame({
                        "predictor": model.params.index,  # Variable names
                        "coef (log OR)": np.round(model.params.values, 2),  # Log-odds (log(OR))
                        "OR": np.round(np.exp(model.params.values), 2),  # Exponentiated OR
                        "P>|z|": np.round(model.pvalues.values, 3),  # P-values formatted to 2 decimals
                        "[0.025": np.round(np.exp(model.conf_int()[0]).values, 2),  # Lower bound of CI in OR
                        "0.975]": np.round(np.exp(model.conf_int()[1]).values, 2)   # Upper bound of CI in OR
                    })
                    result_df = result_df[result_df["predictor"] != "Intercept"]
                
                    # Append results to summary_df
                    summary_df = pd.concat([summary_df, result_df], ignore_index=True)
            
            # Multivariate analysis

            formula_parts = []    
            
            for var in independent_var:
                if independent_assign and var in independent_assign:  # If variable has a reference group
                    formula_parts.append(f"C({var}, Treatment(reference='{independent_assign[var]}'))")
                else:  # Otherwise, include normally
                    formula_parts.append(var)
                    
            multivar_formula = f"{dependent_var} ~ " + " + ".join(formula_parts)  # Ensure no extra '+'
            
            if n_outcome == 2:
                
                model = smf.logit(multivar_formula, df_test).fit(disp=0)
                
                result_df = pd.DataFrame({
                    "predictor": model.params.index,  # Variable names
                    "coef (log OR)": np.round(model.params.values, 2),  # Log-odds (log(OR))
                    "OR": np.round(np.exp(model.params.values), 2),  # Exponentiated OR
                    "P>|z|": np.round(model.pvalues.values, 3),  # P-values formatted to 2 decimals
                    "[0.025": np.round(np.exp(model.conf_int()[0]).values, 2),  # Lower bound of CI in OR
                    "0.975]": np.round(np.exp(model.conf_int()[1]).values, 2)   # Upper bound of CI in OR
                })
                result_df = result_df[result_df["predictor"] != "Intercept"]    
               
            elif n_outcome > 2:
                
                model = smf.mnlogit(multivar_formula, df_test).fit(disp=0)      
               
                print(f"Pseudo R-squ.: {model.prsquared:.2f}")
                print(f"Log-Likelihood: {model.llf:.2f}")
                print(f"LL-Null: {model.llnull:.2f}")
                print(f"LLR p-value: {model.llr_pvalue:.2f}")
                
                # Extract model outputs
                params = model.params           # (predictors x outcomes)
                pvals = model.pvalues
                conf_int = model.conf_int()     # Must have 'lower', 'upper' columns

                # Reshape params to long form
                param_long = params.stack().reset_index()
                param_long.columns = ['predictor', 'outcome', 'coef (log OR)']
                param_long['coef (log OR)'] = np.round(param_long['coef (log OR)'], 3)
                param_long['OR'] = np.round(np.exp(param_long['coef (log OR)']), 3)

                # Reshape p-values
                pval_long = pvals.stack().reset_index()
                pval_long.columns = ['predictor', 'outcome', 'P>|z|']
                pval_long['P>|z|'] = np.round(pval_long['P>|z|'], 3)

                # Reshape confidence intervals
                conf_long = conf_int.stack().reset_index()
                conf_long.columns = ['outcome', 'predictor', 'bound', 'value']
                conf_long['value'] = np.round(np.exp(conf_long['value']), 3)

                # Pivot to wide form for CI
                conf_wide = conf_long.pivot(index=['predictor', 'outcome'], columns='bound', values='value').reset_index()
                conf_wide.columns.name = None
                conf_wide.rename(columns={'lower': '[0.025', 'upper': '0.975]'}, inplace=True)
                conf_wide['[0.025'] = np.round(conf_wide['[0.025'], 3)
                conf_wide['0.975]'] = np.round(conf_wide['0.975]'], 3)
                conf_wide['outcome'] = conf_wide['outcome'].astype(int) - 1
                

                # Ensure consistent dtype for merging
                outcome_labels = dependent_assign[1:]
                param_long['outcome'] = param_long['outcome'].astype(int).map(dict(enumerate(outcome_labels)))
                pval_long['outcome'] = pval_long['outcome'].astype(int).map(dict(enumerate(outcome_labels)))
                conf_wide['outcome'] = conf_wide['outcome'].astype(int).map(dict(enumerate(outcome_labels)))

                # Merge all components
                result_df = param_long.merge(pval_long, on=['predictor', 'outcome']).merge(conf_wide, on=['predictor', 'outcome'])
                result_df = result_df.sort_values(['outcome', 'predictor'])
                result_df = result_df[['outcome', 'predictor', 'coef (log OR)', 'OR', 'P>|z|', '[0.025', '0.975]']]
        
        summary_df = result_df.copy()       
        # summary_df = pd.merge(summary_df, result_df, how='left', on='predictor', suffixes=('', '_multivar'))
            
        print(tabulate(summary_df, showindex=False, headers="keys"))
        if export_result == True:
            ResultExport.add_to_docx(summary_df, table_name, output_dir='output/analyse')

        return summary_df