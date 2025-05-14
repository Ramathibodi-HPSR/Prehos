import pandas as pd
import os
import numpy as np
from tabulate import tabulate

def print_line(prefix, char='─', total_width=None):
    """
    Print a dynamic line with a prefix and filling the rest with a specified character up to the total width of the terminal
    or a specified width.

    Parameters:
    - prefix (str): The text to display at the start of the line.
    - char (str): The character used to fill the line. Default is '─'.
    - total_width (int): Optional. Specify the total width for the line. If None, it tries to adjust to the terminal width or uses a default.
    """
    if total_width is None:
        try:
            _, total_width = os.get_terminal_size(0)  # using 0 to refer to standard output
        except OSError:
            total_width = 80  # default width if terminal size can't be determined

    # Calculate the number of characters needed to fill the line after the prefix
    fill_length = total_width - len(prefix) - 1  # -1 for the space after prefix

    # Ensure that the fill length is not negative
    if fill_length < 0:
        fill_length = 0

    # Print the line
    print(f"\n{prefix} {char * fill_length}")

def type_count(dataframe):
    type_counts = dataframe.dtypes.value_counts().rename_axis('Column type').reset_index(name='Frequency')
    for index, row in type_counts.iterrows():
        # Format line to align with the longest type name
        line = f"   {row['Column type']}{' ' * (20 - len(str(row['Column type'])))}{row['Frequency']}"
        print(line)
    return type_counts

def check_duplicate(dataframe):
    duplicates = dataframe.duplicated(keep=False)
    duplicate_count = duplicates.sum()
    print(f"Number of duplicated rows: {duplicate_count}")

def summarize_numeric(df, column):
    desc = df[column].describe(percentiles=[.01, .1, .25, .5, .75, .9, .99])
    data_clean = df[column].dropna()  # Drop NaN values for accurate calculations
    
    if data_clean.empty or data_clean.nunique() == 1:  # Check if data is empty or all values are the same
        hist_str = "Not applicable"  # No variation or data to make histogram
    else:
        # Calculate histogram with finite range
        histogram, bins = np.histogram(data_clean, bins=5)
        max_hist = max(histogram) if max(histogram) > 0 else 1  # Prevent division by zero
        hist_str = ''.join(['▁▂▃▄▅▆▇'[min(int(x), 6)] for x in np.floor(7 * histogram / max_hist)])  # Use min to cap the index

    stats = {
        'skim_variable': column,
        'n_missing': df[column].isnull().sum(),
        'complete_rate': round(1 - df[column].isnull().mean(), 2),
        'mean': round(desc['mean'], 2) if not data_clean.empty else 'NaN',
        'sd': round(desc['std'], 3) if not data_clean.empty else 'NaN',
        'p0': round(desc['min'], 1) if not data_clean.empty else 'NaN',
        'p1': round(desc['1%'], 1) if '1%' in desc else 'NaN',
        'p10': round(desc['10%'], 1) if '10%' in desc else 'NaN',
        'p25': round(desc['25%'], 1) if not data_clean.empty else 'NaN',
        'p50': round(desc['50%'], 1) if not data_clean.empty else 'NaN',
        'p75': round(desc['75%'], 1) if not data_clean.empty else 'NaN',
        'p90': round(desc['90%'], 1) if '90%' in desc else 'NaN',
        'p99': round(desc['99%'], 1) if '99%' in desc else 'NaN',
        'p100': round(desc['max'], 1) if not data_clean.empty else 'NaN',
        'hist': hist_str
    }
    return stats

def summarize_object(df, column):
    col = df[column].dropna().astype(str)
    top_counts = col.value_counts().head(3).to_dict()
    
    # Adjusting top values to be exactly 10 characters long
    adjusted_top_counts = {k[:10] if len(k) > 10 else k: v for k, v in top_counts.items()}
    
    stats = {
        'skim_variable': column,
        'n_missing': df[column].isnull().sum(),
        'complete_rate': round(1 - df[column].isnull().mean(), 2),
        'n_unique': df[column].nunique(),
        'top_counts': adjusted_top_counts
    }
    return stats

def summarize_string(df, column):
    col = df[column].dropna().astype(str)  # Convert non-null data to string
    top_values = col.value_counts().head(3).to_dict()
    
    # Adjusting top values to be exactly 10 characters long
    adjusted_top_values = {k[:10] if len(k) > 10 else k: v for k, v in top_values.items()}
    
    stats = {
        'skim_variable': column,
        'n_missing': df[column].isnull().sum(),
        'complete_rate': round(1 - df[column].isnull().mean(), 2),
        'n_unique': col.nunique(),
        'min_length': col.map(len).min(),
        'max_length': col.map(len).max(),
        'mean_length': col.map(len).mean(),
        'whitespace': col.apply(lambda x: ' ' in x).sum(),
        'top_values': adjusted_top_values,
    }
    return stats

def format_timedelta(td):
    # Extract components from the timedelta
    days = td.days
    seconds = td.seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Determine the largest non-zero unit to display
    if days != 0:
        return f"{days} days"
    elif hours != 0:
        return f"{hours} hours"
    elif minutes != 0:
        return f"{minutes} minutes"
    elif seconds != 0:
        return f"{seconds} seconds"
    else:
        return "0 seconds"

def summarize_datetime(df, column):
    datetime_data = df[column].dropna()

    # Basic statistics
    min_date = datetime_data.min()
    max_date = datetime_data.max()
    date_range = max_date - min_date

    # Calculating basic statistics
    stats = {
        'skim_variable': column,
        'n_missing': df[column].isnull().sum(),
        'complete_rate': round(1 - df[column].isnull().mean(), 2),
        'min_date': min_date,
        'max_date': max_date,
        'range': format_timedelta(date_range),  # Use format_timedelta to display the range
    }

    # Calculate intervals and their statistics if there are enough data points
    if len(datetime_data) > 1:
        intervals = datetime_data.sort_values().diff().dropna()
        stats.update({
            'mean_interval': format_timedelta(intervals.mean()),  # Format mean interval
            'min_interval': format_timedelta(intervals.min()),    # Format minimum interval
            'max_interval': format_timedelta(intervals.max()),    # Format maximum interval
        })

    # Adding frequency of common dates or periods
    frequency = datetime_data.value_counts().nlargest(1)
    stats['most_frequent'] = frequency.to_dict()

    return stats

def summarize_timedelta(df, column, unit='minutes'):
    # Convert timedelta to specified calculation unit
    if unit == 'seconds':
        data_seconds = df[column].dropna().dt.total_seconds()
    elif unit == 'minutes':
        data_seconds = df[column].dropna().dt.total_seconds() / 60
    elif unit == 'hours':
        data_seconds = df[column].dropna().dt.total_seconds() / 3600
    elif unit == 'days':
        data_seconds = df[column].dropna().dt.total_seconds() / 86400
    else:
        raise ValueError("Unsupported calculation unit. Use 'seconds', 'minutes', 'hours', or 'days'.")

    # Convert data for display
    data_clean = data_seconds

    # Describe the data
    desc = data_clean.describe(percentiles=[.01, .1, .25, .5, .75, .9, .99])

    # Histogram computation
    if data_clean.empty or data_clean.nunique() == 1:
        hist_str = "Not applicable"  # No variation or data to make histogram
    else:
        # Calculate histogram with finite range
        histogram, bins = np.histogram(data_clean, bins=5)
        max_hist = max(histogram) if max(histogram) > 0 else 1  # Prevent division by zero
        hist_str = ''.join(['▁▂▃▄▅▆▇'[min(int(x), 6)] for x in np.floor(7 * histogram / max_hist)])  # Use min to cap the index

    # Build stats dictionary
    stats = {
        'skim_variable': column,
        'n_missing': df[column].isnull().sum(),
        'complete_rate': round(1 - df[column].isnull().mean(), 2),
        'mean': round(desc['mean'], 2) if not data_clean.empty else 'NaN',
        'sd': round(desc['std'], 3) if not data_clean.empty else 'NaN',
        'p0': round(desc['min'], 1) if not data_clean.empty else 'NaN',
        'p1': round(desc['1%'], 1) if '1%' in desc else 'NaN',
        'p10': round(desc['10%'], 1) if '10%' in desc else 'NaN',
        'p25': round(desc['25%'], 1) if not data_clean.empty else 'NaN',
        'p50': round(desc['50%'], 1) if not data_clean.empty else 'NaN',
        'p75': round(desc['75%'], 1) if not data_clean.empty else 'NaN',
        'p90': round(desc['90%'], 1) if '90%' in desc else 'NaN',
        'p99': round(desc['99%'], 1) if '99%' in desc else 'NaN',
        'p100': round(desc['max'], 1) if not data_clean.empty else 'NaN',
        'hist': hist_str
    }
    return stats
    

def skim(df, df_preprocess=None, df_name="DataFrame"):
    process = None
    while process == None:
        process = input("Did you process your dataframe before skimming (Yes/No): ").lower()
        if process == 'no':
            print("Please process your data before skimming")
        elif process == 'yes':
            
            #Building the report
            
            print_line("── Data Summary")
            
            # Information about the DataFrame
            df_summary = {
                'Name': df_name,
                'Number of rows': df.shape[0],
                'Number of columns': df.shape[1]
            }

            # Printing out the information
            for key, value in df_summary.items():
                print(f"{key:25}: {value}")
            
            if df_preprocess is not None:
                print_line("── Pre-processing Data")
                print(f"{'Number of rows':25}: {len(df_preprocess)}")
                check_duplicate(df_preprocess)
                print("Variable types:")
                type_count(df_preprocess)
            
            print_line("── Variable Types")
            print(f"Number of rows : {len(df)}")

            # Count the number of incomplete rows
            incomplete_rows = df.isnull().any(axis=1).sum()
            incomplete_percentage = (incomplete_rows / len(df)) * 100

            print(f"Number of incomplete rows: {incomplete_rows} ({incomplete_percentage:.2f}%)")
            print("Variable types:")
            types = type_count(df)
            
            for _, row in types.iterrows():
                type = row['Column type']
                print_line(f"── Variable type: {type}")
                if type == 'string':
                    stats = []
                    for column in df.select_dtypes(include=[type]).columns:
                        stat = summarize_string(df, column)
                        stats.append(stat)
                    stats_df = pd.DataFrame(stats).sort_values(by='skim_variable')
                    print(tabulate(stats_df, showindex=False, headers=stats_df.columns))
                elif type == 'object':
                    stats = []
                    for column in df.select_dtypes(include=[type]).columns:
                        stat = summarize_object(df, column)
                        stats.append(stat)
                    stats_df = pd.DataFrame(stats).sort_values(by='skim_variable')
                    print(tabulate(stats_df, showindex=False, headers=stats_df.columns))
                elif type in ['float64', 'int64']:
                    stats = []
                    for column in df.select_dtypes(include=[type]).columns:
                        stat = summarize_numeric(df, column)
                        stats.append(stat)
                    stats_df = pd.DataFrame(stats).sort_values(by='skim_variable')
                    print(tabulate(stats_df, showindex=False, headers=stats_df.columns))
                elif type in ['datetime64[ns]']:
                    stats = []
                    for column in df.select_dtypes(include=[type]).columns:
                        stat = summarize_datetime(df, column)
                        stats.append(stat)
                    stats_df = pd.DataFrame(stats).sort_values(by='skim_variable')
                    print(tabulate(stats_df, showindex=False, headers=stats_df.columns))
                elif type in ['timedelta64[ns]']:
                    stats = []
                    unit = input(f"Which unit do you want to display ('seconds', 'minutes', 'hours', or 'days'): ").lower()
                    for column in df.select_dtypes(include=[type]).columns:
                        stat = summarize_timedelta(df, column, unit=unit)
                        stats.append(stat)
                    stats_df = pd.DataFrame(stats).sort_values(by='skim_variable')
                    print(tabulate(stats_df, showindex=False, headers=stats_df.columns))
        else:
            process = None
        
        
        