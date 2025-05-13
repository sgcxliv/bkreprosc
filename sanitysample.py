#whenever i do something to a csv i take the first 50 lines to make sure i didn't horrendously mess it up
import pandas as pd

def extract_first_n_rows(input_file, output_file, n=50):
    """
    Extract the first n rows from a CSV file and save to a new file.
    
    Parameters:
    input_file (str): Path to the input CSV file
    output_file (str): Path to save the output CSV file
    n (int): Number of rows to extract (default: 500)
    """
    try:
        df = pd.read_csv(input_file, nrows=n)
        
        df.to_csv(output_file, index=False)
        
        print(f"Successfully extracted first {n} rows from {input_file} and saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = "bk21_spr_with_folds.csv"
    output_file = "fifty.csv"  
    
    extract_first_n_rows(input_file, output_file, 50)

