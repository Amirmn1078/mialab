import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # Path to the "results.csv" file
    result_file = "C:/Users/amirm/Desktop/files/BME master/semester3/Medical image analysis lab/project/mialab/mia-result/2024-12-17-23-20-37/results.csv" # Adjust the path if needed

    try:
        # Read the data using pandas with the correct delimiter
        data = pd.read_csv(result_file, delimiter=';')

        # Check column names
        print("Columns in the dataset:", data.columns)

        # Verify the correct column names for labels, Dice coefficients, and Hausdorff distances
        label_column = "LABEL"  # Adjust based on actual column name
        dice_column = "DICE"  # Adjust based on actual column name
        hausdorff_column = "HDRFDST"  # Adjust based on actual column name

        if label_column not in data.columns or dice_column not in data.columns or hausdorff_column not in data.columns:
            raise KeyError(f"Columns '{label_column}', '{dice_column}', and/or '{hausdorff_column}' not found in the dataset.")

        # Extract the relevant columns for Dice coefficients
        labels = ['WhiteMatter', 'GreyMatter', 'Hippocampus', 'Amygdala', 'Thalamus']
        dice_data = {label: data[data[label_column] == label][dice_column] for label in labels}

        # Plot the Dice coefficients in a boxplot
        plt.boxplot(dice_data.values(), labels=dice_data.keys())
        plt.title("Dice Coefficients per Label")
        plt.ylabel("Dice Coefficient")
        plt.xlabel("Label")
        plt.show()

        # Extract the relevant columns for Hausdorff distances
        hausdorff_data = {label: data[data[label_column] == label][hausdorff_column] for label in labels}

        # Plot the Hausdorff distances in a boxplot
        plt.boxplot(hausdorff_data.values(), labels=hausdorff_data.keys())
        plt.title("Hausdorff Distances per Label")
        plt.ylabel("Hausdorff Distance (mm)")
        plt.xlabel("Label")
        plt.show()

    except FileNotFoundError:
        print(f"File not found: {result_file}. Ensure the results.csv file exists.")
    except KeyError as e:
        print(f"KeyError: {e}. Please verify the column names in the dataset.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
