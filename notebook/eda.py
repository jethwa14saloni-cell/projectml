import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import sys

warnings.filterwarnings("ignore")
import seaborn as sns
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import OneHotEncoder


@dataclass
class EDAConfig:
    eda_report_path: str = "artifacts/eda_report.csv"
    eda_plots_path: str = "artifacts/eda_plots/"


class EDA:
    def __init__(self):
        self.eda_config = EDAConfig()
        self.file_path = None

    @property
    def load_data(self) -> pd.DataFrame:
        try:
            logging.info(f"Loading data from {self.file_path}")
            df = pd.read_csv(self.file_path)
            logging.info("Data loaded successfully")
            # split the data into features and target
            X = df.drop("math_score", axis=1)
            y = df["math_score"]
            return df
        except Exception as e:
            logging.error("Error loading data")
            raise CustomException(e, sys)

    def pre_process(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Generating pre_process report")
            data_shape = df.shape
            logging.info(f"Data shape: {data_shape}")
            missing_values = df.isnull().sum()
            logging.info(f"Missing values:\n{missing_values}")
            duplicate_rows = df.duplicated().sum()
            logging.info(f"Duplicate rows: {duplicate_rows}")
            data_description = df.describe()
            logging.info(f"Data description:\n{data_description}")
            data_unique = df.nunique()
            logging.info(f"Unique values per column:\n{data_unique}")
            report = {
                "shape": data_shape,
                "missing_values": missing_values,
                "duplicate_rows": duplicate_rows,
                "data_description": data_description,
                "unique_values": data_unique,
            }

            return report
        except Exception as e:
            logging.error("Error generating EDA report")
            raise CustomException(e, sys)

    def handle_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Handling categorical features")
            categorical_cols = df.select_dtypes(include="object").columns.tolist()
            logging.info(f"Categorical columns: {categorical_cols}")

            # number of unique categories in each categorical column
            for col in categorical_cols:
                num_unique = df[col].nunique()
                logging.info(f"Column '{col}' has {num_unique} unique categories")

            # make table  of all categorical columns with their unique values
            categorical_summary = {}
            for col in categorical_cols:
                categorical_summary[col] = df[col].value_counts().to_dict()
            categorical_summary_df = pd.DataFrame.from_dict(
                categorical_summary, orient="index"
            ).transpose()

            # Log the DataFrame
            logging.info(f"Categorical summary DataFrame:\n{categorical_summary_df}")

            return df
        except Exception as e:
            logging.error("Error handling categorical features")
            raise CustomException(e, sys)

    def plot_exploring_data(self, df: pd.DataFrame) -> None:
        try:
            logging.info("Generating EDA plots")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include="object").columns.tolist()

            # sum of numeric columns and calculate average and total
            df["total_score"] = df[numeric_cols].sum(axis=1)
            df["average_score"] = df[numeric_cols].mean(axis=1)

            # Idenify people who scored full marks in all subjects
            df_full_marks = df[numeric_cols].apply(lambda x: all(x == 100), axis=1)
            # Idenify people who scored zero in all subjects
            df_zero_scores = df[numeric_cols].apply(lambda x: all(x == 0), axis=1)
            logging.info(f"Number of students with full marks: {df_full_marks.sum()}")
            logging.info(f"Number of students with zero scores: {df_zero_scores.sum()}")
            # Histograms for numeric columns

            df["avg_rounded"] = df["average_score"].round(0)

            # Create a count plot
            plt.figure(figsize=(12, 6))
            sns.countplot(x="avg_rounded", data=df, color="skyblue")

            plt.title("Number of People by Average Marks", fontsize=14)
            plt.xlabel("Average Marks", fontsize=12)
            plt.ylabel("Number of People", fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            for col in numeric_cols:
                plt.figure(figsize=(8, 6))
                sns.histplot(df[col], kde=True)
                plt.title(f"Histogram of {col}")
                plt.savefig(f"{col}_histogram.png")
                plt.show()

                return

        except Exception as e:
            logging.error("Error generating EDA plots")
            raise CustomException(e, sys)

    # split data train and test dataset



if __name__ == "__main__":
    eda = EDA()
    eda.file_path = r"D:\saloni\mlproject\data\stud.csv"
    data = eda.load_data
    report = eda.pre_process(data)
    data = eda.handle_categorical_variables(data)
    eda.plot_exploring_data(data)
    data = eda.initiate_data_ingestion()

    print(report)
