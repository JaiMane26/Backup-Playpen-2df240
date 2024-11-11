import os
import pandas as pd

RISK_TYPES = [
    "bias",
    "misinformation",
    "misalignment",
    "hallucination",
    "toxicity",
    "privacy",
    "ip_copyright",
    "attacks",
    "economic_crime",
    "environmental",
]

class DatasetHandler:
    def __init__(self, data_folder='data'):
        self.data_folder = data_folder
        self.datasets = {}

    def load_dataset(self, risk_type):
        if risk_type not in RISK_TYPES:
            print(f"Invalid risk type: {risk_type}")
            return None

        if risk_type in self.datasets:
            return self.datasets[risk_type]

        file_path_csv = os.path.join(self.data_folder, f"{risk_type}.csv")
        file_path_parquet = os.path.join(self.data_folder, f"{risk_type}.parquet")

        if os.path.exists(file_path_csv):
            self.datasets[risk_type] = pd.read_csv(file_path_csv)
        elif os.path.exists(file_path_parquet):
            self.datasets[risk_type] = pd.read_parquet(file_path_parquet)
        else:
            print(f"No dataset found for {risk_type}")
            return None

        return self.datasets[risk_type]

    def get_data(self, risk_type):
        return self.load_dataset(risk_type)

    def preprocess_data(self, risk_type):
        data = self.load_dataset(risk_type)
        if data is not None:
            # Example preprocessing steps
            data.dropna(inplace=True)
            data.reset_index(drop=True, inplace=True)
            self.datasets[risk_type] = data
        else:
            print(f"No dataset loaded for {risk_type}")

    def save_data(self, risk_type, output_format='csv'):
        if risk_type in self.datasets:
            output_path = os.path.join(self.data_folder, f"{risk_type}.{output_format}")
            if output_format == 'csv':
                self.datasets[risk_type].to_csv(output_path, index=False)
            elif output_format == 'parquet':
                self.datasets[risk_type].to_parquet(output_path, index=False)
            else:
                print(f"Unsupported format: {output_format}")
        else:
            print(f"No dataset loaded for {risk_type}")

# Example usage
handler = DatasetHandler()
data = handler.get_data('bias')
# print(data)
if data is not None:
    handler.preprocess_data('bias')
    handler.save_data('bias', 'parquet')
