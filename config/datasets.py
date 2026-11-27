"""
Dataset configurations and schemas for PerForm project.
"""

import os
from config.paths import DATA_DIR

DATASET_TYPES = {
    "conductivity": {
        "name": "Ionic Conductivity",
        "required_columns": ["Row_ID", "IC", "LIC", "Temperature"],
        "optional_columns": ["Electrolyte", "Reference"],
        "molecule_columns": {
            "solvents": [
                "Solvent_1",
                "Solvent_2",
                "Solvent_3",
                "Solvent_4",
                "Solvent_5",
            ],
            "salts": ["Salt_1", "Salt_2", "Salt_3"],
        },
        "max_solvents": 5,
        "max_salts": 3,
        "description": "Electrolyte dataset for ionic conductivity prediction",
    },
}


def get_dataset_files():
    """List the active public benchmark table."""
    path = os.path.join(DATA_DIR, "splits-calisol", "calisol.csv")
    return [path] if os.path.isfile(path) else []


def detect_dataset_type(file_path):
    """Detect dataset type based on column names."""
    try:
        import pandas as pd

        df = pd.read_csv(file_path, nrows=1)
        columns = df.columns.tolist()

        if "IC" in columns:
            return "conductivity"
        else:
            return None
    except Exception:
        return None


def get_dataset_config(dataset_type):
    """Get configuration for a specific dataset type."""
    return DATASET_TYPES.get(dataset_type)


def interactive_dataset_choice():
    """Interactive dataset selection with type detection."""
    dataset_files = get_dataset_files()

    print("\n" + "=" * 60)
    print("DATASET SELECTION")
    print("=" * 60)

    if not dataset_files:
        print("No dataset files found in data directory!")
        return None, None

    print("\nAvailable datasets:")
    valid_datasets = []

    for i, file_path in enumerate(dataset_files, 1):
        filename = os.path.basename(file_path)
        dataset_type = detect_dataset_type(file_path)

        if dataset_type:
            type_info = DATASET_TYPES[dataset_type]
            print(f"{i}. {filename} ({type_info['name']})")
            valid_datasets.append((file_path, dataset_type))
        else:
            print(f"{i}. {filename} (Unknown type)")
            valid_datasets.append((file_path, None))

    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(valid_datasets)}): ")
            if not choice.strip():
                print("No dataset selected. Exiting.")
                return None, None

            idx = int(choice) - 1
            if 0 <= idx < len(valid_datasets):
                file_path, dataset_type = valid_datasets[idx]
                if dataset_type is None:
                    print("Warning: Could not determine dataset type for this file.")
                    return file_path, None

                type_info = DATASET_TYPES[dataset_type]
                print(f"Selected: {os.path.basename(file_path)} ({type_info['name']})")
                return file_path, dataset_type
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting.")
            return None, None


def interactive_conductivity_dataset_choice():
    """Interactive dataset selection for conductivity datasets only."""
    dataset_files = get_dataset_files()

    print("\n" + "=" * 60)
    print("CONDUCTIVITY DATASET SELECTION")
    print("=" * 60)

    if not dataset_files:
        print("No dataset files found in data directory!")
        return None, None

    print("\nAvailable conductivity datasets:")
    conductivity_datasets = []

    for file_path in dataset_files:
        filename = os.path.basename(file_path)
        dataset_type = detect_dataset_type(file_path)

        if dataset_type == "conductivity":
            conductivity_datasets.append((file_path, dataset_type))

    if not conductivity_datasets:
        print("No conductivity datasets found!")
        return None, None

    for i, (file_path, dataset_type) in enumerate(conductivity_datasets, 1):
        filename = os.path.basename(file_path)
        type_info = DATASET_TYPES[dataset_type]
        print(f"{i}. {filename} ({type_info['name']})")

    while True:
        try:
            choice = input(
                f"\nSelect conductivity dataset (1-{len(conductivity_datasets)}): "
            )
            if not choice.strip():
                print("No dataset selected. Exiting.")
                return None, None

            idx = int(choice) - 1
            if 0 <= idx < len(conductivity_datasets):
                file_path, dataset_type = conductivity_datasets[idx]
                type_info = DATASET_TYPES[dataset_type]
                print(f"Selected: {os.path.basename(file_path)} ({type_info['name']})")
                return file_path, dataset_type
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting.")
            return None, None
