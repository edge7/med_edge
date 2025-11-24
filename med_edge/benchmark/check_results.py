"""
Check benchmark results and print accuracies.

Scans benchmark directories, identifies models, and reports accuracy for:
- med_qa (train, val, test)
- medagents (test only, multiple configs)
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict


# Directories
MED_QA_DIR = "/media/edge7/Extreme Pro/med_edge/benchmarks"
MEDAGENTS_DIR = "/media/edge7/Extreme Pro/med_edge/benchmarks_test"


def extract_model_from_filename(filename):
    """
    Extract model name from filename.

    Examples:
        openai_gpt-oss-120b_train.csv -> openai/gpt-oss-120b
        openai_gpt-oss-120b_medagents_AfrimedQA_test.csv -> openai/gpt-oss-120b
    """
    # Remove .csv extension
    name = filename.replace('.csv', '')

    # Split by underscore
    parts = name.split('_')

    # Model is first parts (before split or medagents)
    # Examples:
    #   openai_gpt-oss-120b_train -> ['openai', 'gpt-oss-120b', 'train']
    #   openai_gpt-oss-120b_medagents_... -> ['openai', 'gpt-oss-120b', 'medagents', ...]

    # Find where the model name ends (at 'train', 'val', 'test', or 'medagents')
    model_parts = []
    for part in parts:
        if part in ['train', 'val', 'test', 'medagents']:
            break
        model_parts.append(part)

    # Reconstruct model name with / instead of first _
    if len(model_parts) >= 2:
        model = f"{model_parts[0]}/{model_parts[1]}"
        if len(model_parts) > 2:
            model += '-' + '-'.join(model_parts[2:])
        return model

    return None


def calculate_accuracy(csv_file):
    """Calculate accuracy from CSV file."""
    try:
        df = pd.read_csv(csv_file)

        if 'is_correct' not in df.columns:
            return None, 0, "Missing 'is_correct' column"

        total = len(df)
        correct = df['is_correct'].sum()
        accuracy = (correct / total * 100) if total > 0 else 0

        return accuracy, total, None
    except Exception as e:
        return None, 0, str(e)


def check_med_qa():
    """Check med_qa results (train, val, test)."""
    print("="*80)
    print("MED_QA RESULTS (bigbio/med_qa)")
    print("="*80)

    results_dir = Path(MED_QA_DIR)

    if not results_dir.exists():
        print(f"Directory not found: {MED_QA_DIR}")
        return

    # Group files by model
    models = defaultdict(dict)

    for csv_file in results_dir.glob("*.csv"):
        filename = csv_file.name

        # Skip temp files
        if filename.endswith('.tmp'):
            continue

        # Extract model and split
        model = extract_model_from_filename(filename)

        # Determine split (train, val, test)
        if '_train.csv' in filename:
            split = 'train'
        elif '_val.csv' in filename:
            split = 'val'
        elif '_test.csv' in filename:
            split = 'test'
        else:
            continue

        if model:
            models[model][split] = csv_file

    if not models:
        print("No results found.")
        return

    # Print results for each model
    for model in sorted(models.keys()):
        print(f"\nModel: {model}")
        print("-" * 80)

        for split in ['train', 'val', 'test']:
            if split in models[model]:
                csv_file = models[model][split]
                accuracy, total, error = calculate_accuracy(csv_file)

                if error:
                    print(f"  {split:6s}: ERROR - {error}")
                else:
                    print(f"  {split:6s}: {accuracy:6.2f}% ({int(accuracy * total / 100)}/{total})")
            else:
                print(f"  {split:6s}: Not found")


def check_medagents():
    """Check medagents results (test only, multiple configs)."""
    print("\n")
    print("="*80)
    print("MEDAGENTS RESULTS (super-dainiu/medagents-benchmark)")
    print("="*80)

    results_dir = Path(MEDAGENTS_DIR)

    if not results_dir.exists():
        print(f"Directory not found: {MEDAGENTS_DIR}")
        return

    # Group files by model and config
    models = defaultdict(lambda: defaultdict(dict))

    for csv_file in results_dir.glob("*.csv"):
        filename = csv_file.name

        # Skip temp files
        if filename.endswith('.tmp'):
            continue

        # Extract model
        model = extract_model_from_filename(filename)

        # Extract config (between medagents_ and _test)
        # Example: openai_gpt-oss-120b_medagents_AfrimedQA_test.csv
        if '_medagents_' in filename and '_test.csv' in filename:
            parts = filename.split('_medagents_')[1].split('_test.csv')[0]
            config = parts
        else:
            continue

        if model:
            models[model][config] = csv_file

    if not models:
        print("No results found.")
        return

    # Print results for each model
    for model in sorted(models.keys()):
        print(f"\nModel: {model}")
        print("-" * 80)

        # Calculate overall accuracy across all configs
        total_all = 0
        correct_all = 0

        for config in sorted(models[model].keys()):
            csv_file = models[model][config]
            accuracy, total, error = calculate_accuracy(csv_file)

            if error:
                print(f"  {config:20s}: ERROR - {error}")
            else:
                correct = int(accuracy * total / 100)
                total_all += total
                correct_all += correct
                print(f"  {config:20s}: {accuracy:6.2f}% ({correct}/{total})")

        # Print overall
        if total_all > 0:
            overall_acc = (correct_all / total_all) * 100
            print(f"  {'-'*20}")
            print(f"  {'OVERALL':20s}: {overall_acc:6.2f}% ({correct_all}/{total_all})")


def main():
    """Main function."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "BENCHMARK RESULTS CHECKER" + " "*33 + "║")
    print("╚" + "="*78 + "╝")

    check_med_qa()
    check_medagents()

    print("\n" + "="*80)
    print("Done.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()