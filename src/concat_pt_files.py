import os
import torch
import argparse
from tqdm import tqdm
import re

def concat_pt_files(output_folder, save_path=None, disable_filter=False):
    """
    Load and concatenate all .pt files in the output_folder.

    Args:
        output_folder (str): Path to the folder containing .pt files.
        save_path (str, optional): Path to save the concatenated results (as .pt). If None, won't save.

    Returns:
        List[dict]: List of loaded dictionaries from .pt files.
    """
    if not os.path.exists(output_folder):
        raise FileNotFoundError(f"Output folder '{output_folder}' does not exist.")

    pt_files = [f for f in os.listdir(output_folder) if f.endswith('.pt') and not f.startswith('.')]
    if args.disable_filter:
        pt_files = [f for f in os.listdir(output_folder) if f.endswith('.pt')]


    print(f'Found pt files: {pt_files}, len(pt_files) = {len(pt_files)}')

    def extract_task_number(filename):
        match = re.search(r'Task_(\d+)\.pt', filename)
        return int(match.group(1)) if match else float('inf')

    if not args.disable_filter:
        pt_files.sort(key=extract_task_number)

    results = []
    for filename in tqdm(pt_files, desc="Loading .pt files"):
        file_path = os.path.join(output_folder, filename)
        try:
            data = torch.load(file_path)
            results.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(results)} samples.")

    # Optionally save concatenated results
    if save_path:
        torch.save(results, save_path)
        print(f"Concatenated results saved to {save_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate .pt files from a folder.")
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the folder containing the .pt files (e.g., 'output/')."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="concatenated_results.pt",
        help="Path to save the concatenated results. Set to None to skip saving."
    )
    parser.add_argument("--disable_filter", action='store_true')

    args = parser.parse_args()

    results = concat_pt_files(args.output_folder, args.save_path, args.disable_filter)