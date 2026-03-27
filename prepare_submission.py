import os
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="Prepare Hackathon Submission Folder")
    parser.add_argument("--team", type=str, required=True, help="Your Team Name")
    parser.add_argument("--cls_model", type=str, required=True, help="Path to best classification model")
    parser.add_argument("--seg_model", type=str, required=True, help="Path to best segmentation model")
    parser.add_argument("--excel_path", type=str, required=True, help="Path to generated results Excel")
    parser.add_argument("--masks_dir", type=str, required=True, help="Path to predicted masks folder")
    args = parser.parse_args()

    team_root = args.team
    if os.path.exists(team_root):
        shutil.rmtree(team_root)
    os.makedirs(team_root)

    # 1. Excel File
    shutil.copy(args.excel_path, os.path.join(team_root, f"{args.team} test_ground_truth.xlsx"))

    # 2. Masks Folder
    shutil.copytree(args.masks_dir, os.path.join(team_root, args.team))

    # 3. Models & Scripts Folder
    models_dir = os.path.join(team_root, "models")
    os.makedirs(os.path.join(models_dir, "classification"))
    os.makedirs(os.path.join(models_dir, "segmentation"))

    # Copy Scripts
    shutil.copy("classify.py", os.path.join(models_dir, "classification", "classify.py"))
    shutil.copy("segment.py", os.path.join(models_dir, "segmentation", "segment.py"))
    
    # Copy Models
    shutil.copy(args.cls_model, os.path.join(models_dir, "classification", "best_model.pth"))
    shutil.copy(args.seg_model, os.path.join(models_dir, "segmentation", "best_model.pth"))

    # Copy Requirements
    if os.path.exists("requirements.txt"):
        shutil.copy("requirements.txt", os.path.join(models_dir, "classification", "requirements.txt"))
        shutil.copy("requirements.txt", os.path.join(models_dir, "segmentation", "requirements.txt"))

    print(f"✅ Submission folder '{team_root}' created successfully!")
    print(f"📦 ZIP this folder and upload to Google Form.")

if __name__ == "__main__":
    main()
