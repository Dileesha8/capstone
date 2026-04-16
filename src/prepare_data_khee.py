import os
import shutil

def organize_knee_dataset():
    source_base = 'khee_oes_dataset'
    target_base = 'knee_dataset'

    grade_map = {
        '0': 'normal',
        '1': 'doubtful',
        '2': 'mild',
        '3': 'moderate',
        '4': 'severe'
    }

    # Clear old knee_dataset if exists
    if os.path.exists(target_base):
        shutil.rmtree(target_base)
        print("Cleared old knee_dataset")

    # Create all target directories
    for split in ['train', 'val', 'test']:
        for class_name in grade_map.values():
            os.makedirs(os.path.join(target_base, split, class_name), exist_ok=True)

    def copy_grade_folders(source_folder, target_split):
        path = os.path.join(source_base, source_folder)
        if not os.path.exists(path):
            return 0
        total = 0
        print(f"\n  Processing {source_folder} -> {target_split}/...")
        for grade, class_name in grade_map.items():
            grade_path = os.path.join(path, grade)
            if not os.path.exists(grade_path):
                continue
            images = [f for f in os.listdir(grade_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img in images:
                src = os.path.join(grade_path, img)
                dst = os.path.join(target_base, target_split, class_name, img)
                shutil.copy2(src, dst)
            print(f"    Grade {grade} ({class_name}): {len(images)} images")
            total += len(images)
        return total

    total = 0
    total += copy_grade_folders('train', 'train')
    total += copy_grade_folders('val', 'val')
    total += copy_grade_folders('test', 'test')
    total += copy_grade_folders('auto_test', 'test')

    # khee_normal -> train/normal
    khee_path = os.path.join(source_base, 'khee_normal')
    if os.path.exists(khee_path):
        print(f"\n  Processing khee_normal -> train/normal...")
        for folder in ['0Normal', '1Normal']:
            fp = os.path.join(khee_path, folder)
            if not os.path.exists(fp):
                continue
            images = [f for f in os.listdir(fp)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img in images:
                src = os.path.join(fp, img)
                dst = os.path.join(target_base, 'train', 'normal', img)
                shutil.copy2(src, dst)
            print(f"    {folder}: {len(images)} images")
            total += len(images)

    print(f"\nTotal images: {total}")
    print("\nFinal counts:")
    for split in ['train', 'val', 'test']:
        print(f"\n  {split.upper()}:")
        for class_name in grade_map.values():
            path = os.path.join(target_base, split, class_name)
            count = len(os.listdir(path)) if os.path.exists(path) else 0
            print(f"    {class_name}: {count}")

if __name__ == "__main__":
    organize_knee_dataset()