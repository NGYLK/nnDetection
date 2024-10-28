import os
import glob

# Define paths
image_path = "/home/usst/znn/nnDetection/det_data/Task002_ProstateX/raw_splitted/imagesTr"
label_path = "/home/usst/znn/nnDetection/det_data/Task002_ProstateX/raw_splitted/labelsTr"

# Extract the IDs from the warning log
problematic_ids = [
    "QG12072089", "QG12064695", "QG12068575", "V0142824X", "QG12093197", "U00347643",
    "QG12084526", "QG12027981", "ProstateX-0084", "P31953885", "ProstateX-0198",
    "M05278255", "K06839144", "ProstateX-0099", "P10273303", "M0279764X", "K01463766",
    "J02346898", "K03348008", "7100212028", "K00407631", "10049284", "J03856355",
    "06250821", "06120981", "05128148"
]

# Helper function to delete files based on prefix
def delete_files_by_prefix(folder_path, prefix):
    files = glob.glob(os.path.join(folder_path, f"{prefix}*"))
    for file in files:
        print(f"Deleting: {file}")
        os.remove(file)


# Delete files related to problematic IDs
for file_id in problematic_ids:
    delete_files_by_prefix(image_path, file_id)  # Delete images
    delete_files_by_prefix(label_path, file_id)  # Delete labels and JSONs
