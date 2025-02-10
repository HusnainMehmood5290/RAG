import os
import shutil

# Directories to remove completely
DIRS_TO_REMOVE = [
    "ingestion/converted",
    "ingestion/__pycache__",
    "models/__pycache__",
    "store/local_store",
    "store/vector_store",
]

def remove_directory(directory):
    """Deletes a directory and all its contents if it exists."""
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)  # Remove entire directory
            print(f"Deleted directory: {directory}")
        except Exception as e:
            print(f"ERROR: Failed to delete {directory} - {e}")
    else:
        print(f"Skipping: {directory} (Does not exist)")

def clean_all():
    """Removes all specified directories."""
    for directory in DIRS_TO_REMOVE:
        remove_directory(directory)

if __name__ == "__main__":
    clean_all()
    print("Cleanup complete! 🚀")
