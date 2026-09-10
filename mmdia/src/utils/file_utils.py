import os
import shutil

class FileUploader:
    def __init__(self, server_ip, upload_dir):
        self.server_ip = server_ip
        self.upload_dir = upload_dir

    def upload(self, local_path, prefix=""):
        filename = f"{prefix}_{os.path.basename(local_path)}"
        target_path = os.path.join(self.upload_dir, filename)
        try:
            shutil.copy(local_path, target_path)
            return f"http://{self.server_ip}/upload/{filename}"
        except Exception as e:
            print(f"Upload failed: {e}")
            return None

    def delete(self, filename, prefix=""):
        full_name = f"{prefix}_{filename}"
        target_path = os.path.join(self.upload_dir, full_name)
        if os.path.exists(target_path):
            os.remove(target_path)
            return True
        return False