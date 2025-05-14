import os
from huggingface_hub import snapshot_download

def ensure_model(model_name, local_dir):
    if not os.path.exists(local_dir) or not os.path.exists(os.path.join(local_dir, "pytorch_model.bin")):
        print(f"Downloading {model_name} to {local_dir} ...")
        snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)
    else:
        print(f"Model {model_name} already exists at {local_dir}")

if __name__ == "__main__":
    ensure_model("skt/kobert-base-v1", "data/models/kobert")
    ensure_model("beomi/kcbert-base", "data/models/kcbert")
