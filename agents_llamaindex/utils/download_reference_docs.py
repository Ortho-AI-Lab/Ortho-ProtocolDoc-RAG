from pathlib import Path
import gdown

from constants import PACKAGE_DIR


path_to_reference_docs = PACKAGE_DIR / "reference_docs"
path_to_reference_docs.mkdir(exist_ok=True, parents=True)


def download_reference_docs(
    id: str = "1kXniocenvR3EjaFQb8HuDbthkGizH6zr",
    output: Path = path_to_reference_docs / "reference_docs.zip",
) -> None:
    gdown.download(id=id, output=str(output))


def unzip_reference_docs(
    file_path: Path = path_to_reference_docs / "reference_docs.zip",
) -> None:
    import zipfile

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(file_path.parent)


def clean_up_reference_docs(dir_path: Path = path_to_reference_docs) -> None:
    (dir_path / "reference_docs.zip").unlink()
    try:
        (dir_path / "__MACOSX").rmdir()
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    download_reference_docs()
    unzip_reference_docs()
    clean_up_reference_docs()
    print("Reference documents downloaded successfully.")
