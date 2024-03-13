import pathlib


def rmtree(f: pathlib.Path, missing_ok: bool = True):
    if not f.exists() and missing_ok:
        return

    if f.is_file():
        f.unlink()
    else:
        for child in f.iterdir():
            rmtree(child)
        f.rmdir()
