from pathlib import Path


def clean():
    print('Cleaning repository...')
    print(Path.cwd())
    print(Path('testing').is_dir())
    exts_to_remove = ['.tiff', '.txt', '.stl', '.vtk']
    for case_dir in Path('testing/cases').iterdir():
        for file in case_dir.rglob('*'):
            if (
                file.suffix in exts_to_remove
                or file.match('segment_cake_*.yml')
                or file.match('stdErr*')
                or file.match('tty*')
                or file.match('*~')
            ):
                file.unlink()


if __name__ == '__main__':
    clean()