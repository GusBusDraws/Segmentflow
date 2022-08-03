from pathlib import Path


def clean():
    print('Cleaning repository...')
    n_removed = 0
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
                n_removed += 1
    print(f'Cleaning complete. {n_removed} file(s) removed.')


if __name__ == '__main__':
    clean()