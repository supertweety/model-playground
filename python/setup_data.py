"""Create empty split/class folders; never move or copy photos automatically."""
import argparse
from pathlib import Path
from common import CLASSES

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data'))
    args = parser.parse_args()
    for split in ('train', 'validation', 'test'):
        for label in CLASSES:
            (args.data / split / label).mkdir(parents=True, exist_ok=True)
    print(f'Created empty O/X folders in {args.data.resolve()}')
