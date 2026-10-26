"""Import explicit human review into a separate version-bound LATeD sidecar."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from seeread.v5.lated_annotations import import_reviews

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = import_reviews(a.package, a.review, a.output)
    print(json.dumps({k: result[k] for k in ('reviewed_instances', 'total_instances')}, ensure_ascii=False))
