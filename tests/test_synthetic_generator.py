import hashlib

import pytest

from experiments.generate_synthetic import generate_dataset


def test_synthetic_generator_reproducible_and_preserves_existing_data(tmp_path):
    first = generate_dataset(tmp_path / 'one', per_class=3, seed=42)
    second = generate_dataset(tmp_path / 'two', per_class=3, seed=42)
    def hashes(root):
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob('*') if p.is_file()}
    assert hashes(first) == hashes(second)
    assert len(list(first.rglob('*.png'))) == 9
    assert 'Synthetic smoke data only' in (first / 'SYNTHETIC_ONLY.txt').read_text()
    with pytest.raises(ValueError, match='empty directory'):
        generate_dataset(first)
