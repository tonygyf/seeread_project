import json
from pathlib import Path
import tempfile
import unittest

from tools.v5_final_reading import FORMAT, digest, evaluate_locked, freeze


class FinalProtocolTests(unittest.TestCase):
    def test_modified_locked_protocol_fails_before_model_or_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lock.json'
            protocol = dict(format=FORMAT, baseline='frozen', config={'a': 1})
            protocol['protocol_sha256'] = digest(protocol)
            protocol['config']['a'] = 2
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, 'modified'):
                evaluate_locked(path)

    def test_last_checkpoint_cannot_be_locked_as_dev_selected_best(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.yaml'
            config.write_text('recognizer: {trainability: lora}\n')
            with self.assertRaisesRegex(ValueError, 'best.pt'):
                freeze(config, Path(directory) / 'lock.json', 'bad', Path(directory) / 'out',
                       checkpoint=Path(directory) / 'last.pt')


if __name__ == '__main__':
    unittest.main()
