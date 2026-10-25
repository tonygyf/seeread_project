"""Continue the fixed crop-adaptation run in auditable 2 -> 5 -> 10 epoch phases."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seeread.v5.reading import sha256
from tools.v5_adapt_reading import main as adapt
from tools.v5_eval_connector import run as evaluate_connector
from tools.v5_reading import write_json

RUN = Path('results/v5/fusion/read_connector_lora_20261025')
REFERENCE = Path('results/v5/fusion/connector_ctw250_adapted_20261025/run.json')


def extension_gate(reference, candidate):
    if reference['status'] != 'complete' or candidate['status'] != 'complete':
        raise ValueError('Extension needs complete paired development evaluations')
    if reference['data_hashes'] != candidate['data_hashes']:
        raise ValueError('Extension requires the same fixed development data')
    for name in ('seeread/v5/spotting.py', 'seeread/models/rectify.py', 'seeread/v5/seeing_decode.py'):
        if reference['source_hashes'][name] != candidate['source_hashes'][name]:
            raise ValueError('Extension requires the same connector and decoder implementation')
    for report in (reference, candidate):
        if report['config']['cases'] != ['boundary_when_supported'] or report['config']['enhancement'] != 'full':
            raise ValueError('Use the retained crop and full enhancer for the extension gate')
    a, b = [r['weights'][r['config']['weights']['detector']] for r in (reference, candidate)]
    if a != b or reference['config']['crop'] != candidate['config']['crop']:
        raise ValueError('Extension requires identical detector weights and cropping')
    def case(report, illumination):
        result = report['results'][illumination]
        if not result['complete']:
            raise ValueError('Partial development results cannot authorize extension')
        return result['cases']['boundary_when_supported']
    dark_a, dark_b = case(reference, 'dark'), case(candidate, 'dark')
    bright_a, bright_b = case(reference, 'bright'), case(candidate, 'bright')
    values = dict(dark_cer_improvement=dark_a['gt_crop_raw']['cer'] - dark_b['gt_crop_raw']['cer'],
                  dark_e2e_f_delta=dark_b['spotting']['e2e_F'] - dark_a['spotting']['e2e_F'],
                  bright_e2e_f_delta=bright_b['spotting']['e2e_F'] - bright_a['spotting']['e2e_F'])
    return dict(passed=values['dark_cer_improvement'] >= .01 and values['dark_e2e_f_delta'] >= 0
                and values['bright_e2e_f_delta'] >= -.01, **values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, choices=(5, 10), required=True)
    args = parser.parse_args()
    previous = 2 if args.epochs == 5 else 5
    phase_report = RUN / f'phase_epoch_{args.epochs:02d}.json'
    if phase_report.exists():
        raise FileExistsError(phase_report)
    report = dict(status='preflight', previous_epochs=previous, target_epochs=args.epochs,
                  source_sha256=sha256(__file__), reference=str(REFERENCE))
    if args.epochs == 10:
        candidate = Path('results/v5/fusion/connector_ctw250_read_epoch05_20261025/run.json')
        report['gate'] = extension_gate(json.loads(REFERENCE.read_text()), json.loads(candidate.read_text()))
        if not report['gate']['passed']:
            report['status'] = 'skipped_no_development_gain'
            write_json(phase_report, report)
            return
    current = json.loads((RUN / 'run.json').read_text())
    samples = current['manifests']['train']['samples']
    batch = current['config']['train']['batch_size']
    steps_per_epoch = (samples + batch - 1) // batch
    saved = torch.load(RUN / 'last.pt', map_location='cpu', weights_only=False)
    if (current['status'] != 'complete' or not saved['resume_safe']
            or saved['state']['global_step'] != previous * steps_per_epoch):
        raise ValueError('Finish the preceding complete phase before extending its healthy last')
    del saved
    snapshot = RUN / 'phases' / f'epoch_{previous:02d}'
    snapshot.mkdir(parents=True, exist_ok=False)
    for name in ('best.pt', 'last.pt'):
        os.link(RUN / name, snapshot / name)
    shutil.copyfile(RUN / 'run.json', snapshot / 'run.json')
    report.update(status='running', snapshot=str(snapshot),
                  snapshot_hashes={name: sha256(snapshot / name) for name in ('best.pt', 'last.pt')},
                  target_updates=args.epochs * steps_per_epoch,
                  recipe='same LoRA, frozen enhancer, batch, data, optimizer and LR; only update cap increases')
    write_json(phase_report, report)
    sys.argv = ['tools/v5_adapt_reading.py', '--output', str(RUN), '--resume', str(RUN / 'last.pt'),
                '--max-steps', str(report['target_updates']), '--max-seconds', '0']
    try:
        adapt()
        for name, detector in (('ctw250', 'CTW1500_dsconv_fp32/best.pt'), ('mlt100', 'MLT17_dsconv_fp32/last.pt')):
            output = Path(f'results/v5/fusion/connector_{name}_read_epoch{args.epochs:02d}_20261025')
            evaluate_connector(Path('results/v5/detection') / detector, RUN / 'best.pt', output,
                               samplings=['boundary_when_supported'], crop_pad=1.1)
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(phase_report, report)


if __name__ == '__main__':
    main()
