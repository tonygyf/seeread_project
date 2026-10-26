import copy
import json
from pathlib import Path

import pytest

from seeread.v5.lated_annotations import import_reviews, labelled_rows, latin_text, write_jsonl
from seeread.v5.lated_evaluation import image_counts, evaluate_records, paired_comparison, reliable_training_matches
from seeread.v5.reading import sha256


def box(x0, x1):
    return [[x0, 0], [x1, 0], [x1, 10], [x0, 10]]


def test_voc_duplicate_does_not_match_second_best_unused_gt():
    gt = [dict(poly=box(0, 10)), dict(poly=box(2, 12))]
    pred = [dict(poly=box(0, 10), text=''), dict(poly=box(0, 10), text='')]
    counts, matches = image_counts(pred, gt)
    assert counts['tp_det'] == 1 and counts['n_pred'] == 2
    assert matches == [(0, 0)]


def test_latin_e2e_denominator_keeps_unmatched_but_excludes_other_languages():
    gt = [dict(id='en', poly=box(0, 10), text='ABC', ocr_eligible=True),
          dict(id='cn', poly=box(20, 30), text='', ocr_eligible=False)]
    pred = [dict(poly=box(0, 10), text='abc'), dict(poly=box(20, 30), text='bad'), dict(poly=box(40, 50), text='bad')]
    c, _ = image_counts(pred, gt)
    assert (c['tp_det'], c['n_pred'], c['tp_e2e'], c['n_pred_ocr'], c['n_gt_ocr']) == (2, 3, 1, 2, 1)


def test_exact_half_iou_is_not_a_voc_match():
    c, _ = image_counts([dict(poly=box(0, 20), text='')], [dict(poly=box(0, 10))])
    assert c.get('tp_det', 0) == 0
    empty, _ = image_counts([], [dict(poly=box(0, 10))])
    assert empty['n_gt'] == 1 and empty.get('tp_det', 0) == 0


def package(tmp_path):
    (tmp_path/'image.jpg').write_bytes(b'image')
    source = tmp_path / 'val.jsonl'
    rows = [dict(image='image.jpg', instances=[dict(id='one', poly=box(0, 10), text='', lang='en', ignore=False),
                                             dict(id='two', poly=box(20, 30), text='', lang='en', ignore=False)])]
    write_jsonl(source, rows)
    tasks = [dict(id=i, image='image.jpg', image_sha256=sha256(tmp_path/'image.jpg'), split='dev') for i in ('one', 'two')]
    write_jsonl(tmp_path/'tasks.jsonl', tasks)
    protocol = dict(schema='lated_transcription_v1', data_root=str(tmp_path), tasks_sha256=sha256(tmp_path/'tasks.jsonl'),
                    sources={'dev':dict(path=str(source),sha256=sha256(source))})
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    return source, sha256(source)


def test_pending_drafts_cannot_become_evaluation_truth_and_gt_is_unchanged(tmp_path):
    source, original = package(tmp_path)
    export = dict(protocol_sha256=sha256(tmp_path/'protocol.json'), reviewer='human', annotations=[
        dict(id='one', status='reviewed', language='latin', text='Hello'),
        dict(id='two', status='pending', language='unknown', text='OCR draft')])
    review = tmp_path/'review.json';review.write_text(json.dumps(export));sidecar=tmp_path/'sidecar.json'
    assert import_reviews(tmp_path, review, sidecar)['reviewed_instances'] == 1
    with pytest.raises(ValueError, match='incomplete'):
        labelled_rows(tmp_path, sidecar, 'dev')
    export['annotations'][1].update(status='reviewed', language='non_latin', text='')
    review.write_text(json.dumps(export));import_reviews(tmp_path, review, sidecar)
    rows=labelled_rows(tmp_path, sidecar, 'dev')
    assert [i['ocr_eligible'] for i in rows[0]['instances']] == [True, False]
    assert all(not i['ignore'] for i in rows[0]['instances'])
    assert sha256(source) == original
    export['protocol_sha256']='wrong';review.write_text(json.dumps(export))
    with pytest.raises(ValueError,match='identity'):
        import_reviews(tmp_path,review,sidecar)
    assert latin_text('Café 123') and not latin_text('Hello中文')
    assert not latin_text('!!!') and not latin_text('😀')
    export['protocol_sha256']=sha256(tmp_path/'protocol.json')
    export['annotations'][0]['status']='pending';review.write_text(json.dumps(export))
    import_reviews(tmp_path,review,sidecar)
    with pytest.raises(ValueError,match='incomplete'):
        labelled_rows(tmp_path,sidecar,'dev')


def test_failed_crops_stay_in_cer_and_paired_gain_is_image_bootstrapped():
    rows=[dict(image='a',instances=[dict(id='one',poly=box(0,10),text='ABC',ocr_eligible=True)])]
    records=[dict(image='a',predictions=[dict(poly=box(0,10),text='')],gt_crops=[dict(id='one',text='',crop_error='bad')])]
    reference=evaluate_records(rows,records)
    assert reference['gt_crop_raw']['cer']==1 and reference['matched_prediction_crop_raw']['cer']==1
    better=copy.deepcopy(records);better[0]['predictions'][0]['text']='ABC';better[0]['gt_crops'][0]['text']='ABC'
    candidate=evaluate_records(rows,better);paired=paired_comparison(reference,candidate)
    assert paired['e2e_F1_delta']==1 and paired['gt_CER_delta']==-1 and paired['gain_gate']
    records[0]['gt_crops']=[]
    with pytest.raises(ValueError,match='all eligible'):
        evaluate_records(rows,records)


def test_empty_normalized_reference_cannot_make_failed_crop_correct():
    c,_=image_counts([dict(poly=box(0,10),text='',crop_error='failed')],
                     [dict(poly=box(0,10),text='!!!',ocr_eligible=True)])
    assert c.get('tp_e2e',0)==0


def test_pilot_review_only_requires_frozen_pilot_subset(tmp_path):
    package(tmp_path)
    from seeread.v5.lated_annotations import read_jsonl
    tasks=read_jsonl(tmp_path/'tasks.jsonl')
    for task in tasks:task['pilot']=True
    write_jsonl(tmp_path/'tasks.jsonl',tasks)
    protocol=json.loads((tmp_path/'protocol.json').read_text())
    protocol['tasks_sha256']=sha256(tmp_path/'tasks.jsonl')
    # The second image is deliberately not part of the pilot.
    source=Path(protocol['sources']['dev']['path']);rows=read_jsonl(source)
    extra=copy.deepcopy(rows[0]);extra['image']='other.jpg'
    for instance in extra['instances']:instance['id']='other_'+instance['id']
    write_jsonl(source,rows+[extra]);protocol['sources']['dev']['sha256']=sha256(source)
    tasks.extend(dict(t,id='other_'+t['id'],image='other.jpg',pilot=False) for t in tasks[:2])
    (tmp_path/'other.jpg').write_bytes(b'image');write_jsonl(tmp_path/'tasks.jsonl',tasks)
    protocol['tasks_sha256']=sha256(tmp_path/'tasks.jsonl');(tmp_path/'protocol.json').write_text(json.dumps(protocol))
    sidecar=dict(schema='lated_transcription_sidecar_v1',protocol_sha256=sha256(tmp_path/'protocol.json'),annotations=[
        dict(id=t['id'],image=t['image'],split='dev',status='reviewed',language='latin',text='ABC',reviewer='human') for t in tasks if t['pilot']])
    path=tmp_path/'sidecar.json';path.write_text(json.dumps(sidecar))
    assert len(labelled_rows(tmp_path,path,'dev',pilot=True))==1
    with pytest.raises(ValueError,match='incomplete'):labelled_rows(tmp_path,path,'dev')


def test_training_crop_selection_rejects_partial_and_merged_instances():
    gt=[dict(poly=box(0,10),ocr_eligible=True),dict(poly=box(11,21),ocr_eligible=True)]
    pred=[dict(poly=box(0,21)),dict(poly=box(0,5)),dict(poly=box(0,10))]
    assert reliable_training_matches(pred,gt)==[(2,0)]
