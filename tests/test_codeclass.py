"""클래스 → 제품코드 / 제품코드 마스크만 남기기 검증 (데이터셋 불필요). 실행: python3 -m pytest tests/test_codeclass.py -q"""
import os, sys, tempfile, shutil
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mask_reviewer import codeclass
from mask_reviewer.dataset import Dataset


def box(cls, x0, y0, x1, y1):
    return f'{cls} {x0} {y0} {x1} {y0} {x1} {y1} {x0} {y1}'


CENTER = (0.3, 0.3, 0.7, 0.7)     # 중심을 덮는 큰 마스크
INNER = (0.4, 0.4, 0.6, 0.6)      # 중심을 덮는 작은 마스크
CORNER = (0.05, 0.05, 0.2, 0.2)   # 구석


def test_pick_center():
    m = [np.zeros((100, 100), bool) for _ in range(4)]
    m[0][5:20, 5:20] = True
    m[1][40:60, 40:60] = True
    m[2][30:70, 30:70] = True
    assert codeclass.center_distance(m[3]) == float('inf') and codeclass.center_distance(m[1]) == 0.0
    assert codeclass.pick_center(m) == 2               # 중심을 덮는 것 중 넓은 것
    assert codeclass.pick_center(m, [0, 1]) == 1
    assert codeclass.pick_center(m, [0, 3]) == 0 and codeclass.pick_center(m, [3]) is None


def test_convert_text():
    text = '\n'.join([box(3, *CORNER), box(5, *INNER), box(7, *CENTER)]) + '\n'
    out = codeclass.convert_text(text, 9, 'to_code').splitlines()
    assert [l.split()[0] for l in out] == ['9', '9', '9']
    assert [l.split()[1:] for l in out] == [l.split()[1:] for l in text.splitlines()]   # 폴리곤은 그대로
    # 제품코드 클래스가 있으면 그 마스크만 (구석에 있어도)
    assert codeclass.convert_text(text, 3, 'keep_code') == box(3, *CORNER) + '\n'
    # 없으면 중앙 마스크를 제품코드 클래스로
    assert codeclass.convert_text(text, 9, 'keep_code') == box(9, *CENTER) + '\n'
    assert codeclass.convert_text(box(3, *CORNER) + '\n', 9, 'keep_code') == box(9, *CORNER) + '\n'
    assert codeclass.convert_text('', 9, 'keep_code') == ''
    # 최대 마스크만: 클래스는 그대로
    assert codeclass.convert_text(text, None, 'keep_largest') == box(7, *CENTER) + '\n'
    assert codeclass.convert_text('', None, 'keep_largest') == ''


def test_apply_bulk():
    root = tempfile.mkdtemp()
    try:
        for d in ('images', 'labels', 'labels_auto'):
            os.makedirs(os.path.join(root, d))
        codes = ['10T031000NT9', '10P152000NT9', '10K082000NT9']
        with open(os.path.join(root, 'dataset.yaml'), 'w') as f:
            f.write('names:\n' + ''.join(f'  {i}: {c}\n' for i, c in enumerate(codes[:2])))
        labels = {f'a_{codes[0]}': [box(1, *CENTER), box(0, *CORNER)],      # 제품코드 클래스가 구석에 있음
                  f'b_{codes[0]}': [box(1, *CENTER), box(1, *CORNER)],      # 없음 → 중앙
                  f'c_{codes[1]}': [box(1, *CENTER)],                        # 이미 맞음
                  f'd_{codes[2]}': [box(1, *CENTER)],                        # 클래스 목록에 없는 코드
                  f'e_{codes[0]}': []}
        for s, lines in labels.items():
            cv2.imwrite(os.path.join(root, 'images', s + '.png'), np.zeros((8, 8, 3), np.uint8))
            with open(os.path.join(root, 'labels_auto' if s[0] == 'a' else 'labels', s + '.txt'), 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
        ds = Dataset(root)
        a, b, c, d, e = ds.stems
        r = codeclass.apply_bulk(ds, ds.stems, 'keep_code')
        assert r['changed'] == [a, b] and r['same'] == 1 and r['empty'] == 1 and r['no_class'] == {codes[2]: 1}
        assert ds.label_text(a) == box(0, *CORNER) + '\n' and ds.label_text(b) == box(0, *CENTER) + '\n'
        assert ds.is_edited(a) and not ds.is_edited(c) and not ds.is_edited(d)
        assert open(ds.auto_label_path(a)).read().count('\n') == 2          # 자동·원본 라벨은 그대로
        assert codeclass.apply_bulk(ds, ds.stems, 'keep_code')['changed'] == []
        ds.revert(b)
        r = codeclass.apply_bulk(ds, [b], 'to_code')
        assert r['changed'] == [b] and ds.label_text(b) == box(0, *CENTER) + '\n' + box(0, *CORNER) + '\n'
        for s in (a, b):
            ds.revert(s)
        r = codeclass.apply_bulk(ds, ds.stems, 'keep_largest')               # 제품코드가 클래스 목록에 없어도 적용
        assert r['changed'] == [a, b] and r['same'] == 2 and r['empty'] == 1 and not r['no_class']
        assert ds.label_text(a) == box(1, *CENTER) + '\n' and ds.label_text(b) == box(1, *CENTER) + '\n'
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    test_pick_center(); test_convert_text(); test_apply_bulk()
    print('codeclass OK')
