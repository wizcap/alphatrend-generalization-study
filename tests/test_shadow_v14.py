from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sklearn', reason='Study014 uses the pinned supervised runtime')
from neattrade import shadow_data_v14 as d, shadow_v14 as s
from neattrade.market import Market, simulate
from neattrade.adaptive_v13 import utc
from neattrade.indicators_v11 import COLUMNS


def bars(start='2026-09-01', n=192):
    ix = pd.date_range(utc(start), periods=n, freq='15min')
    rows = [[t.value//1_000_000,100,102,99,101,10,t.value//1_000_000+899999,1000,5,4,400,0] for t in ix]
    return rows


def canonical(start='2026-09-01', n=192):
    rows = bars(start, n)
    b = d.candles(rows, start, utc(start)+n*d.STEP, utc(start)+(n+1)*d.STEP)
    b['mark'], b['funding'] = 100., 0.
    return b


def market(n=96):
    ix = pd.date_range(utc('2026-08-31'), periods=n+1, freq='15min')
    prices = 100+np.arange(n+1)*.05+np.sin(np.arange(n+1)/7)
    x = np.zeros((n, len(COLUMNS)))
    x[:,0], x[:,-1] = 1., .4
    rates = np.where(np.arange(n+1)%32 == 0, .0001, 0.)
    return Market('BTCUSDT', ix, x, prices, prices*1.001, rates, prices[:-1]-2, prices[:-1]+2)


@pytest.mark.parametrize('mutation', ['gap','duplicate','reverse','open','badclose','nan','negative','ohlc','volume'])
def test_candle_validation_rejects_corruption(mutation):
    rows = bars()
    if mutation == 'gap': rows.pop(7)
    if mutation == 'duplicate': rows[7] = rows[6].copy()
    if mutation == 'reverse': rows.reverse()
    if mutation == 'open': rows[-1][6] = utc('2026-09-04').value//1_000_000
    if mutation == 'badclose': rows[1][6] -= 1
    if mutation == 'nan': rows[1][2] = float('nan')
    if mutation == 'negative': rows[1][1] = -1
    if mutation == 'ohlc': rows[1][2] = 80
    if mutation == 'volume': rows[1][9] = 11
    with pytest.raises(ValueError):
        d.candles(rows, '2026-09-01', '2026-09-03', '2026-09-03 00:15')


def test_funding_requires_explicit_zero_events():
    rows = [{'symbol':'BTCUSDT','fundingTime':t.value//1_000_000,'fundingRate':'0'}
            for t in pd.date_range(utc('2026-09-01'), periods=6, freq='8h')]
    rates = d.funding(rows, 'BTCUSDT', '2026-09-01', '2026-09-03')
    assert len(rates) == 6 and (rates == 0).all()
    for invalid in [rows[:-1], rows[:2]+rows[3:], rows+[rows[-1]], list(reversed(rows))]:
        with pytest.raises(ValueError):
            d.funding(invalid, 'BTCUSDT', '2026-09-01', '2026-09-03')


def test_funding_jitter_symbol_and_interval():
    rows = [{'symbol':'BTCUSDT','fundingTime':t.value//1_000_000+1,'fundingRate':'.0001'}
            for t in pd.date_range(utc('2026-09-01'), periods=3, freq='8h')]
    assert len(d.funding(rows, 'BTCUSDT', '2026-09-01', '2026-09-02')) == 3
    for field, value in [('symbol','ETHUSDT'),('fundingTime',rows[1]['fundingTime']+900_000),('fundingRate','nan')]:
        bad = deepcopy(rows); bad[1][field] = value
        with pytest.raises(ValueError): d.funding(bad, 'BTCUSDT', '2026-09-01', '2026-09-02')


def test_history_overlap_and_no_revision():
    a, b = canonical(), canonical('2026-09-02')
    c = d.merge_history(a, b)
    pd.testing.assert_frame_equal(a, c.iloc[:len(a)])
    assert len(c) == 288
    changed = b.astype({'open':float}); changed.iloc[3, changed.columns.get_loc('open')] += .001
    with pytest.raises(ValueError, match='revised'): d.merge_history(a, changed)
    with pytest.raises(ValueError): d.merge_history(a, b.iloc[1:])
    with pytest.raises(ValueError): d.merge_history(a, b.drop(columns='mark'))


@pytest.mark.parametrize('cost', [7.,14.])
def test_open_account_matches_independent_terminal_simulator(cost):
    m = market(192)
    p = np.where(np.arange(192) < 10, 0., np.where(np.arange(192)<120, .5, .25))
    allowed = np.arange(192)%8 == 1
    got = s.account(m,p,allowed,cost)
    reference = simulate(m,p,cost,0.,trade_allowed=allowed,rebalance_band=.05)
    np.testing.assert_array_equal(got['equity'][:-1], reference['equity'][:-1])
    np.testing.assert_allclose(got['liquidation_equity'],reference['equity'][-1],atol=1e-13)
    assert got['units'][-1] != 0 and got['equity'][-1] > got['liquidation_equity']
    assert got['funding'][-1] == reference['funding_paid_initial_equity']


def test_no_refresh_close_and_exact_account_prefix():
    m = market(192)
    p, allowed = np.full(192,.5), np.arange(192)%8 == 1
    entire = s.account(m,p,allowed,7.)
    shorter = s.cut(m,0,80)
    partial = s.account(shorter,p[:80],allowed[:80],7.)
    for key in ['equity','units','fees','funding','orders']:
        np.testing.assert_array_equal(partial[key], entire[key][:81])
    assert partial['units'][-1] != 0


def model(date, value):
    return {'features':list(range(18)),'kind':'mean','mean':value,'available':str(utc(date))}


def ctx(n=208):
    m = market(n)
    # A natural entry four hours before the month boundary, then a checkpoint
    # in September. Existing management must remain August's version.
    m.x[:80,0] = -1
    return {'market':m,'atr':np.ones(n),'closes':m.prices[:-1]}


def test_version_latches_across_month_and_prefix():
    c = ctx()
    models = {'4h_expanding_202608':model('2026-08-01',1), '4h_expanding_202609':model('2026-09-01',-1)}
    p, allowed, actions = s.targets(c,'4h','monthly',models,{})
    exits = [a for a in actions if a['head']=='exit']
    assert exits[0]['version'] == '4h_expanding_202608' and not exits[0]['blocked']
    short = {'market':s.cut(c['market'],0,192),'atr':c['atr'][:192],'closes':c['closes'][:192]}
    p2,a2,events = s.targets(short,'4h','monthly',models,{})
    np.testing.assert_array_equal(p2,p[:192]); np.testing.assert_array_equal(a2,allowed[:192])
    assert events == actions


def test_checkpoint_does_not_create_resolved_terminal_label():
    c = ctx()
    labels, _ = s.labelled_events(c,'4h')
    assert len(labels) == 2 and all(r['known'] is None and r['y'] is None for r in labels)


def test_catalog_only_allows_natural_forward_resolution():
    before = {'4h':[{'id':'a','known':None,'head':'exit','decision':'2026-09-01','left_censored':False,'x':[1.], 'y':None}]}
    after = deepcopy(before); after['4h'][0].update(known='2026-09-03',y=.01)
    s.assert_catalog_prefix(before,after)
    changed = deepcopy(after); changed['4h'][0]['x'] = [2.]
    with pytest.raises(AssertionError): s.assert_catalog_prefix(before,changed)
    with pytest.raises(ValueError): s.assert_catalog_prefix(before,{'4h':[]})
    changed = deepcopy(after); changed['4h'][0]['y'] = -.1
    with pytest.raises(AssertionError): s.assert_catalog_prefix(after,changed)


def test_journal_never_upgrades_a_backfilled_entry_to_timely():
    _,_,actions = s.targets(ctx(),'4h','alpha',{}, {})
    initial = s.journal(actions[:1], [], {}, '2026-09-01')
    assert initial[0]['evidence_class'] == 'delayed_replay'
    checkpoint = actions[-1]
    recorded = utc(checkpoint['execution'])-pd.Timedelta(seconds=1)
    result = s.journal(actions, initial, {}, recorded)
    assert result[0] == initial[0] and result[-1]['evidence_class'] == 'delayed_replay'
    with pytest.raises(ValueError): s.journal([], initial, {}, recorded)


def test_timely_entry_requires_actual_model_creation_before_entry_execution():
    models = {'4h_expanding_202608':model('2026-08-01',1)}
    _,_,actions = s.targets(ctx(),'4h','monthly',models,{})
    recorded = utc(actions[0]['execution'])-pd.Timedelta(seconds=1)
    good = s.journal(actions[:1],[],{'4h_expanding_202608':'2026-08-01'},recorded)
    assert good[0]['evidence_class'] == 'timely_shadow'
    bad = s.journal(actions[:1],[],{'4h_expanding_202608':'2026-09-01'},recorded)
    assert bad[0]['evidence_class'] == 'delayed_replay'


def test_journal_idempotency_and_mutation_rejection():
    _,_,actions = s.targets(ctx(),'4h','alpha',{}, {})
    saved = s.journal(actions,[],{},'2026-09-04')
    assert s.journal(actions,saved,{},'2026-09-05') == saved
    changed = deepcopy(actions); changed[-1]['blocked'] = True
    with pytest.raises(ValueError): s.journal(changed,saved,{},'2026-09-05')


def test_lock_is_exclusive_and_released_on_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with s.writer(tmp_path):
            with pytest.raises(FileExistsError):
                with s.writer(tmp_path): pass
            raise RuntimeError('injected')
    with s.writer(tmp_path): pass


def test_os_lock_recovers_after_process_death(tmp_path):
    import subprocess,sys
    code = 'import os,sys; from neattrade.shadow_v14 import writer; ctx=writer(sys.argv[1]); ctx.__enter__(); os._exit(0)'
    subprocess.run([sys.executable,'-c',code,str(tmp_path)],check=True)
    with s.writer(tmp_path): pass


def test_bundle_commit_integrity_and_no_overwrite(tmp_path):
    staging = tmp_path/'runs'/'pending'
    staging.mkdir(parents=True); (staging/'a.json').write_text('{}')
    pointer = s.commit(tmp_path,staging,'20260901',None)
    assert s.read_json(tmp_path/'current.json') == pointer
    s.verify_bundle(tmp_path/'runs'/'20260901',pointer['bundle_sha256'])
    (tmp_path/'runs'/'20260901'/'a.json').write_text('{"changed":true}')
    with pytest.raises(ValueError): s.verify_bundle(tmp_path/'runs'/'20260901')
    with pytest.raises(ValueError): s.commit(tmp_path,staging,'20260901',pointer)


def test_pointer_not_advanced_when_bundle_creation_fails(tmp_path,monkeypatch):
    staging = tmp_path/'runs'/'pending'; staging.mkdir(parents=True)
    (staging/'a.json').write_text('{}')
    def fail(*args,**kwargs): raise OSError('injected disk failure')
    monkeypatch.setattr(Path,'rename',fail)
    with pytest.raises(OSError): s.commit(tmp_path,staging,'20260901',None)
    assert not (tmp_path/'current.json').exists()


def test_crash_after_bundle_can_recover_before_a_later_acquisition(tmp_path,monkeypatch):
    staging = tmp_path/'runs'/'pending'; staging.mkdir(parents=True)
    name = '20260901T040000'
    s.write_json(staging/'status.json',{'protocol_sha256':'p','run':name})
    replace = Path.replace
    def fail_pointer(self,target):
        if self.name == 'current.tmp': raise OSError('crash before pointer')
        return replace(self,target)
    monkeypatch.setattr(Path,'replace',fail_pointer)
    with pytest.raises(OSError): s.commit(tmp_path,staging,name,None)
    assert (tmp_path/'runs'/name).exists() and not (tmp_path/'current.json').exists()
    monkeypatch.setattr(Path,'replace',replace)
    pointer = s.recover_committed_bundle(tmp_path,None,'p')
    assert pointer['run'] == name
    assert s.recover_committed_bundle(tmp_path,pointer,'p') == pointer


def test_receipt_hash_detects_changes(tmp_path):
    (tmp_path/'000.json').write_text('[]')
    d.write_json(tmp_path/'manifest.json',{'responses':[{'file':'000.json','sha256':s.hash_file(tmp_path/'000.json')}]})
    d.verify_receipts(tmp_path)
    (tmp_path/'000.json').write_text('[1]')
    with pytest.raises(ValueError): d.verify_receipts(tmp_path)


def test_independent_continued_controller_agrees_on_promotion():
    from neattrade.verify_v14 import controller_step
    from neattrade.verify_v13 import equal
    models = {f'4h_weighted_2026{month}':model(f'2026-{month}-01',-.01) for month in ['08','09']}
    c = s.old.Controller('4h','test')
    c.step('2026-08-01',[],models)
    state = deepcopy(c.state)
    rows = []
    for i in range(20):
        t = utc('2026-08-02')+pd.Timedelta(hours=i*12)
        base = .03 if i%4 == 0 else -.02
        rows.append({'id':str(i),'symbol':['BTCUSDT','ETHUSDT'][i%2],'entry':str(t),
            'checkpoint':str(t+pd.Timedelta(days=1)),'known':str(t+pd.Timedelta(days=2)),
            'left_censored':False,'x':[0.]*18,
            'base':{'base':base,'double_cost':base-.001},
            'early':{'base':max(base,0.),'double_cost':max(base,0.)-.001}})
    ref = controller_step(state,'4h','2026-09-01',rows,models)
    actual = c.step('2026-09-01',rows,models)
    equal(ref,actual)
    assert 'promote' in actual['actions']
