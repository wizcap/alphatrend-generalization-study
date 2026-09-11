"""Allowlist, provenance, credential scan and SHA-256 public release manifest."""
from hashlib import sha256
import json
import os
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
root_names={'README.md','LICENSE','NOTICE.md','CITATION.cff','pytest.ini','config-feedforward',
            '.gitignore','.gitattributes','PUBLIC_VALIDATION.json','release-notes.md'}
files=[]
for p in ROOT.iterdir():
    if p.is_file() and (p.name in root_names or p.name.startswith('requirements') and p.suffix=='.txt'): files.append(p)
for folder in ['neattrade','tests','tools']:
    files+=list((ROOT/folder).glob('*.py'))
files+=list((ROOT/'paper').glob('*.md'))+list((ROOT/'paper').glob('*.pdf'))+list((ROOT/'paper/figures').glob('*.png'))
files+=list((ROOT/'references/alphatrend').glob('*.pine'))+list((ROOT/'references/alphatrend').glob('*.json'))
files+=[p for p in (ROOT/'evidence/study015').rglob('*') if p.is_file() and p.suffix in {'.json','.pkl','.npz','.csv','.txt','.md'}]
patterns=[re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
          re.compile(rb'(?:ghp_|gho_|github_pat_|sk-ant-api)[A-Za-z0-9_\-]{18,}'),
          re.compile(rb'C:(?:\\\\|\\|/)Users(?:\\\\|\\|/)[A-Za-z0-9_.-]+',re.I)]
secrets={k:v.encode() for k,v in os.environ.items() if re.search('API.?KEY|TOKEN|SECRET|PASSWORD',k,re.I) and len(v)>=12}
findings=[]; hashes={}
for p in sorted(set(files)):
    assert not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve())
    raw=p.read_bytes(); name=p.relative_to(ROOT).as_posix()
    assert len(raw)<95*1024**2,('GitHub file-size limit',name)
    for pattern in patterns:
        if pattern.search(raw): findings.append((name,'credential-or-machine-path-pattern'))
    for k,v in secrets.items():
        if v in raw: findings.append((name,'environment-secret:'+k))
    hashes[name]=sha256(raw).hexdigest()
assert not findings,findings  # Never prints matching values.
protocol=json.loads((ROOT/'evidence/study015/protocol.json').read_text())
matched=0
for path,h in protocol['source_hashes'].items():
    canonical=path.replace('\\','/')
    dest=ROOT/canonical
    if canonical=='reports/study015_method.md': dest=ROOT/'evidence/study015/study015_method.md'
    if dest.is_file():
        assert sha256(dest.read_bytes()).hexdigest()==h,canonical
        matched+=1
assert matched==len(protocol['source_hashes']),('Missing frozen source',matched,len(protocol['source_hashes']))
assert sum(p.suffix=='.pkl' for p in set(files))==768
manifest={'schema':1,'release':'v1.0.0','files':hashes,'frozen_source_files_matched':matched,
          'credentials_or_machine_paths_detected':0,'curation':'No operational state, credentials, environments or raw exchange archives.'}
(ROOT/'PUBLIC_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'allowlisted_files':len(hashes),'bytes':sum(p.stat().st_size for p in set(files)),
                  'frozen_source_files_matched':matched,'credential_matches':0}))
