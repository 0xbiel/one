"""Prepare the full frozen GMDCSA24 data locally; no model inference or scoring.

Only original MP4 files from the official repository's pinned commit are fetched.
Existing pilot files and manifests are never modified. Existing source clips are
verified, and missing clips are atomically written after Git blob verification.
"""
import concurrent.futures
import collections
import hashlib
import json
import pathlib
import re
import subprocess
import time
import urllib.parse
import urllib.request
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parent
BASE = ROOT / 'dataset'
REVISION = '5abac7693229900cf80f722e878fbb119211fc1c'
REPO = 'ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos'
TREE = json.loads((ROOT / 'dataset_tree.json').read_text())
assert TREE['sha'] == REVISION and not TREE['truncated']
ENTRIES = {entry['path']: entry for entry in TREE['tree'] if entry['type'] == 'blob'}

def verified_bytes(data, entry):
    git_sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    assert len(data) == entry['size'], (entry['path'], 'size mismatch')
    assert git_sha == entry['sha'], (entry['path'], 'Git blob hash mismatch')
    return git_sha


def build_manifest():
    # Retain source fields verbatim; only trim leading/trailing field whitespace.
    # maxsplit=5 keeps the Classes field intact in these unquoted source CSVs.
    metadata = {}
    for subject in range(1, 5):
        for label in ('ADL', 'Fall'):
            source_path = f'Subject {subject}/{label}.csv'
            data = (BASE / source_path).read_bytes()
            verified_bytes(data, ENTRIES[source_path])
            rows = data.decode('utf-8-sig').splitlines()
            for line_number, line in enumerate(rows[1:], 2):
                if not line.strip():
                    continue
                row = [field.strip() for field in line.split(',', 5)]
                assert len(row) == 6, (source_path, line_number, line)
                filename, duration, recording, attire, description, annotation = row
                path = f'Subject {subject}/{label}/{filename}'
                assert path not in metadata, ('duplicate metadata', path)
                match = re.search(r'Fall(?:ing)?[^;\[]*\[\s*(\d+(?:\.\d+)?)', annotation, re.IGNORECASE)
                onset = float(match.group(1)) if match else None
                metadata[path] = dict(
                    id=f's{subject}_{label.lower()}_{int(filename[:-4]):02}',
                    subject=subject,
                    subject_folder=subject,
                    split='development' if subject == 1 else 'evaluation',
                    label=label,
                    path=path,
                    description=description,
                    annotation=annotation,
                    fall_onset_s=onset,
                    fall_onset_missing=label == 'Fall' and onset is None,
                    source_reported_duration_s=float(duration),
                    time_of_recording=recording,
                    attire=attire,
                    metadata_source_path=source_path,
                    metadata_source_line=line_number,
                    metadata_source_git_blob_sha1=ENTRIES[source_path]['sha'],
                    source=f'https://github.com/{REPO}/blob/{REVISION}/' + urllib.parse.quote(path),
                )
    video_paths = sorted(path for path in ENTRIES if path.endswith('.mp4'))
    assert set(video_paths) == set(metadata), ('metadata/video mismatch', set(video_paths) ^ set(metadata))
    manifest = []
    for path in video_paths:
        item = metadata[path]
        item['expected_bytes'] = ENTRIES[path]['size']
        item['expected_git_blob_sha1'] = ENTRIES[path]['sha']
        manifest.append(item)
    counts = collections.Counter(item['label'] for item in manifest)
    assert counts == {'Fall': 79, 'ADL': 81} and len(manifest) == 160
    verified_bytes((BASE / 'LICENSE').read_bytes(), ENTRIES['LICENSE'])
    document = dict(
        dataset_revision=REVISION,
        source_repository=f'https://github.com/{REPO}',
        selection='All 160 original MP4 clips listed in the frozen official repository tree, metadata selected before full-dataset inference; no selection by results.',
        split_definition='Subject folder 1 development (16 Fall / 16 ADL); subject folders 2-4 evaluation (63 Fall / 65 ADL).',
        split_caveat='Subject-folder-held-out only: the paper reports three actors while the repository has four subject folders, so independent-person identity is unverified. The prior 32-clip pilot has already been run; this is an expanded retrospective evaluation, not a wholly untouched test set.',
        data_caveat='Consenting staged actors; staged indoor falls and ADL, not clinical or real-incident validation. Raw videos are retained locally and must not be published.',
        annotation_caveat='Preserve source CSV annotation text and the first fall onset in its own semicolon-delimited annotation segment. Missing fall onset is null and must be excluded from onset-latency measures, while clip classification remains eligible. Source annotations contain typos and some inconsistent duration/range fields. Duration and frame counts must use ffprobe, not CSV estimates.',
        frame_sampling=dict(fps=5, method='ffmpeg fps=5; select source frames, no interpolation', time_s='Nominal bins i/5, as in pilot; selected source frame may be near bin center.'),
        metadata_parsing='Split each non-empty source line on the first five commas; all eight checked source files contain exactly six fields per line. Store original annotation and source line.',
        counts=dict(total=160, Fall=79, ADL=81, development=32, evaluation=128),
        clips=manifest,
    )
    (ROOT / 'full_manifest_preinference.json').write_text(json.dumps(document, indent=2) + '\n')
    return manifest


def download(item):
    item = dict(item)
    path = item['path']
    target = BASE / path
    if target.exists():
        data = target.read_bytes()
        verified_bytes(data, ENTRIES[path])
        state = 'verified existing'
    else:
        url = f'https://raw.githubusercontent.com/{REPO}/{REVISION}/' + urllib.parse.quote(path)
        errors = []
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    data = response.read()
                verified_bytes(data, ENTRIES[path])
                break
            except Exception as exc:
                errors.append(repr(exc))
                if attempt == 2:
                    raise RuntimeError((path, errors)) from exc
                time.sleep(1 + attempt)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix('.mp4.partial')
        temp.write_bytes(data)
        temp.replace(target)
        state = 'downloaded and verified'
    item.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), git_blob_sha1=ENTRIES[path]['sha'])
    print(item['id'], state, item['bytes'], flush=True)
    return item


def prepare_one(item):
    item = dict(item)
    source = BASE / item['path']
    directory = ROOT / 'frames_full' / item['id']
    directory.mkdir(parents=True, exist_ok=True)
    video = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-threads', '2', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_frames,nb_read_frames',
        '-of', 'json', str(source),
    ]))['streams'][0]
    item['video'] = video
    item['native_fps'] = float(Fraction(video['avg_frame_rate']))
    item['declared_fps'] = float(Fraction(video['r_frame_rate']))
    item['native_frame_count'] = int(video['nb_read_frames'])
    item['supports_20hz_without_interpolation'] = item['native_fps'] >= 20
    # Isolate this extraction from all pilot frames; never interpolate frames.
    subprocess.run([
        'ffmpeg', '-v', 'error', '-nostdin', '-y', '-threads', '2', '-i', str(source),
        '-vf', 'fps=5', '-q:v', '2', '-threads', '2', str(directory / '%05d.jpg'),
    ], check=True)
    paths = sorted(directory.glob('*.jpg'))
    assert paths and [p.name for p in paths] == [f'{i:05d}.jpg' for i in range(1, len(paths) + 1)]
    item['sampled_frames'] = len(paths)
    frames = [dict(clip=item['id'], split=item['split'], time_s=i / 5, frame=str(path.relative_to(ROOT))) for i, path in enumerate(paths)]
    print(item['id'], 'prepared', len(paths), '5Hz /', item['native_frame_count'], 'native /', item['native_fps'], 'fps', flush=True)
    return item, frames


def main():
    manifest = build_manifest()
    print('PRE-INFERENCE MANIFEST SAVED:', len(manifest), 'clips', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        downloaded = list(pool.map(download, manifest))
    (ROOT / 'full_manifest_downloaded.json').write_text(json.dumps(downloaded, indent=2) + '\n')
    print('ALL DOWNLOADS VERIFIED:', len(downloaded), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        prepared = list(pool.map(prepare_one, downloaded))
    clips = [item for item, _ in prepared]
    frames = [frame for _, clip_frames in prepared for frame in clip_frames]
    (ROOT / 'full_manifest_downloaded.json').write_text(json.dumps(clips, indent=2) + '\n')
    (ROOT / 'full_frames.json').write_text(json.dumps(frames, indent=2) + '\n')
    summary = dict(
        dataset_revision=REVISION,
        clips=len(clips),
        by_label=dict(collections.Counter(item['label'] for item in clips)),
        by_split_label={split: dict(collections.Counter(item['label'] for item in clips if item['split'] == split)) for split in ('development', 'evaluation')},
        video_bytes=sum(item['bytes'] for item in clips),
        all_git_blob_sha1_verified=True,
        min_native_fps=min(item['native_fps'] for item in clips),
        max_native_fps=max(item['native_fps'] for item in clips),
        min_declared_fps=min(item['declared_fps'] for item in clips),
        max_declared_fps=max(item['declared_fps'] for item in clips),
        below_20fps_count=sum(not item['supports_20hz_without_interpolation'] for item in clips),
        below_20fps_clips=[dict(id=item['id'], split=item['split'], label=item['label'], avg_frame_rate=item['video']['avg_frame_rate'], r_frame_rate=item['video']['r_frame_rate'], native_fps=item['native_fps']) for item in clips if not item['supports_20hz_without_interpolation']],
        missing_fall_onset_clips=[item['id'] for item in clips if item.get('fall_onset_missing')],
        all_support_20hz_without_interpolation=all(item['supports_20hz_without_interpolation'] for item in clips),
        total_native_frames=sum(item['native_frame_count'] for item in clips),
        total_5hz_frames=len(frames),
        total_duration_s=sum(float(item['video']['duration']) for item in clips),
        by_split_frames={split: dict(native=sum(item['native_frame_count'] for item in clips if item['split'] == split), sampled_5hz=sum(item['sampled_frames'] for item in clips if item['split'] == split)) for split in ('development', 'evaluation')},
        native_frame_count_mismatches=[item['id'] for item in clips if item['video'].get('nb_frames') not in (None, 'N/A', str(item['native_frame_count']))],
        no_inference_or_scoring_performed=True,
    )
    (ROOT / 'full_dataset_preparation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
