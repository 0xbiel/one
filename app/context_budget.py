"""Deterministic UTF-8 JSON budgets for metadata/provider context packets."""
from copy import deepcopy
from datetime import datetime, timezone
import json


def serialized_size(packet):
    """Match the adapter's ordinary json.dumps serialization exactly."""
    return len(json.dumps(packet, ensure_ascii=False).encode('utf-8'))


def _identity_or_time(key):
    return (key in {'id', 'ids', 'date', 'timezone', 'timezone_name', 'scheduled_for', 'evidence_ids'}
            or key.endswith(('_id', '_ids', '_at')))


def _timestamp(row):
    if not isinstance(row, dict):
        return datetime.min.replace(tzinfo=timezone.utc)
    for key in ('observed_at', 'occurred_at', 'recorded_at', 'scheduled_for', 'updated_at', 'created_at'):
        try:
            stamp = datetime.fromisoformat(str(row.get(key, '')).replace('Z', '+00:00'))
            return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)
        except ValueError:
            continue
    return datetime.min.replace(tzinfo=timezone.utc)


def _stable(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def bound_context_packet(context, max_bytes=24576, *, max_string_bytes=1024):
    """Copy a packet, cap authored strings, then omit whole rows to fit.

    Scope, window, privacy policy, and coverage limitations are required. Camera
    lists inside coverage may be reduced with explicit omissions. Identity and
    timestamp strings are never shortened. Raises ValueError if the required
    packet cannot fit; this function does not claim a tokenizer-specific limit.
    """
    max_bytes, max_string_bytes = int(max_bytes), int(max_string_bytes)
    if max_bytes < 1 or max_string_bytes < 1:
        raise ValueError('Context budgets must be positive')
    packet = deepcopy(context)
    previous = packet.pop('budget', {})
    omitted = deepcopy(previous.get('omitted_counts', {}))
    shortened = deepcopy(previous.get('string_truncations', {}))
    already_truncated = bool(previous.get('truncated'))
    candidates = []
    required = {'scope', 'window', 'privacy_boundary'}

    def walk(value, path='', protected=False, key=''):
        in_coverage = path.startswith('missing_coverage.') or '.missing_coverage.' in path
        if isinstance(value, dict):
            for name in sorted(value):
                child = f'{path}.{name}' if path else name
                locked = protected or name in required
                value[name] = walk(value[name], child, locked, name)
        elif isinstance(value, list):
            # Evidence rows can be dropped; required policy/coverage arrays stay.
            removable = not protected and (not in_coverage or key == 'cameras')
            if removable and value and all(isinstance(row, dict) for row in value):
                value.sort(key=lambda row: (_timestamp(row), _stable(row)), reverse=True)
                if key != 'synthetic_provenance':
                    candidates.append((path, value))
            for index, child in enumerate(value):
                value[index] = walk(child, f'{path}[{index}]', protected or (in_coverage and key != 'cameras'), key)
        elif isinstance(value, str) and not protected and not in_coverage and not _identity_or_time(key):
            data = value.encode('utf-8')
            if len(data) > max_string_bytes:
                shortened[path] = len(data) - len(data[:max_string_bytes].decode('utf-8', errors='ignore').encode('utf-8'))
                value = data[:max_string_bytes].decode('utf-8', errors='ignore')
        return value

    packet = walk(packet)
    def gather_ids(value, found):
        if isinstance(value, dict):
            for key in ('id', 'source_id'):
                if isinstance(value.get(key), str):
                    found.add(value[key])
            for key, child in value.items():
                if key != 'synthetic_provenance':
                    gather_ids(child, found)
        elif isinstance(value, list):
            for child in value:
                gather_ids(child, found)

    def prune_links(value, retained, path=''):
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f'{path}.{key}' if path else key
                if key == 'synthetic_provenance' and isinstance(child, list):
                    kept = [row for row in child if row.get('source_id') in retained]
                    removed = len(child) - len(kept)
                    if removed:
                        omitted[child_path] = omitted.get(child_path, 0) + removed
                        child[:] = kept
                elif key == 'evidence_timestamps' and isinstance(child, dict):
                    for source_id in list(child):
                        if source_id not in retained:
                            del child[source_id]
                elif key == 'evidence_ids' and isinstance(child, list):
                    child[:] = [id for id in child if id in retained]
                else:
                    prune_links(child, retained, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                prune_links(child, retained, f'{path}[{index}]')

    packet['budget'] = {'max_bytes': max_bytes, 'serialized_bytes': 0,
                        'omitted_counts': omitted, 'string_truncations': shortened,
                        'truncated': already_truncated or bool(shortened) or bool(omitted)}

    def mark_coverage(value, path=''):
        if isinstance(value, dict):
            coverage = value.get('missing_coverage')
            if isinstance(coverage, dict):
                prefix = path + '.' if path else ''
                removed_observations = omitted.get(prefix + 'observations', 0)
                removed_here = sum(count for name, count in omitted.items() if name.startswith(prefix) and name != prefix + 'synthetic_provenance')
                if removed_here:
                    coverage['truncated'] = True
                    if removed_observations:
                        coverage['observation_records_truncated'] = True
                    message = 'Coverage describes the pre-budget record snapshot; retrieved evidence rows may be omitted by the JSON byte budget.'
                    limitations = coverage.get('limitations')
                    if isinstance(limitations, list) and message not in limitations:
                        limitations.append(message)
            for name, child in value.items():
                if name != 'budget':
                    mark_coverage(child, f'{path}.{name}' if path else name)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                mark_coverage(child, f'{path}[{index}]')

    def size():
        mark_coverage(packet)
        # The byte-count field itself is included; converge its digit width.
        for _ in range(8):
            count = serialized_size(packet)
            if packet['budget']['serialized_bytes'] == count:
                return count
            packet['budget']['serialized_bytes'] = count
        raise ValueError('Context size metadata did not converge')

    retained = set(); gather_ids(packet, retained); prune_links(packet, retained)
    while size() > max_bytes:
        available = [(path, rows) for path, rows in candidates if rows]
        if not available:
            raise ValueError('Required context metadata exceeds JSON byte budget')
        # Drop the oldest row globally; path and canonical JSON resolve ties.
        path, rows = min(available, key=lambda pair: (_timestamp(pair[1][-1]), pair[0], _stable(pair[1][-1])))
        rows.pop()
        omitted[path] = omitted.get(path, 0) + 1
        packet['budget']['truncated'] = True
        retained = set(); gather_ids(packet, retained); prune_links(packet, retained)
    size()
    return packet
