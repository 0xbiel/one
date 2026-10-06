import json
import pytest
from app.context_budget import bound_context_packet, serialized_size


def packet():
    return {'scope': {'home_id': 'home', 'care_recipient_id': 'bobby'},
            'window': {'start_at': '2026-09-25T00:00:00Z', 'end_at': '2026-10-01T12:00:00Z'},
            'privacy_boundary': {'excludes': ['raw_frames', 'transcripts']},
            'missing_coverage': {'status': 'partial', 'limitations': ['No uptime evidence.'],
                                 'cameras': [{'camera_id': str(i), 'last_scoped_observation_at': None} for i in range(20)]},
            'observations': [{'source_id': str(i), 'observed_at': f'2026-09-{25+i%6:02}T10:00:00Z', 'text': '🙂'*600} for i in range(20)],
            'synthetic_provenance': [{'source_id': str(i), 'synthetic': True} for i in range(20)],
            'evidence_ids': [str(i) for i in range(20)]}


def test_unicode_budget_deterministic_newest_and_links():
    original = packet()
    bounded = bound_context_packet(original, max_bytes=5500)
    assert serialized_size(bounded) <= 5500
    assert bounded['budget']['serialized_bytes'] == len(json.dumps(bounded, ensure_ascii=False).encode('utf-8'))
    assert bounded == bound_context_packet(original, max_bytes=5500)
    assert bounded == bound_context_packet(bounded, max_bytes=5500)
    assert bounded['scope'] == original['scope']
    assert bounded['window'] == original['window']
    assert bounded['privacy_boundary'] == original['privacy_boundary']
    assert bounded['missing_coverage']['limitations'][:1] == original['missing_coverage']['limitations']
    ids = {row['source_id'] for row in bounded['observations']}
    assert set(bounded['evidence_ids']) == ids
    assert {row['source_id'] for row in bounded['synthetic_provenance']} == ids
    assert all(len(row['text'].encode('utf-8')) <= 1024 for row in bounded['observations'])
    stamps = [row['observed_at'] for row in bounded['observations']]
    assert stamps == sorted(stamps, reverse=True)
    assert len(original['observations']) == 20


def test_overlong_required_id_rejected_not_shortened():
    with pytest.raises(ValueError, match='Required context metadata'):
        bound_context_packet({'scope': {'home_id': 'x'*10000}}, max_bytes=2000)


def test_whole_provider_packet_caps_authored_text_preserves_plan_ids_and_time():
    data = {'subject': {'id': 'recipient'}, 'request': '界'*1500,
            'plans': [{'id': 'plan', 'instructions': '界'*1500}],
            'check_ins': [{'id': 'check', 'plan_id': 'plan', 'scheduled_for': '2026-09-30T10:00:00Z', 'note': '界'*1500}]}
    bounded = bound_context_packet(data)
    assert bounded['subject']['id'] == 'recipient'
    assert bounded['check_ins'][0]['scheduled_for'] == data['check_ins'][0]['scheduled_for']
    assert bounded['plans'][0]['id'] == 'plan'
    assert len(bounded['request'].encode('utf-8')) <= 1024
    assert bounded['budget']['truncated'] is True
    assert bounded == bound_context_packet(bounded)


def test_positive_budget_required():
    with pytest.raises(ValueError): bound_context_packet({}, max_bytes=0)


def test_nested_required_metadata_and_timestamp_citations_preserved():
    nested = packet()
    nested['evidence_timestamps'] = {row['source_id']: row['observed_at'] for row in nested['observations']}
    original = {'bounded_context': nested}
    bounded = bound_context_packet(original, max_bytes=5500)['bounded_context']
    assert bounded['scope'] == nested['scope']
    assert bounded['window'] == nested['window']
    assert bounded['privacy_boundary'] == nested['privacy_boundary']
    assert bounded['missing_coverage']['limitations'][:1] == nested['missing_coverage']['limitations']
    ids = {row['source_id'] for row in bounded['observations']}
    assert set(bounded['evidence_timestamps']) == ids


def test_unsupported_citation_ids_are_removed():
    bounded = bound_context_packet({'observations': [{'source_id': 'kept', 'observed_at': '2026-09-30T00:00:00Z'}],
                                    'evidence_ids': ['kept', 'absent'],
                                    'evidence_timestamps': {'kept': '2026-09-30T00:00:00Z', 'absent': '2026-09-29T00:00:00Z'}})
    assert bounded['evidence_ids'] == ['kept']
    assert set(bounded['evidence_timestamps']) == {'kept'}
