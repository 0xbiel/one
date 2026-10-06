"""Contract tests use synthetic keys and temporary stores; never contact providers."""
from dataclasses import FrozenInstanceError
import json
import os
import socket
from uuid import uuid4

import pytest

from app.config import Settings
from app import provider_settings as providers

DUMMY = 'dummy-test-key-not-a-credential'


@pytest.fixture(autouse=True)
def no_provider_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Provider settings must never perform network I/O')
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.delenv('ONE_PROVIDER_CONFIG_DIR', raising=False)


@pytest.fixture
def private(tmp_path):
    directory = tmp_path / 'private'
    directory.mkdir(mode=0o700)
    return directory


def body(**changes):
    return dict(provider='cerebras', base_url='https://api.cerebras.ai/v1', model='unavailable-test-model', api_key=DUMMY, **changes)


def write_file(private, **changes):
    data = body()
    data.update(schema_version=1, home_id='test-home', revision=str(uuid4()), enabled=True)
    data.update(changes)
    target = private / providers.FILENAME
    target.write_text(json.dumps(data))
    target.chmod(0o600)
    return target


def read(private, home='test-home'):
    return providers.load_provider_snapshot(private, home, owner_uid=os.getuid())


def test_read_is_scoped_immutable_secret_free_and_does_not_probe_model(private):
    write_file(private)
    snapshot = read(private)
    assert snapshot.api_key == DUMMY
    assert snapshot.capability_status()['configured'] is True
    assert snapshot.capability_status()['live_connection_verified'] is False
    assert DUMMY not in repr(snapshot)
    assert 'api.cerebras.ai' not in json.dumps(snapshot.capability_status())
    assert 'unavailable-test-model' not in repr(snapshot)
    with pytest.raises(FrozenInstanceError):
        snapshot.api_key = 'changed'
    with pytest.raises(providers.ProviderConfigurationError):
        read(private, 'other-home')


@pytest.mark.parametrize('changes', [dict(schema_version=True), dict(home_id='other'), dict(revision='invalid'), dict(extra=DUMMY), dict(model='bad\nmodel'), dict(api_key='bad\nkey'), dict(enabled=1)])
def test_invalid_files_fail_without_secret_echo(private, changes):
    write_file(private, **changes)
    with pytest.raises(providers.ProviderConfigurationError) as error:
        read(private)
    assert DUMMY not in str(error.value)


@pytest.mark.parametrize('url', [
    'https://127.0.0.1/v1', 'https://10.0.0.1/v1', 'https://169.254.169.254/v1',
    'https://[::1]/v1', 'https://localhost/v1', 'https://host.docker.internal/v1',
    'https://127.1/v1', 'https://2130706433/v1', 'https://api.example.com:8080/v1',
    'https://user:secret@api.example.com/v1', 'https://api.example.com/v1?key=secret',
    'https://api.example.com/v1#secret', 'http://api.example.com/v1',
])
def test_generic_ssrf_urls_fail_closed(url):
    with pytest.raises(providers.ProviderConfigurationError):
        providers.validate_config(dict(provider='openai_compatible', base_url=url, model='model'))


def test_generic_public_and_local_explicit_endpoints_are_accepted_without_dns():
    assert providers.validate_config(dict(provider='openai_compatible', base_url='https://inference.example.com/v1/', model='org/model:version'))['base_url'] == 'https://inference.example.com/v1'
    assert providers.validate_config(dict(provider='lmstudio', base_url='http://127.0.0.1:1234/v1', model='local-model'))['provider'] == 'lmstudio'


def test_private_permissions_links_and_size_required(private):
    target = write_file(private)
    target.chmod(0o644)
    with pytest.raises(providers.ProviderConfigurationError): read(private)
    target.chmod(0o600)
    private.chmod(0o755)
    with pytest.raises(providers.ProviderConfigurationError): read(private)
    private.chmod(0o700)
    target.unlink()
    target.symlink_to(private/'missing')
    with pytest.raises(providers.ProviderConfigurationError): read(private)
    target.unlink()
    target = write_file(private)
    os.link(target, private/'copy')
    with pytest.raises(providers.ProviderConfigurationError): read(private)
    (private/'copy').unlink()
    target.write_text('x' * (providers.MAX_FILE + 1))
    with pytest.raises(providers.ProviderConfigurationError): read(private)


def test_atomic_revisions_key_retention_and_clearing(private):
    store = providers.ProviderStore(private, 'test-home')
    first = store.save(body())
    second = store.save({k:v for k,v in body().items() if k != 'api_key'})
    assert first['revision'] != second['revision']
    assert read(private).api_key == DUMMY
    assert store.save({k:v for k,v in body().items() if k != 'api_key'} | {'clear_api_key':True})['key_configured'] is False
    store.save(body())
    switched = store.save(dict(provider='openai_compatible', base_url='https://inference.example.com/v1', model='model'))
    assert switched['key_configured'] is False
    assert not list(private.glob('.provider-*'))


def test_missing_and_corrupt_config_fail_closed_even_with_legacy_settings(private, monkeypatch):
    settings = Settings(_env_file=None, lm_studio_url='http://127.0.0.1:1234/v1', llm_api_key=None, lm_studio_api_key=None)
    resolution = providers.resolve_provider_settings(settings, 'test-home', private)
    # Owner mismatch also fails closed; use reader stand-in to exercise absence.
    monkeypatch.setattr(providers, 'load_provider_snapshot', lambda *_args, **_kwargs: None)
    resolution = providers.resolve_provider_settings(settings, 'test-home', private)
    assert resolution.metadata['reason'] == 'unconfigured'
    assert resolution.settings.llm_enabled is False
    assert settings.llm_enabled is True
    def corrupt(*args, **kwargs):
        raise providers.ProviderConfigurationError('sanitized')
    monkeypatch.setattr(providers, 'load_provider_snapshot', corrupt)
    assert providers.resolve_provider_settings(settings, 'test-home', private).metadata['reason'] == 'invalid_private_configuration'


def test_resolver_snapshots_secret_and_preserves_external_master_gate(private, monkeypatch):
    write_file(private)
    original = providers.load_provider_snapshot
    monkeypatch.setattr(providers, 'load_provider_snapshot', lambda directory, home, **_kwargs: original(directory, home, owner_uid=os.getuid()))
    settings = Settings(_env_file=None, llm_external_data_enabled=False, llm_api_key=None, lm_studio_api_key=None)
    result = providers.resolve_provider_settings(settings, 'test-home', private)
    assert result.metadata['configured'] is True
    assert result.settings.llm_api_key == DUMMY
    assert result.settings.llm_external_data_enabled is False
    assert settings.llm_api_key is None
    assert DUMMY not in repr(result)
    write_file(private, api_key='replacement-dummy-key')
    assert result.settings.llm_api_key == DUMMY


def test_builtin_defaults_are_unconfigured_and_explicit_local_setup_is_configured():
    settings = Settings(_env_file=None, llm_api_key=None, lm_studio_api_key=None)
    assert providers.resolve_provider_settings(settings, 'test-home').metadata['reason'] == 'unconfigured'
    explicit = Settings(_env_file=None, lm_studio_url='http://127.0.0.1:9/v1', llm_api_key=None, lm_studio_api_key=None)
    assert providers.resolve_provider_settings(explicit, 'test-home').metadata['configured'] is True


def test_duplicates_and_wrong_owner_are_rejected(private):
    target = write_file(private)
    raw = target.read_text()
    target.write_text(raw[:-1] + ',"api_key":"second-dummy"}')
    with pytest.raises(providers.ProviderConfigurationError): read(private)
    write_file(private)
    with pytest.raises(providers.ProviderConfigurationError):
        providers.load_provider_snapshot(private, 'test-home', owner_uid=os.getuid()+1)


def test_explicit_owner_mapping_resolves_without_changing_permissions(private, monkeypatch):
    write_file(private)
    monkeypatch.setenv('ONE_PROVIDER_CONFIG_OWNER_UID', str(os.getuid()))
    monkeypatch.setenv('ONE_PROVIDER_CONFIG_DIRECTORY_OWNER_UID', str(os.getuid()))
    result = providers.resolve_provider_settings(Settings(_env_file=None), 'test-home', private)
    assert result.metadata['configured'] is True
    assert private.stat().st_mode & 0o777 == 0o700
    assert (private / providers.FILENAME).stat().st_mode & 0o777 == 0o600
    monkeypatch.setenv('ONE_PROVIDER_CONFIG_DIRECTORY_OWNER_UID', '-1')
    result = providers.resolve_provider_settings(Settings(_env_file=None), 'test-home', private)
    assert result.metadata['reason'] == 'invalid_private_configuration'
