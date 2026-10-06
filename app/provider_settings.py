"""Private provider snapshots. No network calls, credentials provisioning or consent writes."""
from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from urllib.parse import urlsplit
from uuid import UUID, uuid4

MAX_FILE = 8192
FILENAME = 'provider.json'
PROVIDERS = {'lmstudio', 'cerebras', 'openai_compatible'}


class ProviderConfigurationError(ValueError):
    """Sanitized error; never include configuration or secret values."""


def configured_owner_uid(name='ONE_PROVIDER_CONFIG_OWNER_UID', default=10001):
    value = os.environ.get(name, str(default))
    if not value.isascii() or not value.isdecimal() or len(value) > 10 or int(value) > 2147483647:
        raise ProviderConfigurationError('Invalid private store owner configuration')
    return int(value)


def validate_config(body: dict) -> dict:
    allowed = {'provider', 'base_url', 'model', 'api_key', 'clear_api_key', 'enabled'}
    if not isinstance(body, dict) or set(body) - allowed:
        raise ProviderConfigurationError('Invalid provider configuration')
    provider, url, model = body.get('provider'), body.get('base_url'), body.get('model')
    if not isinstance(provider, str) or provider not in PROVIDERS:
        raise ProviderConfigurationError('Unsupported provider')
    if not isinstance(url, str) or len(url) > 1024 or any(ord(c) < 33 or ord(c) > 126 for c in url):
        raise ProviderConfigurationError('Invalid endpoint')
    if not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,159}', model):
        raise ProviderConfigurationError('Invalid model identifier')
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise ProviderConfigurationError('Invalid endpoint') from None
    host = parsed.hostname
    if not host or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment or parsed.path.rstrip('/') != '/v1' or port == 0:
        raise ProviderConfigurationError('Invalid endpoint')
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if provider == 'lmstudio':
        if parsed.scheme not in {'http', 'https'} or not (host == 'host.docker.internal' or address and address.is_loopback):
            raise ProviderConfigurationError('Local endpoint required')
    else:
        # Runtime transport must additionally reject nonpublic DNS answers and
        # pin the validated address before connecting; validation never probes.
        if parsed.scheme != 'https' or port not in {None, 443}:
            raise ProviderConfigurationError('Public HTTPS endpoint required')
        if address is not None:
            if not address.is_global:
                raise ProviderConfigurationError('Public endpoint required')
        elif not re.fullmatch(r'(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{1,62}', host) or host.endswith(('.local', '.localhost', '.internal', '.test', '.invalid', '.example', '.home', '.lan')):
            raise ProviderConfigurationError('Public endpoint required')
        if provider == 'cerebras' and host != 'api.cerebras.ai':
            raise ProviderConfigurationError('Invalid Cerebras endpoint')
    key = body.get('api_key')
    if key is not None and (not isinstance(key, str) or not 1 <= len(key) <= 2048 or any(ord(c) < 33 or ord(c) > 126 for c in key)):
        raise ProviderConfigurationError('Invalid key')
    clear, enabled = body.get('clear_api_key', False), body.get('enabled', True)
    if type(clear) is not bool or type(enabled) is not bool or key and clear:
        raise ProviderConfigurationError('Invalid configuration operation')
    # Scheme/hostname case and the final slash are canonicalized for safe key reuse.
    hostname = '[' + host + ']' if ':' in host else host
    canonical = parsed.scheme + '://' + hostname + (':' + str(port) if port else '') + '/v1'
    return dict(provider=provider, base_url=canonical, model=model, api_key=key, clear_api_key=clear, enabled=enabled)


@dataclass(frozen=True)
class ProviderSnapshot:
    provider: str
    base_url: str = field(repr=False)
    model: str = field(repr=False)
    api_key: str | None = field(repr=False)
    home_id: str = field(repr=False)
    revision: str
    enabled: bool = True

    def capability_status(self) -> dict:
        external = self.provider != 'lmstudio'
        reason = 'disabled' if not self.enabled else 'external_key_required' if external and not self.api_key else None
        return dict(provider=self.provider, enabled=self.enabled, configured=reason is None,
                    reason=reason, external=external, locality='external' if external else 'local',
                    household_consent_required=external, live_connection_verified=False,
                    configuration_saved=True, key_configured=bool(self.api_key), revision=self.revision)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProviderConfigurationError('Invalid configuration file')
        result[key] = value
    return result


def _open_directory(directory: Path, owner_uid: int) -> int:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    info = os.fstat(descriptor)
    if info.st_uid != owner_uid or stat.S_IMODE(info.st_mode) != 0o700:
        os.close(descriptor)
        raise ProviderConfigurationError('Private store permissions required')
    return descriptor


def load_provider_snapshot(directory: Path, home_id: str, *, owner_uid: int = 10001, directory_owner_uid: int | None = None) -> ProviderSnapshot | None:
    """Read a bounded, validated, household-scoped snapshot through one file handle."""
    if not isinstance(home_id, str) or not home_id:
        raise ProviderConfigurationError('Household scope required')
    try:
        directory_fd = _open_directory(Path(directory), owner_uid if directory_owner_uid is None else directory_owner_uid)
        try:
            try:
                descriptor = os.open(FILENAME, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
            except FileNotFoundError:
                return None
            with os.fdopen(descriptor, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_uid != owner_uid or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > MAX_FILE or info.st_nlink != 1:
                    raise ProviderConfigurationError('Invalid private configuration file')
                raw = stream.read(MAX_FILE + 1)
        finally:
            os.close(directory_fd)
        if len(raw) > MAX_FILE:
            raise ProviderConfigurationError('Invalid configuration file')
        data = json.loads(raw, object_pairs_hook=_unique_object)
        fields = {'schema_version', 'home_id', 'revision', 'provider', 'base_url', 'model', 'api_key', 'enabled'}
        if not isinstance(data, dict) or set(data) != fields or type(data['schema_version']) is not int or data['schema_version'] != 1 or data['home_id'] != home_id or str(UUID(data['revision'])) != data['revision']:
            raise ProviderConfigurationError('Invalid configuration scope or revision')
        config = validate_config({key:data[key] for key in ('provider', 'base_url', 'model', 'api_key', 'enabled')})
        return ProviderSnapshot(**{key:config[key] for key in ('provider', 'base_url', 'model', 'api_key', 'enabled')}, home_id=home_id, revision=data['revision'])
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, AttributeError):
        raise ProviderConfigurationError('Private provider configuration unavailable') from None


class ProviderStore:
    """Single-writer atomic store; directory must be provisioned separately."""
    def __init__(self, directory: Path, home_id: str, *, owner_uid: int | None = None, directory_owner_uid: int | None = None):
        self.directory, self.home_id = Path(directory), home_id
        self.owner_uid = os.getuid() if owner_uid is None else owner_uid
        self.directory_owner_uid = self.owner_uid if directory_owner_uid is None else directory_owner_uid
        self.lock = threading.Lock()

    def status(self):
        with self.lock:
            snapshot = load_provider_snapshot(self.directory, self.home_id, owner_uid=self.owner_uid, directory_owner_uid=self.directory_owner_uid)
            return snapshot.capability_status() if snapshot else dict(provider=None, enabled=False, configured=False, reason='unconfigured', configuration_saved=False, key_configured=False, live_connection_verified=False)

    def save(self, body):
        body = validate_config(body)
        with self.lock:
            old = load_provider_snapshot(self.directory, self.home_id, owner_uid=self.owner_uid, directory_owner_uid=self.directory_owner_uid)
            key = body['api_key']
            if key is None and not body['clear_api_key'] and old and old.provider == body['provider'] and old.base_url == body['base_url']:
                key = old.api_key
            snapshot = ProviderSnapshot(**{k:body[k] for k in ('provider', 'base_url', 'model', 'enabled')}, api_key=None if body['clear_api_key'] else key, home_id=self.home_id, revision=str(uuid4()))
            data = dict(schema_version=1, home_id=self.home_id, revision=snapshot.revision, **{k:getattr(snapshot,k) for k in ('provider', 'base_url', 'model', 'api_key', 'enabled')})
            fd, temporary = tempfile.mkstemp(prefix='.provider-', dir=self.directory)
            try:
                if os.fstat(fd).st_uid != self.owner_uid:
                    os.close(fd)
                    raise ProviderConfigurationError('Private file owner mapping required')
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, 'w') as stream:
                    json.dump(data, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.directory / FILENAME)
                directory_fd = _open_directory(self.directory, self.directory_owner_uid)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return snapshot.capability_status()


@dataclass(frozen=True)
class ProviderResolution:
    settings: object = field(repr=False)
    metadata: dict
    snapshot: ProviderSnapshot | None = field(default=None, repr=False)


def resolve_provider_settings(settings, home_id: str, directory: Path | None = None) -> ProviderResolution:
    """Resolve once per request. No writes, provider probes, or consent changes."""
    configured_dir = directory if directory is not None else os.environ.get('ONE_PROVIDER_CONFIG_DIR')
    if configured_dir:
        try:
            snapshot = load_provider_snapshot(Path(configured_dir), home_id, owner_uid=configured_owner_uid(), directory_owner_uid=configured_owner_uid('ONE_PROVIDER_CONFIG_DIRECTORY_OWNER_UID', configured_owner_uid()))
            if snapshot:
                resolved = settings.model_copy(update=dict(llm_enabled=snapshot.enabled, llm_provider=snapshot.provider, llm_base_url=snapshot.base_url, llm_model=snapshot.model, llm_api_key=snapshot.api_key, lm_studio_api_key=None))
                return ProviderResolution(resolved, snapshot.capability_status(), snapshot)
            reason = 'unconfigured'
        except ProviderConfigurationError:
            reason = 'invalid_private_configuration'
        return ProviderResolution(settings.model_copy(update={'llm_enabled':False}), dict(configured=False, enabled=False, provider=None, external=False, locality='local', reason=reason, live_connection_verified=False))
    # Preserve explicit legacy/env setup; built-in LM Studio defaults alone do
    # not claim configuration. Consent/global external flags are separate gates.
    fields = getattr(settings, 'model_fields_set', set())
    relevant = fields & {'llm_provider', 'llm_base_url', 'llm_model', 'llm_api_key', 'lm_studio_url', 'lm_studio_model', 'lm_studio_api_key'}
    explicit = any(getattr(settings, name, None) not in {None, ''} for name in relevant)
    if not explicit:
        return ProviderResolution(settings.model_copy(update={'llm_enabled':False}), dict(configured=False, enabled=False, provider='lmstudio', external=False, locality='local', reason='unconfigured', live_connection_verified=False))
    try:
        provider = settings.llm_provider or 'lmstudio'
        if provider == 'lm_studio':
            provider = 'lmstudio'
        config = validate_config(dict(provider=provider, base_url=settings.effective_llm_base_url, model=settings.effective_llm_model, api_key=settings.effective_llm_api_key, enabled=settings.llm_enabled))
        snapshot = ProviderSnapshot(**{k:config[k] for k in ('provider','base_url','model','api_key','enabled')}, home_id=home_id, revision='legacy')
        metadata = snapshot.capability_status()
        metadata.pop('revision', None)
        metadata.pop('key_configured', None)
        metadata['configuration_saved'] = False
        return ProviderResolution(settings.model_copy(update=dict(llm_provider=provider, llm_base_url=config['base_url'])), metadata)
    except (ProviderConfigurationError, AttributeError):
        return ProviderResolution(settings.model_copy(update={'llm_enabled':False}), dict(configured=False, enabled=False, provider=None, external=False, locality='local', reason='invalid_provider_configuration', live_connection_verified=False))


def get_provider_settings(db, home_id: str, settings):
    return resolve_provider_settings(settings, home_id).settings
