"""Mock-only provider gates: no credentials or provider connections."""
import json
import urllib.error
import pytest
from app.config import Settings
from app.integrations import LMStudioAdapter, _NoProviderRedirect


def adapter(**overrides):
    return LMStudioAdapter(Settings(_env_file=None, **overrides))


def external(**overrides):
    return adapter(**({'llm_provider':'cerebras','llm_base_url':'https://api.cerebras.ai/v1','llm_model':'gpt-oss-120b','llm_api_key':'test-dummy-not-a-credential', **overrides}))


@pytest.mark.parametrize('method',['summarize','family_summary'])
@pytest.mark.parametrize('gate,expected', [('off','external_data_disabled'),('missing','external_consent_required'),('string','external_consent_required')])
def test_external_requires_explicit_configuration_and_verified_consent(monkeypatch, method, gate, expected):
    def forbidden(*a,**kw): raise AssertionError('Network must not be called')
    monkeypatch.setattr(LMStudioAdapter,'_open_request',forbidden)
    instance=external(llm_external_data_enabled=gate!='off')
    context={'privacy_boundary':{'external_provider_authorized':'true'}} if gate=='string' else {}
    assert getattr(instance,method)(context) is None
    assert instance.last_error==expected


@pytest.mark.parametrize('url',['https://outside.example/v1','http://127.0.0.1.evil.example/v1','http://user:pass@127.0.0.1:1234/v1','http://127.0.0.1:1234/v1?key=hidden','http://localhost.example/v1','http://127.0.0.1:bad/v1'])
def test_local_provider_rejects_nonlocal_or_unsafe_urls(url):
    instance=adapter(llm_base_url=url)
    assert instance.summarize({}) is None
    assert instance.last_error=='invalid_provider_url'


@pytest.mark.parametrize('url',['http://127.0.0.1:1234/v1','http://[::1]:1234/v1','http://host.docker.internal:1234/v1'])
def test_local_provider_supports_loopback_and_docker_host(url):
    assert adapter(llm_base_url=url).capability_status()['configured'] is True


def test_unknown_provider_and_unsafe_external_configuration_fail_closed():
    assert adapter(llm_provider='unknown').capability_status()['reason']=='unsupported_provider'
    assert external(llm_external_data_enabled=True,llm_base_url='https://evil.example/v1').capability_status()['reason']=='invalid_provider_url'
    assert external(llm_external_data_enabled=True,llm_base_url='http://api.cerebras.ai/v1').capability_status()['reason']=='invalid_provider_url'
    assert external(llm_external_data_enabled=True,llm_model=None).capability_status()['reason']=='external_model_required'


def test_capability_metadata_never_contains_secret_or_url():
    info=external(llm_base_url='https://username:password@example.com/v1?token=secret').capability_status()
    rendered=json.dumps(info)
    assert not any(value in rendered for value in ['test-dummy','password','username','token=','example.com'])
    assert info['live_connection_verified'] is False


@pytest.mark.parametrize('method',['summarize','family_summary'])
def test_authorized_external_request_preserves_boundary_with_mock_transport(monkeypatch,method):
    seen=[]
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self): return json.dumps({'choices':[{'message':{'content':json.dumps({'status':'unknown','summary':'Unknown','next_action':'Review','evidence_ids':[]})}}]}).encode()
    def fake(self,request): seen.append(request); return Response()
    monkeypatch.setattr(LMStudioAdapter,'_open_request',fake)
    instance=external(llm_external_data_enabled=True)
    context={'privacy_boundary':{'external_provider_authorized':True},'scope':{'home_id':'synthetic'},'window':{}}
    assert getattr(instance,method)(context) is not None
    body=json.loads(seen[0].data)
    assert seen[0].full_url=='https://api.cerebras.ai/v1/chat/completions'
    assert body['model']=='gpt-oss-120b' and 'reasoning_effort' not in body
    assert 'untrusted data' in body['messages'][0]['content']
    assert len(seen[0].data)<=32768


def test_redirect_is_blocked_without_following_target():
    with pytest.raises(urllib.error.HTTPError):
        _NoProviderRedirect().redirect_request(type('Req',(),{'full_url':'http://127.0.0.1/v1'})(),None,302,'Found',{},'https://outside.example')


def test_transport_disables_proxies_and_redirects(monkeypatch):
    seen=[]
    class Opener:
        def open(self,request,timeout): seen.append(timeout); return 'mock-response'
    def build(*handlers):
        assert handlers[0].proxies=={}
        assert isinstance(handlers[1],_NoProviderRedirect)
        return Opener()
    monkeypatch.setattr('urllib.request.build_opener',build)
    assert adapter()._open_request(object())=='mock-response'
    assert seen==[15.0]
