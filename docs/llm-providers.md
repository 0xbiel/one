# Local and external summary providers

Default behavior retains LM Studio at `http://127.0.0.1:1234/v1` with
`qwen3.6-35b-a3b`. For Docker on Mac, configure `ONE_LLM_BASE_URL` as
`http://host.docker.internal:1234/v1`. Local URLs accept loopback IPs or the
Docker host name only. Redirects and environment HTTP proxies are disabled.

Explicit supported external choices are `cerebras`, `openrouter`, and
`openai_compatible`. Unknown provider names fail closed. Cerebras requires
`https://api.cerebras.ai/v1`; OpenRouter requires `https://openrouter.ai/api/v1`.
Generic compatible providers require an explicit HTTPS endpoint. Credentials,
userinfo, query strings and fragments cannot appear in endpoint URLs.

External processing is default off. A request requires all of: explicit provider,
endpoint, model, in-memory API key, `ONE_LLM_EXTERNAL_DATA_ENABLED=true`, and
`context.privacy_boundary.external_provider_authorized` equal to boolean true.
The API must derive this last flag from verified household/recipient consent;
never accept it as an unchecked user-submitted field. Key/environment presence
alone is not authorization. No consent grants or live calls were made in this work.
Use `capability_status()` for read-only configuration metadata; it does not
contact providers, expose URLs or keys, or claim a working connection.

Cerebras endpoint and compatible request shape are documented in the
[official OpenAI compatibility guide](https://inference-docs.cerebras.ai/resources/openai).
`gpt-oss-120b` is an example listed in the
[official model reference](https://inference-docs.cerebras.ai/api-reference/models/public-models),
not a fixed default or guarantee of current account availability. Keep model choice
explicit. Structured-output support can vary by provider/model; failures produce
no generated summary and retain the deterministic application fallback.

Only bounded text packets are sent: the existing untrusted-record system boundary,
24 KiB context target and actual 32 KiB encoded request limit remain enforced.
Tests use mock transport and dummy credential strings exclusively. Runtime wiring
and verified consent derivation are required before external processing can run.
