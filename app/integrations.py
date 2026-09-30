import json
import time
import urllib.error
import urllib.request
import base64
import hmac
import hashlib
from datetime import datetime, timezone

from .config import Settings


class LMStudioAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_error: str | None = None

    def summarize(self, context: dict) -> dict | None:
        if not self.settings.llm_enabled:
            self.last_error = "disabled"
            return None
        payload = {
            "model": self.settings.effective_llm_model,
            "temperature": 0.1,
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": "You are ONE's cautious household assistant. Never diagnose or infer medical conditions. Return JSON with status (stable|attention|unknown), trend (stable|improving|changing|unknown), explanation, evidence_ids, limitations."},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "check_in_summary",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["stable", "attention", "unknown"]},
                        "trend": {"type": "string", "enum": ["stable", "improving", "changing", "unknown"]},
                        "explanation": {"type": "string"},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "limitations": {"type": "string"},
                    },
                    "required": ["status", "trend", "explanation", "evidence_ids", "limitations"],
                    "additionalProperties": False,
                },
            }},
        }
        if self.settings.llm_provider in (None, "lmstudio", "lm_studio"):
            payload["reasoning_effort"] = "none"
        headers = {"Content-Type": "application/json"}
        if self.settings.effective_llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.effective_llm_api_key}"
        request = urllib.request.Request(self.settings.effective_llm_base_url + "/chat/completions", data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.settings.llm_timeout_seconds) as response:
                result = json.loads(response.read())
            content = result["choices"][0]["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else content
            if parsed.get("status") not in {"stable", "attention", "unknown"}: return None
            parsed.setdefault("trend", "unknown"); parsed.setdefault("evidence_ids", []); parsed.setdefault("limitations", "Observations are approximate and are not medical advice.")
            self.last_error = None
            return parsed
        except urllib.error.HTTPError as exc:
            self.last_error = f"http_{exc.code}"
            return None
        except TimeoutError:
            self.last_error = "timeout"
            return None
        except (OSError, urllib.error.URLError):
            self.last_error = "connection_error"
            return None
        except (KeyError, IndexError, ValueError, json.JSONDecodeError):
            self.last_error = "invalid_model_response"
            return None

    def family_summary(self, context: dict) -> dict | None:
        """Answer a bounded family-mode request using only supplied records.

        Family mode intentionally has a separate method and prompt boundary so
        a future caller cannot accidentally hand the model the complete event
        or camera stream. The API constructs ``context`` from medication
        records, daily check-ins, and bounded fall-safety analytics only and
        validates the small response shape here.
        """
        if not self.settings.llm_enabled:
            self.last_error = "disabled"
            return None
        payload = {
            "model": self.settings.effective_llm_model,
            "temperature": 0.1,
            "max_tokens": 220,
            "messages": [
                {"role": "system", "content": "You are ONE's cautious administrative family organizer. Use only the supplied medication records, daily check-in summaries, and bounded fall-safety analytics. Do not diagnose, infer a medical condition, recommend medicine changes, or provide emergency advice. Describe fall values as review signals, not confirmed falls. Return JSON with summary, next_action, evidence_ids, limitations. Keep it concise and say when records are missing."},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "family_summary",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "next_action": {"type": "string"},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "limitations": {"type": "string"},
                    },
                    "required": ["summary", "next_action", "evidence_ids", "limitations"],
                    "additionalProperties": False,
                },
            }},
        }
        if self.settings.llm_provider in (None, "lmstudio", "lm_studio"):
            payload["reasoning_effort"] = "none"
        headers = {"Content-Type": "application/json"}
        if self.settings.effective_llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.effective_llm_api_key}"
        request = urllib.request.Request(self.settings.effective_llm_base_url + "/chat/completions", data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.settings.llm_timeout_seconds) as response:
                result = json.loads(response.read())
            content = result["choices"][0]["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else content
            if not isinstance(parsed, dict) or not isinstance(parsed.get("summary"), str) or not isinstance(parsed.get("next_action"), str):
                self.last_error = "invalid_model_response"
                return None
            evidence_ids = parsed.get("evidence_ids", [])
            if not isinstance(evidence_ids, list) or not all(isinstance(item, str) for item in evidence_ids):
                self.last_error = "invalid_model_response"
                return None
            parsed["evidence_ids"] = evidence_ids[:20]
            parsed.setdefault("limitations", "Records are incomplete and this administrative summary is not medical advice.")
            self.last_error = None
            return parsed
        except urllib.error.HTTPError as exc:
            self.last_error = f"http_{exc.code}"
            return None
        except TimeoutError:
            self.last_error = "timeout"
            return None
        except (OSError, urllib.error.URLError):
            self.last_error = "connection_error"
            return None
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            self.last_error = "invalid_model_response"
            return None


def livekit_jwt(api_key: str, api_secret: str, identity: str, room: str, can_publish: bool, can_subscribe: bool) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"iss": api_key, "sub": identity, "nbf": now - 5, "exp": now + 600, "video": {"room": room, "roomJoin": True, "canPublish": can_publish, "canSubscribe": can_subscribe}}
    def enc(value): return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).rstrip(b"=").decode()
    unsigned = enc(header) + "." + enc(payload)
    signature = base64.urlsafe_b64encode(hmac.new(api_secret.encode(), unsigned.encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
    return unsigned + "." + signature


def verify_livekit_webhook(auth_header: str | None, body: bytes, api_key: str, api_secret: str) -> dict:
    """Verify the standard LiveKit webhook JWT and optional body digest claim.

    This mirrors the official SDK's trust boundary without requiring the optional
    livekit-server-sdk package. Deployments should still prefer that SDK when it is
    available and should pin it alongside the LiveKit server version.
    """
    if not auth_header or not auth_header.startswith("Bearer "):
        raise ValueError("missing webhook bearer token")
    parts = auth_header[7:].split(".")
    if len(parts) != 3:
        raise ValueError("malformed webhook token")
    def decode(part):
        return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    header, claims = decode(parts[0]), decode(parts[1])
    if header.get("alg") != "HS256": raise ValueError("unsupported webhook algorithm")
    unsigned = f"{parts[0]}.{parts[1]}".encode()
    expected = base64.urlsafe_b64encode(hmac.new(api_secret.encode(), unsigned, hashlib.sha256).digest()).rstrip(b"=").decode()
    if not hmac.compare_digest(expected, parts[2]) or claims.get("iss") != api_key: raise ValueError("invalid webhook signature")
    if claims.get("exp") is not None and int(claims["exp"]) < int(datetime.now(timezone.utc).timestamp()): raise ValueError("expired webhook token")
    if claims.get("sha256") and not hmac.compare_digest(str(claims["sha256"]), hashlib.sha256(body).hexdigest()): raise ValueError("webhook body digest mismatch")
    return claims
