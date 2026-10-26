from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_META_BASE_URL = (
    "https://api.meta.ai/v1"
)

DEFAULT_META_MODEL = (
    "muse-spark-1.3"
)

DEFAULT_TIMEOUT_SECONDS = 60.0


class MetaModelAPIError(
    RuntimeError
):
    """Raised when Meta Model API communication fails."""


Transport = Callable[
    [
        str,
        dict[str, str],
        dict[str, Any],
        float,
    ],
    dict[str, Any],
]


def _http_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    body = json.dumps(
        payload
    ).encode(
        "utf-8"
    )

    request = Request(
        url=url,
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(
            request,
            timeout=timeout,
        ) as response:
            raw = response.read()

    except HTTPError as exc:
        raise MetaModelAPIError(
            "Meta Model API returned "
            f"HTTP {exc.code}."
        ) from exc

    except URLError as exc:
        raise MetaModelAPIError(
            "Unable to reach Meta Model API."
        ) from exc

    try:
        decoded = json.loads(
            raw.decode(
                "utf-8"
            )
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise MetaModelAPIError(
            "Meta Model API returned an "
            "invalid JSON response."
        ) from exc

    if not isinstance(
        decoded,
        dict,
    ):
        raise MetaModelAPIError(
            "Meta Model API response must "
            "be a JSON object."
        )

    return decoded


class MetaModelAPIClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_META_BASE_URL,
        model: str = DEFAULT_META_MODEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: Transport | None = None,
    ) -> None:
        if (
            not isinstance(
                base_url,
                str,
            )
            or not base_url.strip()
        ):
            raise MetaModelAPIError(
                "'base_url' must be a non-empty string."
            )

        if (
            not isinstance(
                model,
                str,
            )
            or not model.strip()
        ):
            raise MetaModelAPIError(
                "'model' must be a non-empty string."
            )

        if timeout <= 0:
            raise MetaModelAPIError(
                "'timeout' must be positive."
            )

        self._api_key = api_key

        self.base_url = (
            base_url.rstrip("/")
        )

        self.model = (
            model.strip()
        )

        self.timeout = float(
            timeout
        )

        self.transport = (
            transport
            or _http_transport
        )

    def _resolve_api_key(
        self,
    ) -> str:
        key = (
            self._api_key
            or os.environ.get(
                "MODEL_API_KEY"
            )
        )

        if (
            not isinstance(
                key,
                str,
            )
            or not key.strip()
        ):
            raise MetaModelAPIError(
                "MODEL_API_KEY is not configured."
            )

        return key.strip()

    def configuration(
        self,
    ) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "api_key_configured": bool(
                self._api_key
                or os.environ.get(
                    "MODEL_API_KEY"
                )
            ),
        }

    def create_chat_completion(
        self,
        *,
        messages: Sequence[
            Mapping[str, Any]
        ],
        response_format: (
            Mapping[str, Any]
            | None
        ) = None,
    ) -> dict[str, Any]:
        if (
            not isinstance(
                messages,
                Sequence,
            )
            or isinstance(
                messages,
                (
                    str,
                    bytes,
                ),
            )
            or not messages
        ):
            raise MetaModelAPIError(
                "'messages' must contain at least "
                "one message."
            )

        normalized_messages = []

        for message in messages:
            if not isinstance(
                message,
                Mapping,
            ):
                raise MetaModelAPIError(
                    "Each message must be an object."
                )

            role = message.get(
                "role"
            )

            content = message.get(
                "content"
            )

            if (
                not isinstance(
                    role,
                    str,
                )
                or not role.strip()
            ):
                raise MetaModelAPIError(
                    "Each message requires a role."
                )

            if (
                content is not None
                and not isinstance(
                    content,
                    str,
                )
            ):
                raise MetaModelAPIError(
                    "Text planner messages require "
                    "string content."
                )

            normalized_messages.append(
                dict(
                    message
                )
            )

        payload: dict[
            str,
            Any,
        ] = {
            "model": self.model,
            "messages": normalized_messages,
        }

        if response_format is not None:
            if not isinstance(
                response_format,
                Mapping,
            ):
                raise MetaModelAPIError(
                    "'response_format' must "
                    "be an object."
                )

            payload[
                "response_format"
            ] = dict(
                response_format
            )

        api_key = (
            self._resolve_api_key()
        )

        headers = {
            "Authorization": (
                f"Bearer {api_key}"
            ),
            "Content-Type": (
                "application/json"
            ),
        }

        endpoint = (
            f"{self.base_url}"
            "/chat/completions"
        )

        response = self.transport(
            endpoint,
            headers,
            payload,
            self.timeout,
        )

        if not isinstance(
            response,
            dict,
        ):
            raise MetaModelAPIError(
                "Meta transport must return "
                "a JSON object."
            )

        return response

    @staticmethod
    def extract_message_content(
        response: Mapping[str, Any],
    ) -> str:
        try:
            choices = response[
                "choices"
            ]

            first_choice = choices[
                0
            ]

            message = first_choice[
                "message"
            ]

            content = message[
                "content"
            ]

        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise MetaModelAPIError(
                "Meta Model API response does not "
                "contain assistant message content."
            ) from exc

        if not isinstance(
            content,
            str,
        ):
            raise MetaModelAPIError(
                "Assistant message content must "
                "be a string."
            )

        return content

    def structured_completion(
        self,
        *,
        messages: Sequence[
            Mapping[str, Any]
        ],
        schema_name: str,
        schema: Mapping[
            str,
            Any,
        ],
    ) -> dict[str, Any]:
        if (
            not isinstance(
                schema_name,
                str,
            )
            or not schema_name.strip()
        ):
            raise MetaModelAPIError(
                "'schema_name' must be "
                "a non-empty string."
            )

        if not isinstance(
            schema,
            Mapping,
        ):
            raise MetaModelAPIError(
                "'schema' must be an object."
            )

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": (
                    schema_name.strip()
                ),
                "schema": dict(
                    schema
                ),
            },
        }

        response = (
            self.create_chat_completion(
                messages=messages,
                response_format=(
                    response_format
                ),
            )
        )

        content = (
            self.extract_message_content(
                response
            )
        )

        try:
            parsed = json.loads(
                content
            )
        except json.JSONDecodeError as exc:
            raise MetaModelAPIError(
                "Structured Meta response "
                "was not valid JSON."
            ) from exc

        if not isinstance(
            parsed,
            dict,
        ):
            raise MetaModelAPIError(
                "Structured Meta response must "
                "decode to a JSON object."
            )

        return parsed
