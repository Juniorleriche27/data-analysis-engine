from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

import pandas as pd


@dataclass(slots=True)
class Dataset:
    """
    Internal representation of a tabular dataset.

    Pandas is currently the first supported backend,
    but the rest of the engine should interact through
    this object rather than depending directly on files.
    """

    data: pd.DataFrame
    name: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return int(len(self.data))

    @property
    def column_count(self) -> int:
        return int(len(self.data.columns))

    @property
    def columns(self) -> list[str]:
        return [str(column) for column in self.data.columns]

    def fingerprint(self) -> str:
        """
        Deterministic fingerprint based on schema and values.

        Useful later for reproducibility, caching and lineage.
        """
        hashed = pd.util.hash_pandas_object(
            self.data,
            index=True,
        ).values

        payload = (
            "|".join(self.columns).encode()
            + hashed.tobytes()
        )

        return sha256(payload).hexdigest()
