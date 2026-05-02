from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, RootModel, field_validator

FileType = Literal["td", "th", "tr", "tp"]


class ECVSchema(RootModel[dict[FileType, dict[str, list[str]]]]):
    @field_validator("root")
    @classmethod
    def validate_schema(cls, value: dict[str, dict[str, list[str]]]):
        required_sections = {"td", "th", "tr", "tp"}
        missing = required_sections - set(value)

        if missing:
            raise ValueError(f"Missing schema sections: {sorted(missing)}")

        for section, variables in value.items():
            if not isinstance(variables, dict):
                raise TypeError(f"Section {section} must be a dictionary")

            for clean_name, aliases in variables.items():
                if not aliases:
                    raise ValueError(
                        f"{section}.{clean_name} must have at least one candidate column"
                    )

                if not all(isinstance(alias, str) for alias in aliases):
                    raise TypeError(
                        f"{section}.{clean_name} aliases must all be strings"
                    )

        return value


def load_ecv_schema(
    path: str | Path = "ecv_schema.yml",
) -> dict[str, dict[str, list[str]]]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    schema = ECVSchema.model_validate(raw)
    return schema.root
