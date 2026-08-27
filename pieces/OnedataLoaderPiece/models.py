from pydantic import BaseModel, Field
from typing import Literal


class InputModel(BaseModel):
    operation: Literal["download", "upload"] = Field(
        default="download",
        description="'download' fetches data from OneData into local_path; 'upload' pushes local_path contents to OneData."
    )
    onezone_host: str = Field(
        default="data.spice-platform.eu",
        description="Hostname of the Onezone service."
    )
    access_token: str = Field(
        description="OneData access token. Use a token with opw-* service scope so it can reach Oneproviders."
    )
    space_name: str = Field(
        default="UC5Space",
        description="Name of the OneData space."
    )
    remote_path: str = Field(
        description="Path inside the space, e.g. 'uc4/histo_data/images'."
    )
    local_path: str = Field(
        description="Local directory to download into (download) or upload from (upload)."
    )


class OutputModel(BaseModel):
    local_path: str = Field(
        description="Local directory where data was written (download) or read from (upload)."
    )
    remote_path: str = Field(
        description="Full OneData path used: <space_name>/<remote_path>."
    )
    num_files: int = Field(
        description="Number of files transferred."
    )
    total_size_mb: float = Field(
        description="Total transferred size in MB."
    )
