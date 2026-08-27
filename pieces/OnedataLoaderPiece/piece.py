from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import os
import time


def _patch_provider_version():
    """
    Cloud-SK and cloud-pl Oneproviders report version=None in the Onezone API.
    onedatafilerestclient calls packaging.version.parse(None) which raises TypeError,
    causing NoAvailableProviderForSpaceError before any transfer attempt.
    Substitute a high version so the version check passes.
    """
    try:
        import onedatafilerestclient.provider_selector as _ps
        from packaging.version import parse as _parse
        from onedatafilerestclient.provider_selector import SpaceSupportingProvider as _SSP

        @staticmethod
        def _safe_build(provider_id, support_attributes, provider_details):
            raw = provider_details.get("version") or "99.0.0"
            return _SSP(
                id=provider_id,
                version=_parse(raw),
                domain=provider_details["domain"],
                online=provider_details["online"],
                readonly_support=support_attributes["readonly"],
            )

        _ps.ProviderSelector._build_space_supporting_provider = _safe_build
    except Exception:
        pass  # if the library version ever fixes this, the patch is harmless to skip


def _human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


class OnedataLoaderPiece(BasePiece):
    """
    Transfers datasets between OneData federated storage and local workflow storage.

    download: OneData space/remote_path  →  local_path
    upload:   local_path                 →  OneData space/remote_path
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        _patch_provider_version()

        try:
            from onedatarestfsspec import OnedataFileSystem
        except ImportError:
            raise RuntimeError(
                "onedatarestfsspec is not installed. Add it to requirements.txt."
            )

        import warnings
        warnings.filterwarnings("ignore", message="Unverified HTTPS")

        op          = input_data.operation
        host        = input_data.onezone_host
        token       = input_data.access_token
        space       = input_data.space_name
        remote_path = input_data.remote_path.strip("/")
        local_path  = input_data.local_path
        full_remote = f"{space}/{remote_path}"

        self.logger.info("=" * 60)
        self.logger.info(f"OnedataLoaderPiece — {op.upper()}")
        self.logger.info(f"  Onezone : {host}")
        self.logger.info(f"  Space   : {space}")
        self.logger.info(f"  Remote  : {full_remote}")
        self.logger.info(f"  Local   : {local_path}")
        self.logger.info("=" * 60)

        fs = OnedataFileSystem(
            onezone_host=host,
            token=token,
            verify_ssl=False,
            preferred_providers=[],
        )

        if op == "download":
            num_files, total_bytes = self._download(fs, full_remote, local_path)
        else:
            num_files, total_bytes = self._upload(fs, local_path, full_remote)

        total_mb = total_bytes / (1024 ** 2)
        self.logger.info(
            f"Done: {num_files} files, {_human(total_bytes)} "
            f"({'downloaded' if op == 'download' else 'uploaded'})"
        )

        return OutputModel(
            local_path=local_path,
            remote_path=full_remote,
            num_files=num_files,
            total_size_mb=round(total_mb, 2),
        )

    # ── download ──────────────────────────────────────────────────────────────

    def _download(self, fs, remote_dir: str, local_dir: str) -> tuple[int, int]:
        os.makedirs(local_dir, exist_ok=True)

        try:
            entries = fs.ls(remote_dir, detail=False)
        except Exception as e:
            raise RuntimeError(f"Cannot list remote path '{remote_dir}': {e}")

        # Collect all files recursively
        all_files = []
        self._collect_remote_files(fs, remote_dir, all_files)

        if not all_files:
            self.logger.warning(f"No files found at remote path: {remote_dir}")
            return 0, 0

        self.logger.info(f"Downloading {len(all_files)} files...")
        total_bytes = 0

        for i, remote_file in enumerate(all_files, 1):
            # Compute relative path from the root remote_dir
            rel = remote_file[len(remote_dir):].lstrip("/")
            local_file = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(local_file) or local_dir, exist_ok=True)

            try:
                fs.get(remote_file, local_file)
                size = os.path.getsize(local_file)
                total_bytes += size
                if i % 10 == 0 or i == len(all_files):
                    self.logger.info(
                        f"  [{i}/{len(all_files)}] {_human(total_bytes)} downloaded"
                    )
            except Exception as e:
                self.logger.error(f"  FAIL {rel}: {e}")

        return len(all_files), total_bytes

    def _collect_remote_files(self, fs, path: str, result: list):
        try:
            entries = fs.ls(path, detail=True)
        except Exception:
            return
        for entry in entries:
            name = entry if isinstance(entry, str) else entry.get("name", "")
            etype = None if isinstance(entry, str) else entry.get("type", "")
            if etype == "directory" or (isinstance(entry, dict) and entry.get("isdir")):
                self._collect_remote_files(fs, name, result)
            else:
                result.append(name)

    # ── upload ────────────────────────────────────────────────────────────────

    def _upload(self, fs, local_dir: str, remote_dir: str) -> tuple[int, int]:
        if not os.path.isdir(local_dir):
            raise ValueError(f"Local path does not exist or is not a directory: {local_dir}")

        local_path_obj = __import__("pathlib").Path(local_dir)
        files = sorted(f for f in local_path_obj.rglob("*") if f.is_file())

        if not files:
            self.logger.warning(f"No files found in local path: {local_dir}")
            return 0, 0

        total_bytes = sum(f.stat().st_size for f in files)
        self.logger.info(f"Uploading {len(files)} files ({_human(total_bytes)})...")

        uploaded_bytes = 0
        failed = 0

        for i, local_file in enumerate(files, 1):
            rel = str(local_file.relative_to(local_path_obj))
            remote_file = f"{remote_dir}/{rel}".replace("\\", "/")

            try:
                parent = "/".join(remote_file.split("/")[:-1])
                fs.makedirs(parent, exist_ok=True)
                fs.put(str(local_file), remote_file)
                uploaded_bytes += local_file.stat().st_size
                if i % 10 == 0 or i == len(files):
                    self.logger.info(
                        f"  [{i}/{len(files)}] {_human(uploaded_bytes)}/{_human(total_bytes)} uploaded"
                    )
            except Exception as e:
                self.logger.error(f"  FAIL {rel}: {e}")
                failed += 1

        return len(files) - failed, uploaded_bytes
