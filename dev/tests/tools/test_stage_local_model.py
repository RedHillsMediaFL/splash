# Modified by Red Hills Media on 2026-10-10 for this M1 Ultra source release.
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dev.tools import stage_local_model
from install import models, paths, upstream


class StageLocalModelTests(unittest.TestCase):
    def test_upstream_installs_local_repositories_without_resolving_the_hub(self):
        selection = SimpleNamespace(
            variant=None, revision=None, link=Path("/stage/model")
        )
        target = object()
        draft = object()
        with mock.patch(
            "install.upstream.hub.Repository.local_directory",
            side_effect=(target, draft),
        ) as local_directory, mock.patch("install.upstream._install") as install:
            result = upstream.install_local(selection, "/target", "/draft")

        self.assertEqual(result, selection.link)
        self.assertEqual(
            local_directory.call_args_list,
            [mock.call(Path("/target")), mock.call(Path("/draft"))],
        )
        install.assert_called_once_with(selection, target, None, draft)

    def test_upstream_rejects_revision_or_variant_for_local_staging(self):
        for selection in (
            SimpleNamespace(variant="q4", revision=None),
            SimpleNamespace(variant=None, revision="a" * 40),
        ):
            with self.subTest(selection=selection), self.assertRaisesRegex(
                models.ModelError, "cannot use a variant or revision"
            ):
                upstream.install_local(selection, "/target", "/draft")

    def test_stages_both_local_sources_in_an_isolated_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "models"
            target = root / "target"
            draft = root / "draft"
            target.mkdir()
            draft.mkdir()
            with mock.patch(
                "dev.tools.stage_local_model.upstream.install_local",
                return_value=staging / ".selections" / "model",
            ) as install_local:
                result = stage_local_model.stage_local_model(
                    staging, "local/qwen36-agent-core", target, draft
                )

            selection = install_local.call_args.args[0]
            self.assertEqual(selection.models_root, staging.resolve())
            self.assertEqual(selection.model, "local/qwen36-agent-core")
            self.assertEqual(selection.draft_model, str(draft.resolve()))
            install_local.assert_called_once_with(
                selection, target.resolve(), draft.resolve()
            )
            self.assertEqual(
                result,
                {
                    "draft": str(draft.resolve()),
                    "model_root": str(staging / ".selections" / "model"),
                    "models_root": str(staging.resolve()),
                    "target": str(target.resolve()),
                },
            )

    def test_refuses_the_production_models_root_and_its_children(self):
        installed = Path.home() / "Library/Application Support/Splash/models"
        for root in (
            paths.MODELS,
            paths.MODELS / "candidate",
            installed,
            installed / "candidate",
        ):
            with self.subTest(root=root), self.assertRaisesRegex(
                models.ModelError, "must not use the production"
            ):
                stage_local_model.stage_local_model(
                    root, "local/candidate", "/tmp/target", "/tmp/draft"
                )

    def test_cli_reports_machine_readable_result_and_failure(self):
        result = {
            "draft": "/tmp/draft",
            "model_root": "/tmp/models/local/candidate",
            "models_root": "/tmp/models",
            "target": "/tmp/target",
        }
        with mock.patch(
            "dev.tools.stage_local_model.stage_local_model", return_value=result
        ), contextlib.redirect_stdout(io.StringIO()) as stdout:
            status = stage_local_model.main(
                [
                    "--models",
                    "/tmp/models",
                    "--name",
                    "local/candidate",
                    "--target",
                    "/tmp/target",
                    "--draft",
                    "/tmp/draft",
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue()), result)

        with mock.patch(
            "dev.tools.stage_local_model.stage_local_model",
            side_effect=models.ModelError("invalid candidate"),
        ), contextlib.redirect_stderr(io.StringIO()) as stderr:
            status = stage_local_model.main(
                [
                    "--models",
                    "/tmp/models",
                    "--name",
                    "local/candidate",
                    "--target",
                    "/tmp/target",
                    "--draft",
                    "/tmp/draft",
                ]
            )
        self.assertEqual(status, 1)
        self.assertEqual(stderr.getvalue(), "error: invalid candidate\n")


if __name__ == "__main__":
    unittest.main()
