from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from contextlib import suppress
from pathlib import Path

from fastapi.testclient import TestClient

from explicit_content_cut import ExplicitCutPlan, TimeSpan


class ExplicitCutApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix='explicit_cut_api_test_')
        temp_path = Path(self._temp_dir.name)
        os.environ['INTEGRATION_EXPLICIT_CUT_PLAN_DB'] = str(
            temp_path / 'explicit_cut_plans.sqlite3'
        )
        os.environ['TRANSCRIBE_VIDEO_FOLDER'] = str(temp_path)

        for module_name in [
            'integration_service.app.main',
            'integration_service.app.config',
        ]:
            sys.modules.pop(module_name, None)

        self.app_main = importlib.import_module('integration_service.app.main')
        self._original_create_plan = self.app_main.create_plan
        self._original_apply_plan = self.app_main.apply_plan
        self._original_create_pool = self.app_main.create_pool
        self.app_main.create_plan = self._fake_create_plan
        self.app_main.apply_plan = self._fake_apply_plan
        self.app_main.create_pool = self._fake_create_pool

    def tearDown(self) -> None:
        self.app_main.create_plan = self._original_create_plan
        self.app_main.apply_plan = self._original_apply_plan
        self.app_main.create_pool = self._original_create_pool
        with suppress(PermissionError):
            self._temp_dir.cleanup()

    @staticmethod
    async def _fake_create_pool(*args, **kwargs):
        raise RuntimeError('redis disabled in explicit-cut API test')

    @staticmethod
    def _fake_create_plan(*args, **kwargs) -> ExplicitCutPlan:
        return ExplicitCutPlan(
            video_file='Toy.Soldiers.1991.720p.BluRay.x264-[YTS.AG].mp4',
            sidecar_file='Toy.Soldiers.1991.720p.BluRay.x264-[YTS.AG].srt',
            sidecar_encoding_used='utf-8',
            keywords=['panties'],
            matched_block_indexes=[10, 11],
            cut_spans=[TimeSpan(start=742.968, end=856.279)],
            original_duration_sec=6722.688,
            cut_duration_sec=113.311,
            result_duration_sec=6609.377,
            frame_verification_backend='off',
            frame_verification_passed=None,
            frame_verification_summary=None,
            frame_verification_samples=[],
        )

    @staticmethod
    def _fake_apply_plan(settings, request, plan: ExplicitCutPlan) -> ExplicitCutPlan:
        return ExplicitCutPlan(
            video_file=plan.video_file,
            sidecar_file=plan.sidecar_file,
            sidecar_encoding_used=plan.sidecar_encoding_used,
            keywords=plan.keywords,
            matched_block_indexes=plan.matched_block_indexes,
            cut_spans=plan.cut_spans,
            original_duration_sec=plan.original_duration_sec,
            cut_duration_sec=plan.cut_duration_sec,
            result_duration_sec=plan.result_duration_sec,
            frame_verification_backend=plan.frame_verification_backend,
            frame_verification_passed=plan.frame_verification_passed,
            frame_verification_summary=plan.frame_verification_summary,
            frame_verification_samples=plan.frame_verification_samples,
            applied=True,
            backup_dir='C:\\temp\\backup',
            removed_stale_outputs=['C:\\temp\\stale.json'],
        )

    def test_explicit_cut_plan_lifecycle(self) -> None:
        with TestClient(self.app_main.app) as client:
            create_response = client.post(
                '/explicit-cut/plans',
                json={'video_name': 'Toy.Soldiers.1991.720p.BluRay.x264-[YTS.AG].mp4'},
            )
            self.assertEqual(create_response.status_code, 201)
            created_payload = create_response.json()
            self.assertEqual(created_payload['status'], 'planned')
            plan_id = created_payload['id']

            get_response = client.get(f'/explicit-cut/plans/{plan_id}')
            self.assertEqual(get_response.status_code, 200)
            plan_payload = get_response.json()
            self.assertEqual(plan_payload['status'], 'planned')
            self.assertEqual(plan_payload['plan']['cut_duration_sec'], 113.311)

            rejected_apply = client.post(f'/explicit-cut/plans/{plan_id}/apply')
            self.assertEqual(rejected_apply.status_code, 409)

            approve_response = client.post(
                f'/explicit-cut/plans/{plan_id}/approve',
                json={'approved': True, 'note': 'looks good'},
            )
            self.assertEqual(approve_response.status_code, 200)
            self.assertEqual(approve_response.json()['status'], 'approved')

            apply_response = client.post(f'/explicit-cut/plans/{plan_id}/apply')
            self.assertEqual(apply_response.status_code, 200)
            applied_payload = apply_response.json()
            self.assertEqual(applied_payload['status'], 'applied')
            self.assertTrue(applied_payload['plan']['applied'])

    def test_explicit_cut_plan_form_flow(self) -> None:
        with TestClient(self.app_main.app) as client:
            create_response = client.post(
                '/explicit-cut/plans/form',
                data={
                    'video_name': 'Toy.Soldiers.1991.720p.BluRay.x264-[YTS.AG].mp4',
                    'frame_verification_backend': 'off',
                },
                follow_redirects=False,
            )
            self.assertEqual(create_response.status_code, 303)


if __name__ == '__main__':
    unittest.main()
