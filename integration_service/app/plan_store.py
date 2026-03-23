from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import ExplicitCutPlanRecord


class SqliteExplicitCutPlanStore:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS explicit_cut_plans (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    note TEXT,
                    result_message TEXT
                )
                """
            )

    def add_plan(self, record: ExplicitCutPlanRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO explicit_cut_plans (
                    id,
                    status,
                    created_at,
                    updated_at,
                    request_json,
                    plan_json,
                    note,
                    result_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.status,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.request.model_dump_json(),
                    json.dumps(record.plan, ensure_ascii=True),
                    record.note,
                    record.result_message,
                ),
            )

    def get_plan(self, plan_id: str) -> ExplicitCutPlanRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    status,
                    created_at,
                    updated_at,
                    request_json,
                    plan_json,
                    note,
                    result_message
                FROM explicit_cut_plans
                WHERE id = ?
                """,
                (plan_id,),
            ).fetchone()
        if row is None:
            return None
        return self._record_from_row(row)

    def list_plans(self, limit: int = 100) -> list[ExplicitCutPlanRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    status,
                    created_at,
                    updated_at,
                    request_json,
                    plan_json,
                    note,
                    result_message
                FROM explicit_cut_plans
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def update_plan(self, plan_id: str, **fields: object) -> ExplicitCutPlanRecord:
        current = self.get_plan(plan_id)
        if current is None:
            raise KeyError(f'Explicit cut plan not found: {plan_id}')

        payload = current.model_dump(mode='json')
        payload.update(fields)
        payload['updated_at'] = _now_utc().isoformat()
        updated = ExplicitCutPlanRecord.model_validate(payload)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE explicit_cut_plans
                SET
                    status = ?,
                    updated_at = ?,
                    request_json = ?,
                    plan_json = ?,
                    note = ?,
                    result_message = ?
                WHERE id = ?
                """,
                (
                    updated.status,
                    updated.updated_at.isoformat(),
                    updated.request.model_dump_json(),
                    json.dumps(updated.plan, ensure_ascii=True),
                    updated.note,
                    updated.result_message,
                    plan_id,
                ),
            )
        return updated

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> ExplicitCutPlanRecord:
        return ExplicitCutPlanRecord.model_validate(
            {
                'id': row['id'],
                'status': row['status'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
                'request': json.loads(row['request_json']),
                'plan': json.loads(row['plan_json']),
                'note': row['note'],
                'result_message': row['result_message'],
            }
        )


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)
