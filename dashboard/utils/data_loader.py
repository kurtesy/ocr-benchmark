import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy.sql import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from typing import Dict, Any, List, TypedDict, Optional

load_dotenv()


class BenchmarkRunMetadata(TypedDict):
    timestamp: str
    status: str
    run_by: Optional[str]
    description: Optional[str]
    total_documents: Optional[int]
    created_at: Optional[str]
    completed_at: Optional[str]


def load_run_list_from_folder(
    results_dir: str = "results",
) -> List[BenchmarkRunMetadata]:
    """Load list of benchmark runs from the results directory"""
    results_path = Path(results_dir)
    result_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    runs = []

    for dir_path in result_dirs:
        timestamp = dir_path.name
        json_path = dir_path / "results.json"
        if json_path.exists():
            runs.append(
                {
                    "timestamp": timestamp,
                    "status": "completed",  # Assuming completed if file exists
                    "run_by": None,
                    "description": None,
                    "total_documents": None,
                    "created_at": format_timestamp(timestamp),
                    "completed_at": format_timestamp(timestamp),
                }
            )

    return sorted(runs, key=lambda x: x["timestamp"], reverse=True)


def load_run_list_from_db() -> List[BenchmarkRunMetadata]:
    """Load list of benchmark runs from database"""
    database_url = os.getenv("DATABASE_URL")
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    query = text(
        """
        SELECT 
            timestamp,
            status,
            run_by,
            description,
            total_documents,
            created_at,
            completed_at
        FROM benchmark_runs
        ORDER BY created_at DESC
    """
    )

    rows = session.execute(query)
    runs = []

    for row in rows:
        runs.append(
            {
                "timestamp": row.timestamp,
                "status": row.status,
                "run_by": row.run_by,
                "description": row.description,
                "total_documents": row.total_documents,
                "created_at": (
                    row.created_at.strftime("%Y-%m-%d %H:%M:%S")
                    if row.created_at
                    else None
                ),
                "completed_at": (
                    row.completed_at.strftime("%Y-%m-%d %H:%M:%S")
                    if row.completed_at
                    else None
                ),
            }
        )

    session.close()
    return runs


def load_results_for_run_from_folder(
    timestamp: str, results_dir: str = "results"
) -> Dict[str, Any]:
    """Load results for a specific run from folder"""
    results_path = Path(results_dir) / timestamp / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
            # Assign id to each result if not already present
            for idx, result in enumerate(results):
                if "id" not in result:
                    result["id"] = idx
            total_documents = len(results)
            return {
                "results": results,
                "status": "completed",
                "run_by": None,
                "description": None,
                "total_documents": total_documents,
                "created_at": format_timestamp(timestamp),
                "completed_at": format_timestamp(timestamp),
            }
    return {}


def load_results_for_run_from_db(
    timestamp: str, include_metrics_only: bool = True
) -> Dict[str, Any]:
    """Load results for a specific run from database"""
    database_url = os.getenv("DATABASE_URL")
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    if not include_metrics_only:
        output_string = """
        'trueMarkdown', bres.true_markdown,
        'predictedMarkdown', bres.predicted_markdown,
        'trueJson', bres.true_json,
        'predictedJson', bres.predicted_json,
        'jsonDiff', bres.json_diff,
        'fullJsonDiff', bres.full_json_diff,
        """
    else:
        output_string = ""

    query = text(
        f"""
        WITH filtered_run AS (
            SELECT id, timestamp, status, run_by, description, total_documents, created_at, completed_at
            FROM benchmark_runs
            WHERE timestamp = :timestamp
        )
        SELECT 
            fr.timestamp,
            fr.status,
            fr.run_by,
            fr.description,
            fr.total_documents,
            fr.created_at,
            fr.completed_at,
            json_agg(
                json_build_object(
                    'id', bres.id,
                    'fileUrl', bres.file_url,
                    'ocrModel', bres.ocr_model,
                    'extractionModel', bres.extraction_model,
                    'directImageExtraction', bres.direct_image_extraction,
                    {output_string}
                    'levenshteinDistance', bres.levenshtein_distance,
                    'jsonAccuracy', bres.json_accuracy,
                    'jsonAccuracyResult', bres.json_accuracy_result,
                    'jsonDiffStats', bres.json_diff_stats,
                    'metadata', bres.metadata,
                    'usage', bres.usage,
                    'error', bres.error
                )
            ) as results
        FROM filtered_run fr
        LEFT JOIN benchmark_results bres ON fr.id = bres.benchmark_run_id
        GROUP BY fr.id, fr.timestamp, fr.status, fr.run_by, fr.description, fr.total_documents, fr.created_at, fr.completed_at
    """
    )

    row = session.execute(query, {"timestamp": timestamp}).first()
    session.close()

    if row:
        return {
            "results": row.results,
            "status": row.status,
            "total_documents": row.total_documents,
            "run_by": row.run_by,
            "description": row.description,
            "created_at": (
                row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else None
            ),
            "completed_at": (
                row.completed_at.strftime("%Y-%m-%d %H:%M:%S")
                if row.completed_at
                else None
            ),
        }
    return {}


def load_one_result_from_db(timestamp: str, id: str) -> Dict[str, Any]:
    """Load one test case result from database for a specific run and file"""
    database_url = os.getenv("DATABASE_URL")
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    query = text(
        """
        WITH filtered_results AS (
            SELECT *
            FROM benchmark_results
            WHERE id = :id
        )
        SELECT 
            br.timestamp,
            br.status,
            br.run_by,
            br.description,
            br.total_documents,
            br.created_at,
            br.completed_at,
            json_build_object(
                'id', fr.id,
                'fileUrl', fr.file_url,
                'ocrModel', fr.ocr_model,
                'extractionModel', fr.extraction_model,
                'directImageExtraction', fr.direct_image_extraction,
                'trueMarkdown', fr.true_markdown,
                'predictedMarkdown', fr.predicted_markdown,
                'trueJson', fr.true_json,
                'predictedJson', fr.predicted_json,
                'jsonDiff', fr.json_diff,
                'fullJsonDiff', fr.full_json_diff,
                'jsonDiffStats', fr.json_diff_stats,
                'levenshteinDistance', fr.levenshtein_distance,
                'jsonAccuracy', fr.json_accuracy,
                'jsonAccuracyResult', fr.json_accuracy_result,
                'jsonSchema', fr.json_schema,
                'metadata', fr.metadata,
                'usage', fr.usage,
                'error', fr.error
            ) as result
        FROM benchmark_runs br
        INNER JOIN filtered_results fr ON br.id = fr.benchmark_run_id
        WHERE br.timestamp = :timestamp
        LIMIT 1
    """
    )

    row = session.execute(query, {"timestamp": timestamp, "id": id}).first()
    session.close()

    if row:
        return {
            "result": row.result,
            "status": row.status,
            "run_by": row.run_by,
            "description": row.description,
            "created_at": (
                row.created_at.strftime("%Y-%m-%d %H:%M:%S") if row.created_at else None
            ),
            "completed_at": (
                row.completed_at.strftime("%Y-%m-%d %H:%M:%S")
                if row.completed_at
                else None
            ),
        }
    return {}


def load_one_result_from_folder(
    timestamp: str, id: str, results_dir: str = "results"
) -> Dict[str, Any]:
    """Load one test case result from folder for a specific run and file"""
    results_path = Path(results_dir) / timestamp / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
            for idx, result in enumerate(results):
                if idx == id:
                    return {
                        "result": result,
                        "status": "completed",
                        "run_by": None,
                        "description": None,
                        "created_at": format_timestamp(timestamp),
                        "completed_at": format_timestamp(timestamp),
                    }
    return {}


def load_run_list() -> List[BenchmarkRunMetadata]:
    """Load list of benchmark runs from either database or local files"""
    if os.getenv("DATABASE_URL"):
        return load_run_list_from_db()
    return load_run_list_from_folder()


def load_results_for_run(
    timestamp: str, include_metrics_only: bool = True
) -> Dict[str, Any]:
    """Load results for a specific run from either database or local files"""
    if os.getenv("DATABASE_URL"):
        return load_results_for_run_from_db(timestamp, include_metrics_only)
    return load_results_for_run_from_folder(timestamp)


def load_one_result(timestamp: str, id: str) -> Dict[str, Any]:
    """Load one test case result from either database or local files"""
    if os.getenv("DATABASE_URL"):
        return load_one_result_from_db(timestamp, id)
    return load_one_result_from_folder(timestamp, id)


def format_timestamp(timestamp: str) -> str:
    """Convert timestamp string to readable format"""
    return datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S").strftime(
        "%Y-%m-%d %H:%M:%S"
    )
