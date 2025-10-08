param(
    [string]$Root = "$(Split-Path -Parent $MyInvocation.MyCommand.Path)\.."
)

$ErrorActionPreference = "Stop"

Write-Host "[PRISM] Initializing directories at $Root"

$paths = @(
    Join-Path $Root "data",
    Join-Path $Root "data\artifacts",
    Join-Path $Root "output"
)

foreach ($path in $paths) {
    if (-not (Test-Path $path)) {
        Write-Host "[PRISM] Creating $path"
        New-Item -ItemType Directory -Path $path | Out-Null
    } else {
        Write-Host "[PRISM] Found $path"
    }
}

$dbPath = Join-Path $Root "data\prism.db"

if (-not (Test-Path $dbPath)) {
    Write-Host "[PRISM] Creating SQLite database at $dbPath"
    $schema = @'
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT UNIQUE,
    name TEXT,
    phone TEXT,
    email TEXT,
    username TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    type TEXT,
    value TEXT,
    confidence REAL,
    source TEXT,
    raw_path TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS highlights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    title TEXT,
    description TEXT,
    confidence REAL,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS provenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER,
    source TEXT,
    reference TEXT,
    description TEXT,
    created_at TEXT
);
'@

    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "python executable not found in PATH"
    }

    $code = @"
import sqlite3
import pathlib

db_path = r'''$dbPath'''
schema = r'''$schema'''
pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(db_path)
conn.executescript(schema)
conn.close()
"@

    python -c $code
} else {
    Write-Host "[PRISM] SQLite database already exists at $dbPath"
}

Write-Host "[PRISM] Post-install completed"
