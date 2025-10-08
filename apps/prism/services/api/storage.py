"""SQLite storage helpers for PRISM."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterable
from datetime import datetime
from pathlib import Path
from typing import List

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .models import CollectorResult, PersonInput

PRISM_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PRISM_ROOT / "data"
DB_PATH = DATA_DIR / "prism.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"  # pragma: no cover


class Base(DeclarativeBase):
    pass


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String)
    phone: Mapped[str | None] = mapped_column(String, nullable=True)
    email: Mapped[str | None] = mapped_column(String, nullable=True)
    username: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    artifacts: Mapped[List["Artifact"]] = relationship(back_populates="person", cascade="all, delete-orphan")
    highlights: Mapped[List["Highlight"]] = relationship(back_populates="person", cascade="all, delete-orphan")
    provenance: Mapped[List["Provenance"]] = relationship(back_populates="person", cascade="all, delete-orphan")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id"))
    type: Mapped[str] = mapped_column(String)
    value: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String)
    raw_path: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    person: Mapped[Person] = relationship(back_populates="artifacts")


class Highlight(Base):
    __tablename__ = "highlights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id"))
    title: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    person: Mapped[Person] = relationship(back_populates="highlights")


class Provenance(Base):
    __tablename__ = "provenance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id"))
    source: Mapped[str] = mapped_column(String)
    reference: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    person: Mapped[Person] = relationship(back_populates="provenance")


engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


async def upsert_person(session: AsyncSession, slug: str, payload: PersonInput) -> Person:
    result = await session.execute(
        Person.__table__.select().where(Person.slug == slug)
    )
    row = result.first()
    now = datetime.utcnow()
    if row:
        person = await session.get(Person, row["id"])
        assert person is not None
        person.name = payload.name
        person.phone = payload.phone
        person.email = payload.email
        person.username = payload.username
        await session.flush()
    else:
        person = Person(
            slug=slug,
            name=payload.name,
            phone=payload.phone,
            email=payload.email,
            username=payload.username,
            created_at=now,
        )
        session.add(person)
        await session.flush()
    return person


async def clear_previous(session: AsyncSession, person: Person) -> None:
    await session.execute(Artifact.__table__.delete().where(Artifact.person_id == person.id))
    await session.execute(Highlight.__table__.delete().where(Highlight.person_id == person.id))
    await session.execute(Provenance.__table__.delete().where(Provenance.person_id == person.id))


async def persist_results(session: AsyncSession, person: Person, results: Iterable[CollectorResult]) -> None:
    for result in results:
        for artifact in result.artifacts:
            raw_path = None
            if artifact.raw_path:
                try:
                    raw_path = str(artifact.raw_path.relative_to(PRISM_ROOT))
                except ValueError:
                    raw_path = str(artifact.raw_path)
            session.add(
                Artifact(
                    person_id=person.id,
                    type=artifact.type.value,
                    value=artifact.value,
                    confidence=artifact.confidence,
                    source=result.collector.value,
                    raw_path=raw_path,
                )
            )
        for highlight in result.highlights:
            session.add(
                Highlight(
                    person_id=person.id,
                    title=highlight.title,
                    description=highlight.description,
                    confidence=highlight.confidence,
                )
            )
        for prov in result.provenance:
            session.add(
                Provenance(
                    person_id=person.id,
                    source=prov.source.value,
                    reference=prov.reference,
                    description=prov.description,
                )
            )
    await session.flush()


async def fetch_person_summary(session: AsyncSession, person: Person) -> dict:
    artifacts_result = await session.execute(Artifact.__table__.select().where(Artifact.person_id == person.id))
    highlights_result = await session.execute(Highlight.__table__.select().where(Highlight.person_id == person.id))
    provenance_result = await session.execute(Provenance.__table__.select().where(Provenance.person_id == person.id))

    return {
        "artifacts": [dict(row) for row in artifacts_result.mappings()],
        "highlights": [dict(row) for row in highlights_result.mappings()],
        "provenance": [dict(row) for row in provenance_result.mappings()],
    }


def bootstrap_sync() -> None:
    """Synchronous helper for environments without async entry points."""

    asyncio.run(init_db())
