from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    Text,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ExperimentRecord(Base):
    __tablename__ = "experiments"

    id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    status = Column(String(32), default="running")
    current_round = Column(Integer, default=1)
    best_score = Column(Float, default=0.0)
    config_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    trees = relationship("DiscoveryTreeRecord", back_populates="experiment", cascade="all, delete-orphan")
    traces = relationship("RoutingTraceRecord", back_populates="experiment", cascade="all, delete-orphan")


class DiscoveryTreeRecord(Base):
    __tablename__ = "discovery_trees"

    id = Column(String(64), primary_key=True)
    experiment_id = Column(String(64), ForeignKey("experiments.id"), nullable=False)
    round_id = Column(Integer, nullable=False)
    problem_id = Column(String(64), nullable=False)
    root_id = Column(String(64), nullable=True)
    node_count = Column(Integer, default=0)
    tree_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    experiment = relationship("ExperimentRecord", back_populates="trees")
    nodes = relationship("DiscoveryNodeRecord", back_populates="tree", cascade="all, delete-orphan")


class DiscoveryNodeRecord(Base):
    __tablename__ = "discovery_nodes"

    id = Column(String(64), primary_key=True)
    tree_id = Column(String(64), ForeignKey("discovery_trees.id"), nullable=False)
    parent_id = Column(String(64), nullable=True)
    agent_id = Column(String(64), nullable=False)
    proposal = Column(Text, nullable=False)
    artifact_uri = Column(String(256), nullable=False)
    score = Column(Float, nullable=False)
    correct = Column(Boolean, nullable=False)
    runtime_ms = Column(Float, nullable=True)
    depth = Column(Integer, default=0)
    round_id = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    tree = relationship("DiscoveryTreeRecord", back_populates="nodes")


class PolicyVersionRecord(Base):
    __tablename__ = "policy_versions"

    id = Column(String(64), primary_key=True)
    version = Column(String(64), nullable=False)
    source_code = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    generation = Column(Integer, default=1)
    train_score = Column(Float, default=0.0)
    val_score = Column(Float, default=0.0)
    promoted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RoutingTraceRecord(Base):
    __tablename__ = "routing_traces"

    id = Column(String(64), primary_key=True)
    experiment_id = Column(String(64), ForeignKey("experiments.id"), nullable=True)
    task_id = Column(String(64), nullable=False)
    selected_agent = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities_json = Column(JSON, nullable=False)
    fallback_used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    experiment = relationship("ExperimentRecord", back_populates="traces")

