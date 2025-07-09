from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String, UniqueConstraint


class Base(DeclarativeBase):
    pass


class UserFacts(Base):
    __tablename__ = "user_facts"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    user_id: str = Column(String, nullable=False)
    fact: str = Column(String, nullable=False)
    category: str = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "category", name="uq_user_category"),)

    @classmethod
    def get_upsert_conflict_target(cls):
        return ["user_id", "category"]

    @classmethod
    def get_upsert_update_fields(cls):
        return ["fact"]
