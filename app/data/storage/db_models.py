from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Integer, String


class Base(DeclarativeBase):
    pass


class UserFacts(Base):
    __tablename__ = "user_facts"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    user_id: str = Column(String, nullable=False)
    fact: str = Column(String, nullable=False)
    category: str = Column(String, nullable=False)
