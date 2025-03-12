from sqlalchemy import (ARRAY,
                        VARCHAR,
                        Column,
                        Float,
                        ForeignKeyConstraint,
                        Integer,
                        PrimaryKeyConstraint,
                        SmallInteger,
                        Table,)
from sqlalchemy.orm import (declarative_base,
                            mapped_column, )

Base = declarative_base()
metadata = Base.metadata


class ExperimentsAlpha4(Base):
    __tablename__ = 'experiments_alpha_4'
    __table_args__ = (
        PrimaryKeyConstraint('model_id', name='experiments_alpha_4_pkey'),
    )

    model_id = mapped_column(VARCHAR(3))
    model_name = mapped_column(SmallInteger, nullable=False)
    x_coordinates = mapped_column(ARRAY(Float()), nullable=False)
    z_coordinates = mapped_column(ARRAY(Float()), nullable=False)
    face_number = mapped_column(ARRAY(SmallInteger()), nullable=False)


class ExperimentsAlpha6(Base):
    __tablename__ = 'experiments_alpha_6'
    __table_args__ = (
        PrimaryKeyConstraint('model_id', name='experiments_alpha_6_pkey'),
    )

    model_id = mapped_column(Integer)
    model_name = mapped_column(VARCHAR(3), nullable=False)
    x_coordinates = mapped_column(ARRAY(Float()), nullable=False)
    z_coordinates = mapped_column(ARRAY(Float()), nullable=False)
    face_number = mapped_column(ARRAY(SmallInteger()), nullable=False)


t_models_alpha_4 = Table(
    'models_alpha_4', metadata,
    Column('model_id', Integer, nullable=False),
    Column('angle', SmallInteger, nullable=False),
    Column('pressure_coefficients', ARRAY(SmallInteger()), nullable=False),
    ForeignKeyConstraint(['model_id'], ['experiments_alpha_4.model_id'], name='fk_e99481390edf4b0e87ccab2040fcde48')
)

t_models_alpha_6 = Table(
    'models_alpha_6', metadata,
    Column('model_id', Integer, nullable=False),
    Column('angle', SmallInteger, nullable=False),
    Column('pressure_coefficients', ARRAY(SmallInteger()), nullable=False),
    ForeignKeyConstraint(['model_id'], ['experiments_alpha_6.model_id'], name='fk_dc23ad1d409d4698bec7b2cfccb04b17')
)
