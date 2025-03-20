from typing import (List,
                    Optional,)

from sqlalchemy import (ARRAY,
                        Column,
                        Float,
                        ForeignKeyConstraint,
                        Integer,
                        PrimaryKeyConstraint,
                        Sequence,
                        SmallInteger,
                        String,
                        Table,
                        UniqueConstraint,)
from sqlalchemy.orm import (Mapped,
                            declarative_base,
                            mapped_column,
                            relationship,)
from sqlalchemy.orm.base import Mapped

Base = declarative_base()
metadata = Base.metadata


class AreasDensity(Base):
    __tablename__ = 'areas_density'
    __table_args__ = (
        PrimaryKeyConstraint('id_area_density', name='areas_density_pkey'),
        UniqueConstraint('area_density', name='areas_density_area_density_key')
    )

    id_area_density = mapped_column(SmallInteger)
    area_density = mapped_column(Float)


class ArrangeOrders(Base):
    __tablename__ = 'arrange_orders'
    __table_args__ = (
        PrimaryKeyConstraint('id_arrange_order', name='arrange_orders_pkey'),
        UniqueConstraint('arrange_order', name='arrange_orders_arrange_order_key')
    )

    id_arrange_order = mapped_column(SmallInteger)
    arrange_order = mapped_column(String(10))


class Buildings(Base):
    __tablename__ = 'buildings'
    __table_args__ = (
        PrimaryKeyConstraint('id_building', name='buildings_pkey'),
        UniqueConstraint('breadth', 'depth', 'height', name='buildings_breadth_depth_height_key')
    )

    id_building = mapped_column(SmallInteger)
    breadth = mapped_column(SmallInteger)
    depth = mapped_column(SmallInteger)
    height = mapped_column(SmallInteger)

    interference: Mapped[List['Interference']] = relationship('Interference', uselist=True, back_populates='buildings')


class EaveTypes(Base):
    __tablename__ = 'eave_types'
    __table_args__ = (
        PrimaryKeyConstraint('id_eave', name='eave_types_pkey'),
        UniqueConstraint('eave', name='eave_types_eave_key')
    )

    id_eave = mapped_column(SmallInteger)
    eave = mapped_column(String(1))


class ExperimentsAlpha4(Base):
    __tablename__ = 'experiments_alpha_4'
    __table_args__ = (
        PrimaryKeyConstraint('model_id', name='experiments_alpha_4_pkey'),
    )

    model_id = mapped_column(Integer)
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
    model_name = mapped_column(SmallInteger, nullable=False)
    x_coordinates = mapped_column(ARRAY(Float()), nullable=False)
    z_coordinates = mapped_column(ARRAY(Float()), nullable=False)
    face_number = mapped_column(ARRAY(SmallInteger()), nullable=False)


class RoofPitches(Base):
    __tablename__ = 'roof_pitches'
    __table_args__ = (
        PrimaryKeyConstraint('id_pitch', name='roof_pitches_pkey'),
        UniqueConstraint('pitch', name='roof_pitches_pitch_key')
    )

    id_pitch = mapped_column(SmallInteger)
    pitch = mapped_column(Float)


class RoofTypes(Base):
    __tablename__ = 'roof_types'
    __table_args__ = (
        PrimaryKeyConstraint('id_roof', name='roof_types_pkey'),
        UniqueConstraint('roof_type', name='roof_types_roof_type_key')
    )

    id_roof = mapped_column(SmallInteger)
    roof_type = mapped_column(String(11))


class SampleFrequencies(Base):
    __tablename__ = 'sample_frequencies'
    __table_args__ = (
        PrimaryKeyConstraint('id_sample_frequency', name='sample_frequencies_pkey'),
        UniqueConstraint('sample_frequency', name='sample_frequencies_sample_frequency_key')
    )

    id_sample_frequency = mapped_column(SmallInteger)
    sample_frequency = mapped_column(Float)


class SamplePeriods(Base):
    __tablename__ = 'sample_periods'
    __table_args__ = (
        PrimaryKeyConstraint('id_sample_period', name='sample_periods_pkey'),
        UniqueConstraint('period', name='sample_periods_period_key')
    )

    id_sample_period = mapped_column(SmallInteger)
    period = mapped_column(Float)


class SurfacesRoof(Base):
    __tablename__ = 'surfaces_roof'
    __table_args__ = (
        PrimaryKeyConstraint('id_surface', name='surfaces_roof_pkey'),
        UniqueConstraint('surface', name='surfaces_roof_surface_key')
    )

    id_surface = mapped_column(SmallInteger)
    surface = mapped_column(ARRAY(SmallInteger()))


class SurroundingHeights(Base):
    __tablename__ = 'surrounding_heights'
    __table_args__ = (
        PrimaryKeyConstraint('id_surrounding_height', name='surrounding_heights_pkey'),
        UniqueConstraint('surrounding_height', name='surrounding_heights_surrounding_height_key')
    )

    id_surrounding_height = mapped_column(SmallInteger)
    surrounding_height = mapped_column(Float)


class WindAzimuths(Base):
    __tablename__ = 'wind_azimuths'
    __table_args__ = (
        PrimaryKeyConstraint('id_wind_azimuth', name='wind_azimuths_pkey'),
        UniqueConstraint('wind_azimuth', name='wind_azimuths_wind_azimuth_key')
    )

    id_wind_azimuth = mapped_column(SmallInteger)
    wind_azimuth = mapped_column(Float)


class XCoordinatesRoof(Base):
    __tablename__ = 'x_coordinates_roof'
    __table_args__ = (
        PrimaryKeyConstraint('id_x_coordinates', name='x_coordinates_roof_pkey'),
        UniqueConstraint('x_coordinates', name='x_coordinates_roof_x_coordinates_key')
    )

    id_x_coordinates = mapped_column(SmallInteger)
    x_coordinates = mapped_column(ARRAY(SmallInteger()))


class YCoordinatesRoof(Base):
    __tablename__ = 'y_coordinates_roof'
    __table_args__ = (
        PrimaryKeyConstraint('id_y_coordinates', name='y_coordinates_roof_pkey'),
        UniqueConstraint('y_coordinates', name='y_coordinates_roof_y_coordinates_key')
    )

    id_y_coordinates = mapped_column(SmallInteger)
    y_coordinates = mapped_column(ARRAY(SmallInteger()))


t_building_with_eaves = Table(
    'building_with_eaves', metadata,
    Column('breadth', SmallInteger),
    Column('depth', SmallInteger),
    Column('height', SmallInteger),
    Column('id_eave', SmallInteger),
    Column('id_x_coordinates', SmallInteger),
    Column('id_y_coordinates', SmallInteger),
    Column('id_surface', SmallInteger),
    Column('id_pitch', SmallInteger),
    Column('id_roof', SmallInteger),
    Column('id_sample_frequency', SmallInteger),
    Column('id_sample_period', SmallInteger),
    Column('id_wind_azimuth', SmallInteger),
    Column('pressure_coefficients', ARRAY(SmallInteger())),
    ForeignKeyConstraint(['id_eave'], ['eave_types.id_eave'], name='building_with_eaves_id_eave_fkey'),
    ForeignKeyConstraint(['id_pitch'], ['roof_pitches.id_pitch'], name='building_with_eaves_id_pitch_fkey'),
    ForeignKeyConstraint(['id_roof'], ['roof_types.id_roof'], name='building_with_eaves_id_roof_fkey'),
    ForeignKeyConstraint(['id_sample_frequency'], ['sample_frequencies.id_sample_frequency'], name='building_with_eaves_id_sample_frequency_fkey'),
    ForeignKeyConstraint(['id_sample_period'], ['sample_periods.id_sample_period'], name='building_with_eaves_id_sample_period_fkey'),
    ForeignKeyConstraint(['id_surface'], ['surfaces_roof.id_surface'], name='building_with_eaves_id_surface_fkey'),
    ForeignKeyConstraint(['id_wind_azimuth'], ['wind_azimuths.id_wind_azimuth'], name='building_with_eaves_id_wind_azimuth_fkey'),
    ForeignKeyConstraint(['id_x_coordinates'], ['x_coordinates_roof.id_x_coordinates'], name='building_with_eaves_id_x_coordinates_fkey'),
    ForeignKeyConstraint(['id_y_coordinates'], ['y_coordinates_roof.id_y_coordinates'], name='building_with_eaves_id_y_coordinates_fkey'),
    UniqueConstraint('breadth', 'depth', 'height', 'id_eave', 'id_wind_azimuth', 'id_pitch', 'id_roof', name='building_with_eaves_breadth_depth_height_id_wind_azimuth_id_key')
)


t_building_without_eaves = Table(
    'building_without_eaves', metadata,
    Column('breadth', SmallInteger),
    Column('depth', SmallInteger),
    Column('height', SmallInteger),
    Column('id_x_coordinates', SmallInteger),
    Column('id_y_coordinates', SmallInteger),
    Column('id_surface', SmallInteger),
    Column('id_pitch', SmallInteger),
    Column('id_roof', SmallInteger),
    Column('sample_frequency', SmallInteger),
    Column('id_sample_period', SmallInteger),
    Column('angle', SmallInteger),
    Column('pressure_coefficients', ARRAY(SmallInteger())),
    ForeignKeyConstraint(['id_pitch'], ['roof_pitches.id_pitch'], name='building_without_eaves_id_pitch_fkey'),
    ForeignKeyConstraint(['id_roof'], ['roof_types.id_roof'], name='building_without_eaves_id_roof_fkey'),
    ForeignKeyConstraint(['id_sample_period'], ['sample_periods.id_sample_period'], name='building_without_eaves_id_sample_period_fkey'),
    ForeignKeyConstraint(['id_surface'], ['surfaces_roof.id_surface'], name='building_without_eaves_id_surface_fkey'),
    ForeignKeyConstraint(['id_x_coordinates'], ['x_coordinates_roof.id_x_coordinates'], name='building_without_eaves_id_x_coordinates_fkey'),
    ForeignKeyConstraint(['id_y_coordinates'], ['y_coordinates_roof.id_y_coordinates'], name='building_without_eaves_id_y_coordinates_fkey'),
    UniqueConstraint('breadth', 'depth', 'height', 'angle', 'id_pitch', 'id_roof', name='building_without_eaves_breadth_depth_height_angle_id_pitch__key')
)


class Interference(Base):
    __tablename__ = 'interference'
    __table_args__ = (
        ForeignKeyConstraint(['id_interfering_building'], ['buildings.id_building'], name='interference_id_interfering_building_fkey'),
        PrimaryKeyConstraint('id_interference', name=' id_interference')
    )

    id_interference = mapped_column(Integer, Sequence('interference_ id_interference_seq'))
    position = mapped_column(SmallInteger)
    angle = mapped_column(SmallInteger)
    id_interfering_building = mapped_column(SmallInteger)
    pressure_coefficients = mapped_column(ARRAY(SmallInteger()))

    buildings: Mapped[Optional['Buildings']] = relationship('Buildings', back_populates='interference')


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


t_non_isolated_building = Table(
    'non_isolated_building', metadata,
    Column('id_area_density', SmallInteger),
    Column('id_arrange_order', SmallInteger),
    Column('breadth', SmallInteger),
    Column('depth', SmallInteger),
    Column('height', SmallInteger),
    Column('id_x_coordinates', SmallInteger),
    Column('id_y_coordinates', SmallInteger),
    Column('id_surface', SmallInteger),
    Column('id_pitch', SmallInteger),
    Column('id_roof', SmallInteger),
    Column('id_sample_frequency', SmallInteger),
    Column('id_sample_period', SmallInteger),
    Column('id_surrounding_height', SmallInteger),
    Column('id_wind_azimuth', SmallInteger),
    Column('pressure_coefficients', ARRAY(SmallInteger())),
    ForeignKeyConstraint(['id_area_density'], ['areas_density.id_area_density'], name='non_isolated_building_id_area_density_fkey'),
    ForeignKeyConstraint(['id_arrange_order'], ['arrange_orders.id_arrange_order'], name='non_isolated_building_id_arrange_order_fkey'),
    ForeignKeyConstraint(['id_pitch'], ['roof_pitches.id_pitch'], name='non_isolated_building_id_pitch_fkey'),
    ForeignKeyConstraint(['id_roof'], ['roof_types.id_roof'], name='non_isolated_building_id_roof_fkey'),
    ForeignKeyConstraint(['id_sample_frequency'], ['sample_frequencies.id_sample_frequency'], name='non_isolated_building_id_sample_frequency_fkey'),
    ForeignKeyConstraint(['id_sample_period'], ['sample_periods.id_sample_period'], name='non_isolated_building_id_sample_period_fkey'),
    ForeignKeyConstraint(['id_surface'], ['surfaces_roof.id_surface'], name='non_isolated_building_id_surface_fkey'),
    ForeignKeyConstraint(['id_surrounding_height'], ['surrounding_heights.id_surrounding_height'], name='non_isolated_building_id_surrounding_height_fkey'),
    ForeignKeyConstraint(['id_wind_azimuth'], ['wind_azimuths.id_wind_azimuth'], name='non_isolated_building_id_wind_azimuth_fkey'),
    ForeignKeyConstraint(['id_x_coordinates'], ['x_coordinates_roof.id_x_coordinates'], name='non_isolated_building_id_x_coordinates_fkey'),
    ForeignKeyConstraint(['id_y_coordinates'], ['y_coordinates_roof.id_y_coordinates'], name='non_isolated_building_id_y_coordinates_fkey'),
    UniqueConstraint('breadth', 'depth', 'height', 'id_wind_azimuth', 'id_pitch', 'id_roof', 'id_area_density', 'id_arrange_order', name='non_isolated_building_breadth_depth_height_id_wind_azimuth__key')
)
