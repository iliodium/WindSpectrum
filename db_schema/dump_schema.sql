BEGIN;

CREATE TABLE areas_density (
    id_area_density smallint NOT NULL,
    area_density real
);



CREATE TABLE arrange_orders (
    id_arrange_order smallint NOT NULL,
    arrange_order character varying(10)
);



CREATE TABLE building_with_eaves (
    id_building_with_eaves int,
    breadth smallint,
    depth smallint,
    height smallint,
    id_eave smallint,
    id_x_coordinates smallint,
    id_y_coordinates smallint,
    id_surface smallint,
    id_pitch smallint,
    id_roof smallint,
    id_sample_frequency smallint,
    id_sample_period smallint,
    id_wind_azimuth smallint,
    pressure_coefficients BLOB
);



CREATE TABLE building_without_eaves (
    breadth smallint,
    depth smallint,
    height smallint,
    id_x_coordinates smallint,
    id_y_coordinates smallint,
    id_surface smallint,
    id_pitch smallint,
    id_roof smallint,
    sample_frequency smallint,
    id_sample_period smallint,
    angle smallint,
    pressure_coefficients BLOB
);



CREATE TABLE buildings (
    id_building smallint NOT NULL,
    breadth smallint,
    depth smallint,
    height smallint
);



CREATE TABLE eave_types (
    id_eave smallint NOT NULL,
    eave character varying(1)
);



CREATE TABLE experiments_alpha_4 (
    model_id integer NOT NULL,
    model_name smallint NOT NULL,
    x_coordinates BLOB NOT NULL,
    z_coordinates BLOB NOT NULL,
    face_number BLOB NOT NULL
);





CREATE TABLE experiments_alpha_6 (
    model_id integer NOT NULL,
    model_name smallint NOT NULL,
    x_coordinates BLOB NOT NULL,
    z_coordinates BLOB NOT NULL,
    face_number BLOB NOT NULL
);




CREATE TABLE interference (
    id_interference integer,
    position smallint,
    angle smallint,
    id_interfering_building smallint,
    pressure_coefficients smallint[]
);



CREATE TABLE mean_wind_speeds (
    id_mean_wind_speed smallint NOT NULL,
    speed real
);



CREATE TABLE models_alpha_4 (
    model_id integer NOT NULL,
    angle smallint NOT NULL,
    pressure_coefficients BLOB NOT NULL
);





CREATE TABLE models_alpha_6 (
    model_id integer NOT NULL,
    angle smallint NOT NULL,
    pressure_coefficients BLOB NOT NULL
);



CREATE TABLE non_isolated_building (
    id_area_density smallint,
    id_arrange_order smallint,
    breadth smallint,
    depth smallint,
    height smallint,
    id_x_coordinates smallint,
    id_y_coordinates smallint,
    id_surface smallint,
    id_pitch smallint,
    id_roof smallint,
    id_sample_frequency smallint,
    id_sample_period smallint,
    id_surrounding_height smallint,
    id_wind_azimuth smallint,
    pressure_coefficients BLOB
);



CREATE TABLE roof_pitches (
    id_pitch smallint NOT NULL,
    pitch real
);



CREATE TABLE roof_types (
    id_roof smallint NOT NULL,
    roof_type character varying(11)
);



CREATE TABLE sample_frequencies (
    id_sample_frequency smallint NOT NULL,
    sample_frequency real
);



CREATE TABLE sample_periods (
    id_sample_period smallint NOT NULL,
    period real
);



CREATE TABLE surfaces_roof (
    id_surface smallint NOT NULL,
    surface BLOB
);



CREATE TABLE surrounding_heights (
    id_surrounding_height smallint NOT NULL,
    surrounding_height real
);



CREATE TABLE wind_azimuths (
    id_wind_azimuth smallint NOT NULL,
    wind_azimuth integer
);



CREATE TABLE x_coordinates_roof (
    id_x_coordinates smallint NOT NULL,
    x_coordinates BLOB
);



CREATE TABLE y_coordinates_roof (
    id_y_coordinates smallint NOT NULL,
    y_coordinates BLOB
);



END;
