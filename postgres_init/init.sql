DO $$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles WHERE rolname = 'boxia_user'
   ) THEN
      CREATE USER boxia_user WITH ENCRYPTED PASSWORD 'boxia';
   END IF;
END
$$;

CREATE DATABASE boxia_db OWNER boxia_user;

\connect boxia_db

CREATE TABLE IF NOT EXISTS reported_questions (
    id SERIAL PRIMARY KEY,
    question VARCHAR,
    answer VARCHAR,
    expert_answer TEXT,
    reported_date TIMESTAMP,
    checked VARCHAR(20)
);
