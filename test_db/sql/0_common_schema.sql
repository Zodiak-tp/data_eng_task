REVOKE CREATE ON SCHEMA public FROM PUBLIC;

CREATE SCHEMA common;
SET search_path TO common;

CREATE TABLE users (
  user_id    SERIAL PRIMARY KEY,
  user_name  TEXT
);

CREATE TABLE customer_meta  (
  meta_id      SERIAL PRIMARY KEY,
  kind         TEXT,
  lat          FLOAT,
  lon          FLOAT,
  seats        INT,
  price_level  INT
);

CREATE TABLE articles (
  article_id     SERIAL PRIMARY KEY,
  article_number INT,
  price          FLOAT,
  year           INT
);
