# Set-up the PostgreSQL and jupyter for data engineering test

The repo contains a set-up for a PostgreSQL database which contains some demo POS data.
Further a and jupyter notebook server is configured which can be started to use notebooks.
Both, the database and the notebook server, can be started with docker.

## Setup with Docker

Docker and docker-compose need to be installed. To start the services run

`docker-compose up --build`

Test running jupyter server by accessing `http://localhost:3000/` with password `data`.
Test running database by accessing the database via `psql -h 0.0.0.0 -p 5430 -U data_engineer test_db`.
