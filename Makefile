ps:
	docker compose ps
build:
	docker compose up -d --build
up:
	docker compose up -d
down:
	docker compose down
stop:
	docker compose stop
node:
	docker compose exec app sh
db:
	docker compose exec db bash
dev:
	docker compose exec app poetry run poe dev
lock:
	docker compose exec app poetry lock
install:
	docker compose exec app poetry install
