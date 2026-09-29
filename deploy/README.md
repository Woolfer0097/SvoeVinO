# CPU deployment on one VM

The cloud override runs PostgreSQL/pgvector, DINOv2 giant (1536), SuperPoint /
LightGlue, OCR / multilingual E5, preprocessing, the pipeline API, the production
Nuxt frontend and Caddy. It does not require a GPU. The current VM has 4 vCPUs,
16 GB RAM and an 80 GB disk. Recognition latency on this CPU is not yet measured;
these resources do not guarantee the organizer's 120-second limit.

## Runtime inputs

The deployment directory is `/home/woolfer0097/SvoeVinO` on the VM. Keep these
inputs outside Git:

- `.env.cloud`: `POSTGRES_PASSWORD`, matching `DATABASE_URL`, and
  `SITE_ADDRESS=wine.knittta.ru`. Restrict this file to mode `600`.
- The populated catalog database, including image and text embeddings. Starting
  an empty PostgreSQL instance is not enough. The initial deployment restores
  a custom-format dump of the local catalog.
- `modules/dinov2_retrieval/data/`: reference images and persistent feedback
  images. Its feedback subdirectory must be writable by UID 1000.
- `modules/dinov2_retrieval/model-cache/`: Hugging Face model cache shared by
  DINOv2 and E5. Preserve symlinks when transferring it (`rsync -a`).
- Named volumes for PostgreSQL, Caddy certificates and OCR / SuperPoint caches.

Use a strong random database password, not the development default. Never share
the environment file or include it in a public archive.

## DNS and networking

Create the DNS record `A wine -> 89.169.173.111` in the authoritative DNS zone of
`knittta.ru`. This uses a subdomain of the existing domain, not a new registration.
Leave the apex (`@`), `www`, mail records and nameservers unchanged. Remove only
conflicting records for `wine`, if any. With Cloudflare, initially select
**DNS only** for this record.

Allow inbound TCP 80 and 443 in the Yandex Cloud security group. Preserve SSH
access on 22; restrict its source if practical. PostgreSQL and model APIs have
no published host ports. The backend and frontend bind only to loopback; Caddy
is the public entry point.

Caddy obtains and renews a trusted HTTPS certificate automatically once DNS and
inbound connectivity work. Its certificate state is persistent in `caddy_data`.
HTTPS enables browser camera access on mobile devices.

The proxy uses HTTP/1.1 and HTTP/2 only. UDP 443 is not published, so advertising
HTTP/3 would point browsers at an unreachable transport. `Alt-Svc: clear`
removes HTTP/3 alternatives cached before this configuration change. Access logs
go to the rotated Docker log, without uploaded image bodies.

When replacing the single-file Caddy bind mount with an atomic file transfer,
recreate only the proxy so it sees the new file inode, then check its response
headers:

```bash
docker compose --env-file .env.cloud \
  -f compose.pipeline.yml -f compose.cloud-cpu.yml \
  up -d --no-deps --force-recreate proxy
```

## Start and operate

Run on the VM:

```bash
cd /home/woolfer0097/SvoeVinO
docker network inspect dinov2_retrieval_default >/dev/null 2>&1 \
  || docker network create dinov2_retrieval_default
docker compose --env-file .env.cloud \
  -f compose.pipeline.yml -f compose.cloud-cpu.yml config --quiet
docker compose --env-file .env.cloud \
  -f compose.pipeline.yml -f compose.cloud-cpu.yml up -d --build
docker compose --env-file .env.cloud \
  -f compose.pipeline.yml -f compose.cloud-cpu.yml ps
```

The merged configuration removes GPU reservations, builds CPU PyTorch wheels,
and switches DINOv2 to float32. Log rotation is enabled. Containers restart after
a Docker / VM restart. Do not run the base compose file alone on this CPU VM.

```bash
# Recent logs; do not publish logs that contain user data.
docker compose --env-file .env.cloud \
  -f compose.pipeline.yml -f compose.cloud-cpu.yml logs --tail=100 backend proxy

# Resource usage.
docker stats --no-stream
```

Once HTTPS is available:

- Website: `https://wine.knittta.ru/`
- Swagger: `https://wine.knittta.ru/docs`
- Evaluation endpoint: `https://wine.knittta.ru/v1/eval/predict`

The evaluation endpoint accepts multipart field `image` and returns Top-1
`slug`. The website uses the asynchronous API and polls processing status.
Readiness checks do not measure recognition quality or evaluation latency.

Until DNS is ready, use an SSH tunnel from your computer:

```bash
ssh -N -L 13000:127.0.0.1:3000 -L 18080:127.0.0.1:8080 \
  woolfer0097@89.169.173.111
```

Open `http://localhost:13000/` for the website or
`http://localhost:18080/docs` for Swagger. Leave the SSH session running.

## Backup and budget

Before updating the deployment, back up the catalog and feedback images:

```bash
cd /home/woolfer0097/SvoeVinO
umask 077
docker compose --env-file .env.cloud \
  -f compose.pipeline.yml -f compose.cloud-cpu.yml \
  exec -T postgres pg_dump -U dinov2 -d dinov2 -Fc \
  > "catalog-$(date +%Y%m%d-%H%M%S).dump"
```

Also copy `modules/dinov2_retrieval/data/reference/feedback/` to backup storage;
the database dump alone does not contain image files. Keep backups private.
Do not use `docker compose down -v`: it removes persistent database / certificate
volumes. Running `down` or stopping Docker containers does not stop VM billing.

Configure a budget alert in Yandex Cloud and review the deployment on October 14.
No automatic VM shutdown has been scheduled. A stopped VM can still incur disk
and reserved-IP charges; review those resources separately before removing them.
