#!/usr/bin/env bash
# Restore the reviewer archive into an EMPTY local catalog only.
set -euo pipefail
task_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_repo_root="$(cd -- "${1:-$task_script_dir/..}" && pwd)"
cd "$task_repo_root"
task_dump="$task_repo_root/modules/dinov2_retrieval/data/exports/catalog-giant-e5.dump"
test -f "$task_dump" || { echo 'Runtime dump is missing. Unpack the archive into modules/dinov2_retrieval/data/.' >&2; exit 1; }
task_compose=(docker compose -f modules/dinov2_retrieval/docker-compose.yml)
for task_attempt in {1..30}; do
  if "${task_compose[@]}" exec -T postgres pg_isready -U dinov2 -d dinov2 </dev/null >/dev/null 2>&1; then break; fi
  sleep 1
done
task_table_count="$("${task_compose[@]}" exec -T postgres psql -X -U dinov2 -d dinov2 -Atc \
  "SELECT count(*) FROM pg_tables WHERE schemaname='public';" </dev/null)"
if [[ "$task_table_count" == 1 ]]; then
  # Fresh Compose initializes exactly this empty table before restoring a dump.
  # Validate the known scaffold inside one transaction, never a populated table.
  "${task_compose[@]}" exec -T postgres psql -X -v ON_ERROR_STOP=1 -U dinov2 -d dinov2 -c '
    DO $runtime$
    DECLARE columns_signature text;
    BEGIN
      IF (SELECT array_agg(tablename::text ORDER BY tablename) FROM pg_tables WHERE schemaname = '\''public'\'')
         IS DISTINCT FROM ARRAY['\''reference_images'\'']::text[] THEN
        RAISE EXCEPTION '\''Refusing to overwrite existing tables'\'';
      END IF;
      IF EXISTS (SELECT 1 FROM reference_images LIMIT 1) THEN
        RAISE EXCEPTION '\''Refusing to overwrite a populated reference_images table'\'';
      END IF;
      SELECT string_agg(attname || '\'':'\'' || format_type(atttypid, atttypmod), '\'', '\'' ORDER BY attnum)
        INTO columns_signature FROM pg_attribute WHERE attrelid = '\''public.reference_images'\''::regclass
          AND attnum > 0 AND NOT attisdropped;
      IF columns_signature IS DISTINCT FROM '\''id:bigint, wine_id:text, slug:text, image_uri:text, model_name:text, embedding:vector(1536), created_at:timestamp with time zone, updated_at:timestamp with time zone'\'' THEN
        RAISE EXCEPTION '\''Refusing to replace an unknown initialization schema'\'';
      END IF;
      DROP TABLE public.reference_images;
    END $runtime$;' </dev/null
elif [[ "$task_table_count" != 0 ]]; then
  echo 'Refusing to overwrite a database containing tables. Use a new, empty PostgreSQL instance.' >&2
  exit 1
fi
"${task_compose[@]}" exec -T postgres pg_restore -U dinov2 -d dinov2 \
  --exit-on-error --no-owner --no-acl < "$task_dump"
"${task_compose[@]}" exec -T postgres psql -X -U dinov2 -d dinov2 -c \
  'SELECT count(*) AS wines, count(DISTINCT slug) AS slugs FROM wines; SELECT count(*) AS reference_images FROM reference_images;' </dev/null
