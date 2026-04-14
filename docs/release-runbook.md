# Release Runbook

## Goal

Ship the production stack with a repeatable process that covers preflight, deploy, health verification, smoke checks, and rollback.

## Preconditions

- Local branch is synced to the intended release ref.
- `PYTHONPATH=Project ./.venv310/bin/pytest -q tests` passes on Python 3.10.
- `terraform -chdir=terraform/aws validate` passes.
- `docker compose config` resolves cleanly for the target env file.
- AWS credentials and `terraform/aws/terraform.tfvars` are configured.

## Release Inputs

- Git ref to deploy: branch or tag
- Target environment: `staging` or `prod`
- Artifact bucket name
- Admin CIDR allowlist
- API token value
- Database URL

## Preflight

1. Confirm repo status is clean for the files being released.
2. Confirm production templates are populated:
   - `.env.prod`
   - `terraform/aws/terraform.tfvars`
3. Run:
   ```bash
   PYTHONPATH=Project ./.venv310/bin/pytest -q tests
   terraform -chdir=terraform/aws fmt -check
   terraform -chdir=terraform/aws validate
   ```
4. Record the current deployed git ref and Terraform state snapshot.

## Deploy

### AWS Infrastructure

```bash
cd terraform/aws
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

Capture these outputs:

- `frontend_url`
- `api_admin_url`
- `streamlit_admin_url`
- `public_ip`

### Application Update On Host

If Terraform created the host for the first time, cloud-init will clone the repo and start the stack automatically.

For an in-place update on an existing host:

```bash
ssh ubuntu@<public_ip>
cd /opt/time-series-forecast/repo
git fetch --all --tags
git checkout <release-ref>
git pull --ff-only origin <release-ref>
docker compose --project-directory . --env-file .env.prod -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.prod.yml up -d --build
```

## Health Verification

Run these checks after deployment:

```bash
curl -f http://<public_ip>/health
curl -f http://<public_ip>:8002/health/ready
curl -f http://<public_ip>:8002/models
curl -f http://<public_ip>:8002/metrics
```

If `TSF_API_TOKEN` is set, add:

```bash
-H "Authorization: Bearer <token>"
```

## Smoke Run

Run the repo smoke script against the admin API:

```bash
API_BASE=http://<public_ip>:8002 bash scripts/test_smoke.sh
```

Expected outcome:

- health succeeds
- sync train returns a `run_id`
- `/artifacts/latest` returns the same run
- `/models` returns at least one available model

## Rollback

Rollback conditions:

- `/health/ready` stays degraded or failing
- smoke run fails
- user-facing frontend returns 5xx or cannot reach API

Rollback procedure:

1. SSH to the host.
2. Switch back to the previous known-good ref:
   ```bash
   cd /opt/time-series-forecast/repo
   git checkout <previous-ref>
   docker compose --project-directory . --env-file .env.prod -f infra/compose/docker-compose.yml -f infra/compose/docker-compose.prod.yml up -d --build
   ```
3. Re-run health verification and smoke run.
4. If infrastructure itself is broken, use:
   ```bash
   terraform -chdir=terraform/aws apply
   ```
   against the previous reviewed Terraform commit.

## Post-Release

1. Record release ref, time, operator, and smoke result.
2. Archive Terraform plan output and deployment logs.
3. Review alerts for 30 minutes after release.
