# Terraform

`terraform/aws` now provisions a runnable single-host AWS production baseline for this repository:

- VPC with public subnets and internet gateway
- security group with separate public and admin ingress rules
- encrypted/versioned S3 bucket for artifacts
- CloudWatch log group
- SSM-enabled EC2 instance with Docker and Docker Compose
- bootstrapping that clones this repository and starts `docker-compose.prod.yml`

## Prerequisites

- Terraform `>= 1.5`
- AWS credentials with permissions for EC2, IAM, VPC, S3, CloudWatch, and EIP
- An existing EC2 key pair if you want SSH access

## Quickstart

```bash
cd terraform/aws
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

## Required customization

- Set `artifacts_bucket_name` to a globally unique value
- Restrict `admin_cidr_blocks` to your office/VPN/home IPs
- Set `ssh_key_name` if you want SSH access
- Override `repository_ref` if you deploy a release tag instead of `main`

## Result

After `apply`, Terraform outputs the public frontend URL, the admin-only API URL, the admin-only Streamlit URL, and the host public IP.
