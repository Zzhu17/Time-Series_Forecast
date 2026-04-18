variable "project_name" {
  description = "Project slug used in resource names."
  type        = string
  default     = "time-series-forecast"
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
  default     = "prod"
}

variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "CIDR block for the production VPC."
  type        = string
  default     = "10.42.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDRs. Use at least two for AZ resilience."
  type        = list(string)
  default     = ["10.42.0.0/24", "10.42.1.0/24"]
}

variable "availability_zones" {
  description = "Optional explicit AZ names matching public_subnet_cidrs."
  type        = list(string)
  default     = []
}

variable "app_ingress_cidrs" {
  description = "CIDR blocks allowed to reach the public frontend."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "admin_cidr_blocks" {
  description = "CIDR blocks allowed to reach SSH, admin Streamlit, and direct API diagnostics."
  type        = list(string)
  default     = []

  validation {
    condition = length(var.admin_cidr_blocks) > 0 && alltrue([
      for cidr in var.admin_cidr_blocks : cidr != "0.0.0.0/0"
    ])
    error_message = "admin_cidr_blocks must be explicitly set and must not contain 0.0.0.0/0."
  }
}

variable "instance_type" {
  description = "EC2 instance type for the single-host production deployment."
  type        = string
  default     = "t3.large"
}

variable "root_volume_size_gb" {
  description = "Root EBS volume size in GiB."
  type        = number
  default     = 64
}

variable "ssh_key_name" {
  description = "Optional existing EC2 key pair name for SSH."
  type        = string
  default     = null
}

variable "artifacts_bucket_name" {
  description = "Globally unique S3 bucket name for artifacts and reports."
  type        = string
}

variable "artifacts_noncurrent_retention_days" {
  description = "Retention in days for non-current S3 object versions."
  type        = number
  default     = 30
}

variable "log_retention_in_days" {
  description = "CloudWatch log retention for instance/app logs."
  type        = number
  default     = 30
}

variable "repository_clone_url" {
  description = "Git clone URL used by cloud-init to fetch the application code."
  type        = string
  default     = "https://github.com/Zzhu17/Time-Series_Forecast.git"
}

variable "repository_ref" {
  description = "Git branch or tag to deploy."
  type        = string
  default     = "main"
}

variable "tsf_api_token_ssm_parameter_name" {
  description = "Name of a pre-created SSM SecureString parameter that stores the API bearer token."
  type        = string
  default     = ""

  validation {
    condition     = length(trimspace(var.tsf_api_token_ssm_parameter_name)) > 0
    error_message = "tsf_api_token_ssm_parameter_name must be set to an existing SSM SecureString parameter name."
  }
}

variable "database_url" {
  description = "DATABASE_URL value injected into .env.prod on the instance."
  type        = string
  default     = "sqlite:///Project/output/tasks_prod.db"
}

variable "celery_enabled" {
  description = "Whether the production stack should start the Celery worker."
  type        = bool
  default     = true
}

variable "celery_broker_url" {
  description = "Broker URL injected into .env.prod."
  type        = string
  default     = "redis://redis:6379/0"
}

variable "celery_result_backend" {
  description = "Result backend URL injected into .env.prod."
  type        = string
  default     = "redis://redis:6379/0"
}
