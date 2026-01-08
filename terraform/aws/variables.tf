variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "artifacts_bucket_name" {
  description = "Globally unique S3 bucket name for model artifacts."
  type        = string
}
