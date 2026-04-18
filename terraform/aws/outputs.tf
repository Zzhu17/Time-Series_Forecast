output "vpc_id" {
  description = "VPC ID for the deployment."
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs."
  value       = values(aws_subnet.public)[*].id
}

output "security_group_id" {
  description = "Security group attached to the production host."
  value       = aws_security_group.app.id
}

output "artifacts_bucket_name" {
  description = "S3 bucket used for artifacts."
  value       = aws_s3_bucket.artifacts.bucket
}

output "instance_id" {
  description = "EC2 instance ID running the production stack."
  value       = aws_instance.app.id
}
