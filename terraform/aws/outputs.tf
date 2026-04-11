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

output "public_ip" {
  description = "Elastic IP assigned to the production host."
  value       = aws_eip.app.public_ip
}

output "frontend_url" {
  description = "Public URL for the React frontend."
  value       = "http://${aws_eip.app.public_ip}"
}

output "api_admin_url" {
  description = "Direct API diagnostics URL, intended for admin CIDRs only."
  value       = "http://${aws_eip.app.public_ip}:8002"
}

output "streamlit_admin_url" {
  description = "Direct Streamlit UI URL, intended for admin CIDRs only."
  value       = "http://${aws_eip.app.public_ip}:8503"
}

output "ssh_command" {
  description = "Convenience SSH command for the production host."
  value       = var.ssh_key_name == null ? "Configure ssh_key_name to enable SSH access." : "ssh ubuntu@${aws_eip.app.public_ip}"
}
