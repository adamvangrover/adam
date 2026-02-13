output "vpc_id" {
  description = "The ID of the Sovereign AI VPC"
  value       = aws_vpc.sovereign_ai_vpc.id
}

output "vector_db_subnet_id" {
  description = "The ID of the private subnet for Vector DB"
  value       = aws_subnet.vector_db_subnet.id
}
