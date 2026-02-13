# The "Iron Bank" Isolation Layer
resource "aws_vpc" "sovereign_ai_vpc" {
  cidr_block = "10.0.0.0/16"
  enable_dns_support = true
  tags = {
    Name = "Adam-Sovereign-VPC"
    Compliance = "GLBA-Strict"
  }
}

# Strictly Private Subnet for Vector DB
resource "aws_subnet" "vector_db_subnet" {
  vpc_id     = aws_vpc.sovereign_ai_vpc.id
  cidr_block = "10.0.1.0/24"
  # CRITICAL: No public IP assignment
  map_public_ip_on_launch = false
}
