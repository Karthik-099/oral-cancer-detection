terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  backend "s3" {
    bucket = "oralaid-tfstate"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Variables ─────────────────────────────────────────────────────────────────
variable "aws_region"    { default = "us-east-1" }
variable "env"           { default = "prod" }
variable "db_password"   { sensitive = true }
variable "secret_key"    { sensitive = true }

locals {
  prefix = "oralaid-${var.env}"
  tags   = { Project = "OralAID", Env = var.env, ManagedBy = "Terraform" }
}

# ── VPC ───────────────────────────────────────────────────────────────────────
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  tags = merge(local.tags, { Name = "${local.prefix}-vpc" })
}

resource "aws_subnet" "public_a" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true
  tags = merge(local.tags, { Name = "${local.prefix}-public-a" })
}

resource "aws_subnet" "public_b" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = true
  tags = merge(local.tags, { Name = "${local.prefix}-public-b" })
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = merge(local.tags, { Name = "${local.prefix}-igw" })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route { cidr_block = "0.0.0.0/0"; gateway_id = aws_internet_gateway.igw.id }
  tags = merge(local.tags, { Name = "${local.prefix}-rt-public" })
}

resource "aws_route_table_association" "a" {
  subnet_id      = aws_subnet.public_a.id
  route_table_id = aws_route_table.public.id
}
resource "aws_route_table_association" "b" {
  subnet_id      = aws_subnet.public_b.id
  route_table_id = aws_route_table.public.id
}

# ── Security Groups ───────────────────────────────────────────────────────────
resource "aws_security_group" "alb" {
  name   = "${local.prefix}-alb-sg"
  vpc_id = aws_vpc.main.id
  ingress { from_port=80;  to_port=80;  protocol="tcp"; cidr_blocks=["0.0.0.0/0"] }
  ingress { from_port=443; to_port=443; protocol="tcp"; cidr_blocks=["0.0.0.0/0"] }
  egress  { from_port=0;   to_port=0;   protocol="-1";  cidr_blocks=["0.0.0.0/0"] }
  tags = merge(local.tags, { Name = "${local.prefix}-alb-sg" })
}

resource "aws_security_group" "app" {
  name   = "${local.prefix}-app-sg"
  vpc_id = aws_vpc.main.id
  ingress { from_port=5000; to_port=5000; protocol="tcp"; security_groups=[aws_security_group.alb.id] }
  ingress { from_port=3000; to_port=3000; protocol="tcp"; cidr_blocks=["0.0.0.0/0"] }  # Grafana
  ingress { from_port=9090; to_port=9090; protocol="tcp"; cidr_blocks=["0.0.0.0/0"] }  # Prometheus
  ingress { from_port=22;   to_port=22;   protocol="tcp"; cidr_blocks=["0.0.0.0/0"] }  # SSH
  egress  { from_port=0;    to_port=0;    protocol="-1";  cidr_blocks=["0.0.0.0/0"] }
  tags = merge(local.tags, { Name = "${local.prefix}-app-sg" })
}

# ── ECR Repository ────────────────────────────────────────────────────────────
resource "aws_ecr_repository" "oralaid" {
  name                 = "oralaid"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
  tags = local.tags
}

# ── S3 Bucket (uploads + models) ─────────────────────────────────────────────
resource "aws_s3_bucket" "uploads" {
  bucket = "${local.prefix}-uploads"
  tags   = local.tags
}

resource "aws_s3_bucket_versioning" "uploads" {
  bucket = aws_s3_bucket.uploads.id
  versioning_configuration { status = "Enabled" }
}

# ── RDS PostgreSQL ────────────────────────────────────────────────────────────
resource "aws_db_subnet_group" "main" {
  name       = "${local.prefix}-db-subnet"
  subnet_ids = [aws_subnet.public_a.id, aws_subnet.public_b.id]
  tags       = local.tags
}

resource "aws_db_instance" "postgres" {
  identifier             = "${local.prefix}-db"
  engine                 = "postgres"
  engine_version         = "15.4"
  instance_class         = "db.t3.micro"
  allocated_storage      = 20
  db_name                = "oralaid"
  username               = "oralaid"
  password               = var.db_password
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.app.id]
  skip_final_snapshot    = true
  tags                   = local.tags
}

# ── ECS Cluster ───────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "main" {
  name = "${local.prefix}-cluster"
  setting { name = "containerInsights"; value = "enabled" }
  tags = local.tags
}

# ── IAM Role for ECS Task ─────────────────────────────────────────────────────
resource "aws_iam_role" "ecs_task" {
  name = "${local.prefix}-ecs-task-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole"; Effect = "Allow"; Principal = { Service = "ecs-tasks.amazonaws.com" } }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "ecs_task_exec" {
  role       = aws_iam_role.ecs_task.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_s3" {
  name = "${local.prefix}-ecs-s3"
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow"; Action = ["s3:*"]; Resource = ["${aws_s3_bucket.uploads.arn}/*", aws_s3_bucket.uploads.arn] }]
  })
}

# ── ECS Task Definition ───────────────────────────────────────────────────────
resource "aws_ecs_task_definition" "app" {
  family                   = "${local.prefix}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "2048"
  memory                   = "4096"
  execution_role_arn       = aws_iam_role.ecs_task.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = "oralaid"
    image     = "${aws_ecr_repository.oralaid.repository_url}:latest"
    essential = true
    portMappings = [{ containerPort = 5000; protocol = "tcp" }]
    environment = [
      { name = "DATABASE_URL"; value = "postgresql://oralaid:${var.db_password}@${aws_db_instance.postgres.endpoint}/oralaid" },
      { name = "SECRET_KEY";   value = var.secret_key },
      { name = "LOG_LEVEL";    value = "INFO" },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/${local.prefix}"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:5000/metrics')\""]
      interval    = 30
      timeout     = 10
      retries     = 3
      startPeriod = 60
    }
  }])
  tags = local.tags
}

# ── ALB ───────────────────────────────────────────────────────────────────────
resource "aws_lb" "main" {
  name               = "${local.prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = [aws_subnet.public_a.id, aws_subnet.public_b.id]
  tags               = local.tags
}

resource "aws_lb_target_group" "app" {
  name        = "${local.prefix}-tg"
  port        = 5000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"
  health_check { path = "/metrics"; healthy_threshold = 2; unhealthy_threshold = 3; interval = 30 }
  tags = local.tags
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"
  default_action { type = "forward"; target_group_arn = aws_lb_target_group.app.arn }
}

# ── ECS Service ───────────────────────────────────────────────────────────────
resource "aws_ecs_service" "app" {
  name            = "${local.prefix}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = [aws_subnet.public_a.id, aws_subnet.public_b.id]
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "oralaid"
    container_port   = 5000
  }

  deployment_circuit_breaker { enable = true; rollback = true }
  tags = local.tags
}

# ── CloudWatch Log Group ──────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${local.prefix}"
  retention_in_days = 30
  tags              = local.tags
}

# ── Outputs ───────────────────────────────────────────────────────────────────
output "alb_dns"        { value = aws_lb.main.dns_name }
output "ecr_url"        { value = aws_ecr_repository.oralaid.repository_url }
output "db_endpoint"    { value = aws_db_instance.postgres.endpoint }
output "ecs_cluster"    { value = aws_ecs_cluster.main.name }
output "app_url"        { value = "http://${aws_lb.main.dns_name}" }
