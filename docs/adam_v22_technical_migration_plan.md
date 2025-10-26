# Adam v22.0: Technical Migration Plan
**Author:** Principal Architect
**Source Blueprint:** Adam System Evolution: A Comparative Analysis of v21.0 and v22.0
**Guiding Principle:** Strangler Fig Pattern

## 1. Executive Migration Strategy
Our migration will follow the Strangler Fig Pattern, as specified in the blueprint. This is a phased, risk-averse approach. We will not conduct a "big bang" rewrite. Instead, we will build a new, modern "trellis" around the legacy monolith and incrementally "strangle" old functionality by routing traffic to new, purpose-built microservices.

### Key Milestones:
**Phase 1: Foundation (The Trellis)**
*   Provision all new v22.0 infrastructure as code (IaC).
*   Deploy a new Kubernetes (K8s) cluster.
*   Deploy the event backbone (Apache Kafka) and caching layer (Redis).
*   Deploy the new polyglot databases (MongoDB).
*   Establish the new CI/CD pipeline targeting this infrastructure.

**Phase 2: The Facade (The Fig Vine)**
*   Deploy an API Gateway (e.g., Kong, Apigee, AWS API Gateway) in front of the entire Adam platform.
*   **Crucial Step:** All public traffic to the v21.0 monolith MUST be routed through this gateway.
*   Implement the new, mandatory OAuth 2.0 authentication at this gateway layer. This immediately addresses the breaking security change and centralizes auth logic.

**Phase 3: First New Service (The First Leaf)**
*   Develop and deploy the first net-new feature, "Project Phoenix," onto the new v22.0 stack.
*   This service will run in K8s, use the new stack (e.g., Go, Python), and be exposed via the new GraphQL API endpoint on the gateway.
*   This proves the viability of the new stack in production without touching the monolith.

**Phase 4: Core Service Strangulation (The Strangulation)**
*   Identify and "strangle" the first piece of monolith functionality: FTP Data Ingestion.
*   Build a new `data-ingestion-service` (microservice).
*   This service will replace the FTP poller with a new API-driven (or Kafka-driven) flow.
*   Configure the API Gateway to route all data ingestion calls to this new service. The legacy FTP code is now deprecated and "dead."

**Phase 5: Iterative Refactoring & Strangulation**
*   Continue strangling monolith components:
    *   **Static Reports:** Build a new `reporting-service` to replace the "Static Report Generator." This service will consume data from Kafka to build its own views (CQRS pattern).
    *   **Data Access:** Refactor performance-critical read paths in the monolith to use the new Redis cache, as specified in the blueprint.
*   Incrementally move functionality from the monolith to new microservices until the legacy system is reduced to its stable, core (or is gone entirely).

## 2. Target v22.0 Architecture & Repository Structure

### Target Architecture (Mermaid Diagram)
This diagram illustrates the v22.0 hybrid state, with the API Gateway acting as the central router between the legacy and modern stacks.

```mermaid
graph TD
    subgraph External Users
        direction TB
        User[Browser/Mobile Client]
        Partner[3rd Party Integration]
    end
    subgraph Platform Boundary
        direction TB
        Gateway[API Gateway (OAuth 2.0)]
    end
    subgraph New v22.0 Stack (Kubernetes)
        direction TB
        GraphQL[GraphQL API Layer]
        Phoenix["svc-project-phoenix (Go/Python)"]
        Ingest["svc-data-ingestion (Java/Python)"]
        Report["svc-reporting (Java/Go)"]

        subgraph Internal Comms (gRPC + mTLS)
            Phoenix <-->|gRPC| Report
            Ingest <-->|gRPC| Phoenix
        end

        subgraph v22.0 Data Stores
            Mongo[MongoDB]
            Redis[Redis Cache]
        end
    end
    subgraph Legacy v21.0 Stack (Monolith)
        direction TB
        Monolith[Adam v21.0 Monolith (Java/Spring)]
        LegacyDB[PostgreSQL (Core DB)]
    end

    subgraph Event Backbone
        direction TB
        Kafka[Apache Kafka]
    end

    %% --- Flows ---
    User -->|GraphQL| Gateway
    Partner -->|REST/GraphQL| Gateway
    Gateway -->|GraphQL| GraphQL
    Gateway -->|Legacy REST| Monolith
    Gateway -->|New REST| Ingest
    GraphQL --> Phoenix
    Monolith -->|Write| LegacyDB
    Monolith -->|Read/Write (Refactored)| Redis
    Monolith -.->|Produce Events| Kafka
    Ingest -->|Produce Events| Kafka
    Kafka -->|Consume Events| Phoenix
    Kafka -->|Consume Events| Report
    Phoenix -->|Write| Mongo
    Report -->|Write/Read| Mongo
```

### New Repository Structure
We will adopt a monorepo structure to facilitate shared libraries (SDKs, protobufs), discoverability, and unified CI/CD.

```
/adam-v22/
├── .gitlab-ci.yml        # Root CI/CD pipeline
│
├── apps/                 # All deployable applications
│   ├── monolith-v21/       # The existing v21.0 monolith (will be refactored)
│   ├── svc-project-phoenix/  # New service (e.g., Go)
│   ├── svc-data-ingestion/   # New service (e.g., Python/Kafka)
│   ├── svc-reporting/        # New service (e.g., Java)
│   └── api-gateway/          # Gateway configuration (e.g., Kong config)
│
├── infra/                # Infrastructure as Code
│   ├── terraform/          # Terraform for K8s, Kafka, DBs
│   ├── docker/             # Base Dockerfiles
│   └── kubernetes/         # K8s manifests (Deployments, Services)
│
├── libs/                 # Shared libraries
│   ├── adam-sdk-python/    # Python SDK for external devs
│   ├── adam-sdk-js/          # JavaScript SDK for external devs
│   └── internal-protos/      # gRPC .proto files for internal services
│
└── docs/                 # Documentation (API specs, architecture)
```

## 3. Infrastructure & CI/CD Pipeline (IaC)

### Kubernetes: Sample `deployment.yaml`
This manifest for `svc-project-phoenix` demonstrates HPA-readiness (by setting resource requests) and resilience (with probes), as required by the blueprint.

```yaml
# infra/kubernetes/svc-project-phoenix.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: svc-project-phoenix
  labels:
    app: project-phoenix
spec:
  replicas: 3 # Start with 3, HPA will manage this
  selector:
    matchLabels:
      app: project-phoenix
  template:
    metadata:
      labels:
        app: project-phoenix
    spec:
      containers:
      - name: phoenix-server
        image: our-registry.com/svc-project-phoenix:v1.0.0
        ports:
        - containerPort: 8080 # gRPC port
        resources:
          requests: # Required for Horizontal Pod Autoscaler (HPA)
            cpu: "250m"
            memory: "256Mi"
          limits:
            cpu: "1000m"
            memory: "1024Mi"
        livenessProbe:
          grpc:
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 20
        readinessProbe:
          grpc:
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: svc-project-phoenix
spec:
  selector:
    app: project-phoenix
  ports:
    - protocol: TCP
      port: 8080 # Service port (for gRPC)
      targetPort: 8080
  type: ClusterIP # Internal service only
```

### Event Backbone: Local Development `docker-compose.yaml`
This file allows developers to spin up the v22.0 event-driven stack locally.

```yaml
# /docker-compose.yaml (for local dev)
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.3.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:7.3.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

### CI/CD: Pseudo-code `.gitlab-ci.yml`
This pipeline is path-aware, only building and deploying services that have changed.

```yaml
# /.gitlab-ci.yml
stages:
  - build
  - test
  - scan
  - containerize
  - deploy

.service-rules: &service-rules
  rules:
    # Only run job if files changed in the specific app's directory
    - changes:
        - $CI_PROJECT_DIR/apps/$SERVICE_NAME/**/*
      when: always
    - when: manual # Allow manual trigger

build:phoenix:
  stage: build
  variables:
    SERVICE_NAME: svc-project-phoenix
  <<: *service-rules
  script:
    - echo "Building $SERVICE_NAME..."
    - cd apps/$SERVICE_NAME
    # (script to build Go binary)

test:phoenix:
  stage: test
  variables:
    SERVICE_NAME: svc-project-phoenix
  <<: *service-rules
  script:
    - echo "Testing $SERVICE_NAME..."
    - cd apps/$SERVICE_NAME
    # (script to run Go tests)

scan:phoenix:
  stage: scan
  variables:
    SERVICE_NAME: svc-project-phoenix
  <<: *service-rules
  script:
    - echo "Scanning $SERVICE_NAME..."
    # (Run SAST/vulnerability scanner)

containerize:phoenix:
  stage: containerize
  variables:
    SERVICE_NAME: svc-project-phoenix
  rules:
    - if: $CI_COMMIT_BRANCH == 'main'
      changes:
        - $CI_PROJECT_DIR/apps/$SERVICE_NAME/**/*
  script:
    - echo "Building container for $SERVICE_NAME..."
    # (docker build, docker push to registry)

deploy:k8s:phoenix:
  stage: deploy
  variables:
    SERVICE_NAME: svc-project-phoenix
  rules:
    - if: $CI_COMMIT_BRANCH == 'main'
      changes:
        - $CI_PROJECT_DIR/apps/$SERVICE_NAME/**/*
  script:
    - echo "Deploying $SERVICE_NAME to K8s..."
    - kubectl apply -f infra/kubernetes/$SERVICE_NAME.yaml
```

## 4. Security & Authentication Overhaul (Mandatory)
This is a breaking change and the highest-priority security task.

### Password Hashing: SHA-1 (v21.0) vs. Argon2 (v22.0)
We must migrate user credentials from the weak SHA-1 to the modern, memory-hard Argon2, as specified.

**Before (v21.0 - `monolith-v21/.../SecurityService.java` - DANGEROUS):**

```java
// DO NOT USE THIS - VULNERABLE v21.0 EXAMPLE
import java.security.MessageDigest;
// ...
public String hashPassword_SHA1(String password) {
    MessageDigest md = MessageDigest.getInstance("SHA-1");
    md.update(password.getBytes());
    byte[] bytes = md.digest();
    // ... (convert bytes to hex string) ...
    return hexString;
}
```

**After (v22.0 - `monolith-v21/.../SecurityService.java` - MIGRATED):**
We will use a trusted Argon2 library. This code will replace the v21.0 logic inside the monolith as part of Phase 2.

```java
// MIGRATED v22.0 EXAMPLE
import de.mkammerer.argon2.Argon2;
import de.mkammerer.argon2.Argon2Factory;
// ...
// Create a thread-safe, singleton instance
private final Argon2 argon2 = Argon2Factory.create(
    Argon2Factory.Argon2Types.ARGON2id, 16, 32);

public String hashPassword_Argon2(String password) {
    // Hash: iterations=10, memory=65536KB, parallelism=1
    return argon2.hash(10, 65536, 1, password.toCharArray());
}

public boolean verifyPassword(String hash, String password) {
    return argon2.verify(hash, password.toCharArray());
}

// MIGRATION LOGIC: During login, if a user's hash is SHA-1,
// verify it, and if successful, immediately re-hash with Argon2
// and update the database.
```

### API Authentication: Static Key (v21.0) vs. OAuth 2.0 (v22.0)
This logic will be implemented in the new API Gateway (Phase 2) and will also require refactoring the monolith's security configuration.

**Before (v21.0 - `monolith-v21/.../ApiController.java`):**

```java
// v21.0: Simple, insecure header check
@RestController
@RequestMapping("/api/v1")
public class LegacyController {

    @GetMapping("/data")
    public ResponseEntity<String> getData(
        @RequestHeader("X-API-KEY") String apiKey
    ) {
        if (!authService.isValid(apiKey)) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        }
        // ... logic ...
    }
}
```

**After (v22.0 - `monolith-v21/.../SecurityConfig.java`):**
The monolith is refactored to be a "Resource Server," validating JWTs issued by the new auth provider (managed at the Gateway).

```java
// v22.0: Monolith secured as an OAuth 2.0 Resource Server
@Configuration
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {

    @Value("${spring.security.oauth2.resourceserver.jwt.jwk-set-uri}")
    private String jwkSetUri;

    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http
            .authorizeRequests(authorize -> authorize
                // Secure all /api/v1 endpoints
                .antMatchers("/api/v1/**").authenticated()
                .anyRequest().permitAll() // Allow public access to non-api
            )
            .oauth2ResourceServer(oauth2 -> oauth2
                // Use JWT validation
                .jwt(jwt -> jwt.decoder(jwtDecoder()))
            );
    }

    @Bean
    JwtDecoder jwtDecoder() {
        // Configure the decoder to fetch public keys from our Auth Server
        return NimbusJwtDecoder.withJwkSetUri(this.jwkSetUri).build();
    }
}
```

And in `application.properties`:

```properties
# Point to the new centralized Auth Server (Okta, Auth0, Keycloak, or self-built)
spring.security.oauth2.resourceserver.jwt.jwk-set-uri=https://auth.adam-v22.com/.well-known/jwks.json
```

### Internal Security: mTLS
As per the blueprint, all internal K8s traffic will use mTLS for a zero-trust environment. We will implement this using a service mesh (e.g., Istio).

**Conceptual Configuration (Istio `PeerAuthentication`):**
This policy will be applied to our K8s cluster to enforce that all services within the `adam-services` namespace must use mutual TLS.

```yaml
# infra/kubernetes/mtls-policy.yaml
apiVersion: "security.istio.io/v1beta1"
kind: "PeerAuthentication"
metadata:
  name: "default"
  namespace: "adam-services" # The namespace for our new microservices
spec:
  mtls:
    mode: STRICT # Enforce mTLS for all traffic
```

## 5. Monolith Strangulation & Refactoring Plan

### Data Ingestion (FTP Deprecation)
We will build `svc-data-ingestion` to replace the insecure FTP poller.

**Before (v21.0 - `monolith-v21/.../FtpPoller.java` - Hypothetical):**

```java
// v21.0: Fragile, insecure FTP polling
import org.apache.commons.net.ftp.FTPClient;
// ...
@Scheduled(fixedDelay = 60000) // Poll every 60 seconds
public void pollFtpDirectory() {
    FTPClient ftp = new FTPClient();
    try {
        ftp.connect("ftp.partner.com");
        ftp.login("user", "pass");
        // ... logic to list files, download, parse, and save to DB ...
    } finally {
        if (ftp.isConnected()) {
            ftp.disconnect();
        }
    }
}
```

**After (v22.0 - `apps/svc-data-ingestion/producer.py` - Python/Kafka):**
This new service provides a secure API endpoint. When it receives data, it produces a message to Kafka, achieving the 1400% throughput increase cited in the blueprint.

```python
# v22.0: New Kafka Producer (part of a Flask/FastAPI service)
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='kafka:29092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# This function is called by a new, secure REST endpoint
def handle_api_ingestion(record_batch):
    for record in record_batch:
        print(f"Producing record: {record['id']}")
        # Publish to the new high-throughput topic
        producer.send('data-ingestion-topic', record)
    producer.flush()
    print(f"Batch of {len(record_batch)} records produced.")
```

### Data Access (Polyglot Persistence)
We will refactor the monolith's data access to offload reads to Redis and decouple writes.

**Before (v21.0 - `monolith-v21/.../ComplexQueryRepository.java`):**

```java
// v21.0: Heavy, synchronous query direct to PostgreSQL
public interface ComplexQueryRepository extends JpaRepository<ReportData, Long> {

    @Query("SELECT r FROM ReportData r JOIN r.user u JOIN r.details d " +
           "WHERE u.region = :region AND r.createdAt > :startDate " +
           "ORDER BY r.value DESC")
    List<ReportData> findComplexReportData(String region, Date startDate);
}
```

**After (v22.0 - Refactored):**

**1. Read Offload (Redis Cache):** We refactor the monolith itself to use Redis for caching, reducing its memory/CPU footprint as specified.

```java
// v22.0: The *same* repository method, now cached with Redis
// This is enabled by adding @EnableCaching and Redis config
public interface ComplexQueryRepository extends JpaRepository<ReportData, Long> {

    // Add caching; Spring Boot handles the rest
    @Cacheable(value = "reports", key = "{#region, #startDate}")
    @Query("SELECT r FROM ReportData r JOIN r.user u JOIN r.details d " +
           "WHERE u.region = :region AND r.createdAt > :startDate " +
           "ORDER BY r.value DESC")
    List<ReportData> findComplexReportData(String region, Date startDate);
}
```

**2. Write Decoupling (Kafka Producer):** When data is written in the monolith, we now also publish an event to Kafka. This allows new v22.0 services (like `svc-reporting`) to consume this change without coupling to the monolith's DB.

```java
// v22.0: Inside a monolith @Service method
@Autowired
private KafkaTemplate<String, ReportData> kafkaTemplate;

public void saveNewReportData(ReportData data) {
    // 1. Save to legacy DB for monolith to function
    legacyRepo.save(data);

    // 2. Publish an event for v22.0 services to consume
    // This decouples the monolith from new services
    kafkaTemplate.send("report-data-updated", data.getId().toString(), data);
}
```

### Reporting (Static Report Deprecation)
The "Static Report Generator" will be deprecated and replaced by the new `svc-reporting` microservice.

*   **Source Data:** This new service will **not** query the legacy PostgreSQL database.
*   **New Pattern (CQRS):** It will consume events from Kafka (e.g., `data-ingestion-topic`, `report-data-updated`) to build its own optimized, materialized data views in MongoDB or its own PostgreSQL schema.
*   **Function:** It will power the new "enhanced reporting module" by serving data from these optimized views, allowing it to be far more performant and flexible than the old static generator.

## 6. New API Layer & Developer Ecosystem

### GraphQL API: "Project Phoenix" Schema
This schema (in SDL) provides the new, efficient query language for external developers.

```graphql
# apps/svc-project-phoenix/schema.graphql
# The new real-time analytics dashboard as specified

type Query {
  """
  Query real-time data for Project Phoenix.
  Requires OAuth 2.0 scope: 'phoenix:read'
  """
  projectPhoenix(filter: PhoenixFilterInput): [PhoenixDataPoint]
}

"""
Input filters for the Project Phoenix query.
"""
input PhoenixFilterInput {
  startTime: ISO8601DateTime
  endTime: ISO8601DateTime
  assetIds: [ID!]
  metrics: [String!]
}

"""
A single data point for a real-time asset.
"""
type PhoenixDataPoint {
  id: ID!
  timestamp: ISO8601DateTime!
  assetId: ID!
  metricName: String!
  value: Float!
  status: String
}

scalar ISO8601DateTime
```

### SDKs: `adam-sdk-python` Example
This snippet demonstrates the new, developer-friendly SDK, which handles the new OAuth 2.0 flow.

```python
# docs/sdk-examples/python_example.py
from adam.sdk import AdamClient, PhoenixFilter
from datetime import datetime, timedelta

# 1. Instantiate client using new OAuth 2.0 flow
# The SDK handles the client_credentials grant and token refresh
client = AdamClient(
    client_id="partner-client-id-123",
    client_secret="partner-client-secret-abc"
)

# 2. Define the query filter
start_time = datetime.utcnow() - timedelta(hours=1)
phoenix_filter = PhoenixFilter(
    startTime=start_time.isoformat(),
    assetIds=["asset-001", "asset-002"],
    metrics=["cpu_usage", "memory"]
)

try:
    # 3. Make a simple, fluent API call
    # The SDK calls the GraphQL API under the hood
    data_points = client.project_phoenix.get_realtime_data(filter=phoenix_filter)
    for point in data_points:
        print(f"[{point.timestamp}] Asset {point.assetId}: {point.metricName} = {point.value}")
except Exception as e:
    print(f"Error fetching data: {e}")
```

## 7. Observability Stack Configuration
As per the blueprint, we will use the ELK Stack and Jaeger.

### Centralized Logging: `logstash.conf`
This Logstash config ingests JSON-formatted logs from our K8s cluster (shipped via Filebeat) and sends them to Elasticsearch.

```
# infra/elk/logstash.conf
input {
  # Ingest logs from Filebeat running in the K8s cluster
  beats {
    port => 5044
  }
}

filter {
  # Logs from K8s pods are expected to be JSON
  json {
    source => "message"
  }

  # Add K8s metadata (pod name, namespace, etc.)
  # This data is added by the Filebeat K8s autodiscover
  mutate {
    add_field => {
      "pod_name" => "%{[kubernetes][pod][name]}"
      "namespace" => "%{[kubernetes][namespace]}"
      "service_name" => "%{[kubernetes][container][name]}"
    }
  }
}

output {
  # Send to Elasticsearch
  elasticsearch {
    hosts => ["http://elasticsearch-master:9200"]
    index => "adam-logs-%{+YYYY.MM.dd}"
  }
}
```

### Distributed Tracing: Jaeger `application.properties`
We will configure all new Java/Spring Boot microservices (e.g., `svc-reporting`) to export traces to Jaeger for distributed tracing.

```properties
# apps/svc-reporting/src/main/resources/application.properties

# Enable Spring Cloud Sleuth (for trace/span IDs)
spring.sleuth.enabled=true

# Enable OpenTracing compatibility
spring.sleuth.opentracing.enabled=true

# Configure the Jaeger client
# We assume a Jaeger agent is running as a DaemonSet in K8s
# or as a sidecar.
management.tracing.sampling.probability=0.1 # Sample 10% of requests
management.opentracing.jaeger.udp-sender.host=jaeger-agent.observability.svc.cluster.local
management.opentracing.jaeger.udp-sender.port=6831

# Service name for Jaeger UI
spring.application.name=svc-reporting
```

End of Plan
