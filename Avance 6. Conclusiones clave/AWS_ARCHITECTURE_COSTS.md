# AWS Architecture & Cost Analysis

## Executive Summary

- **Selected Platform**: AWS (SageMaker + Lambda + RDS)
- **Estimated Monthly Cost**: $2,500 - $3,500 USD
- **Annual Cost**: $30,000 - $42,000 USD
- **ROI Estimate**: 8-12 months (assuming 10-15% improvement in productivity)
- **Deployment Time**: 12 weeks
- **Production Ready**: Yes

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    VisionOps Manufacturing Floor             │
│                                                               │
│  YOLO Detector → Embeddings (V-JEPA 2 / DINOv2)             │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   API Gateway (REST)   │
        │  POST /predict         │
        │  Authentication: API   │
        └────────────┬───────────┘
                     │
        ┌────────────▼───────────┐
        │  Lambda (Inference)    │
        │  512 MB, 60s timeout   │
        │  Auto-scale: 0-1000    │
        └────────────┬───────────┘
                     │
        ┌────────────▼──────────────────┐
        │  SageMaker Endpoint            │
        │  ml.m5.xlarge (x2, Multi-AZ)  │
        │  Model: V-JEPA 2 + LogReg     │
        │  <5ms latency                 │
        └────────────┬──────────────────┘
                     │
        ┌────────────┴────────────────────┐
        │                                 │
        ▼                                 ▼
    ┌────────┐                      ┌──────────┐
    │   S3   │                      │   RDS    │
    │ Models │                      │PostgreSQL│
    │ Data   │                      │ 100GB    │
    └────────┘                      └──────────┘
        │                               ▲
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌──────────────────────────┐
        │  CloudWatch Monitoring   │
        │  Logs, Metrics, Alarms   │
        └──────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────┐
        │   SNS Notifications      │
        │   Email/SMS Alerts       │
        └──────────────────────────┘
```

---

## Detailed Component Specifications

### 1. API Gateway

**Purpose**: HTTP/REST interface para acceder al modelo

```
Service: API Gateway
Tier: REST API
Endpoint: https://visionops-api.amazonaws.com/prod/predict

Configuration:
  - Method: POST
  - Request Body: JSON
  {
    "embedding": [float, ...],  # 1024 values
    "model_version": "v1"
  }
  
  - Response: JSON
  {
    "prediction": "action_name",
    "confidence": 0.95,
    "latency_ms": 4.2,
    "timestamp": "2026-06-13T15:23:45Z"
  }
  
  - Authentication: API Key (stored in Secrets Manager)
  - Throttling: 10,000 requests/day
  - Logging: CloudWatch Logs
  - CORS: Only from manufacturing floor IPs
```

**Cost**:
- API calls: $3.50 per million (0 - 1M), then lower tier
- Estimated: 1M calls/month = $3.50/month

---

### 2. Lambda Function (Inference)

**Purpose**: Validar request, invocar SageMaker, guardar resultado en RDS

```python
# Pseudo-code de función Lambda

import json
import boto3
import psycopg2
import logging

sagemaker_runtime = boto3.client('sagemaker-runtime')
logger = logging.getLogger()

def lambda_handler(event, context):
    try:
        # Parse request
        body = json.loads(event['body'])
        embedding = body['embedding']
        model_version = body.get('model_version', 'v1')
        
        # Validar embedding (1024 dims)
        if len(embedding) != 1024:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid embedding dimension'})
            }
        
        # Invocar SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName='v-jepa2-har-prod',
            ContentType='application/json',
            Body=json.dumps({'embedding': embedding})
        )
        
        result = json.loads(response['Body'].read().decode())
        prediction_label = result['label']  # 0-11
        confidence = result['confidence']   # 0.0-1.0
        
        # Conectar a RDS y guardar predicción
        conn = psycopg2.connect(
            host=os.environ['RDS_HOST'],
            database='visionops',
            user=os.environ['RDS_USER'],
            password=os.environ['RDS_PASSWORD']
        )
        cursor = conn.cursor()
        
        query = """
        INSERT INTO predictions (
            embedding_id, model_version, prediction_label,
            confidence, created_at
        ) VALUES (%s, %s, %s, %s, NOW())
        """
        
        cursor.execute(query, (
            body.get('embedding_id'),
            model_version,
            prediction_label,
            confidence
        ))
        conn.commit()
        cursor.close()
        conn.close()
        
        # Retornar predicción
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': prediction_label,
                'confidence': confidence,
                'model_version': model_version
            })
        }
        
    except Exception as e:
        logger.error(f'Error: {str(e)}')
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal server error'})
        }
```

**Configuration**:
- Runtime: Python 3.11
- Memory: 512 MB
- Timeout: 60 seconds
- VPC: Private subnet (acceso a RDS)
- Ephemeral storage: 512 MB
- Concurrency: 1000 requests simultáneos
- Layers: psycopg2, boto3

**Cost**:
- Invocations: $0.20 per million
- Duración: 1M calls × 2s × $0.0000166667/sec = $33.33
- Estimated: 1M calls/month = $33.53/month

---

### 3. SageMaker Endpoint

**Purpose**: Servir predicciones del modelo V-JEPA 2 + Logistic Regression

```
Service: SageMaker
Endpoint Name: v-jepa2-har-prod
Endpoint Type: Multi-Model Endpoint (MME)

Configuration:
  - Model: V-JEPA 2 + LogisticRegression (joblib)
  - Instance Type: ml.m5.xlarge
    * vCPU: 4
    * Memory: 16 GB RAM
    * Network: Up to 10 Gbps
  
  - Deployment:
    * Initial instance count: 2 (for HA)
    * Availability Zones: 2 (multi-AZ)
    * Auto-scaling: 2 - 10 instances
    
  - Inference Config:
    * Content type: application/json
    * Accept: application/json
    * Timeout: 30 seconds
    * Max payload: 6 MB
    
  - Data Capture:
    * Enabled: 10% sampling
    * Destination: S3
    * Purpose: Model monitoring & retraining
```

**How It Works**:
1. Lambda envía embedding (1024 dims) a endpoint
2. SageMaker deserialiaza embedding
3. Logistic Regression predice clase + probabilidades
4. Retorna JSON con predicción y confianza

**Performance**:
- Latency: ~2-5ms (p50)
- Throughput: 100-500 req/s per instance
- Total capacity: 200-5000 req/s (2-10 instances)

**Cost**:
- ml.m5.xlarge: $0.115/hour
- 2 instances × 730 hours = $168.10/month (base)
- Auto-scaling instances: ~$84.05 (50% utilization)
- Estimated: $252.15/month

---

### 4. Lambda (Async Retraining)

**Purpose**: Reentrenar modelo automáticamente cada semana

```python
# Pseudo-código reentrenamiento semanal

import boto3
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

s3 = boto3.client('s3')
rds = boto3.client('rds')
sagemaker = boto3.client('sagemaker')

def retrain_lambda(event, context):
    """Reentrenamiento automático cada domingo 22:00 UTC"""
    
    try:
        # 1. Descargar datos nuevos de RDS
        query = """
        SELECT embedding, label
        FROM training_data
        WHERE created_at > NOW() - INTERVAL '7 days'
        AND valid = TRUE
        """
        df = pd.read_sql(query, conn)
        
        if len(df) < 100:
            print("Not enough new data, skipping retraining")
            return
        
        # 2. Extraer embeddings y labels
        X = np.array(df['embedding'].tolist())  # (n, 1024)
        y = df['label'].values  # (n,)
        
        # 3. Reentrenar Logistic Regression
        model = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            random_state=42
        )
        model.fit(X, y)
        
        # 4. Evaluar en holdout set
        test_df = get_validation_set()
        X_test = np.array(test_df['embedding'].tolist())
        y_test = test_df['label'].values
        
        accuracy = model.score(X_test, y_test)
        
        # 5. Comparar con baseline (78.54%)
        if accuracy >= 0.78:  # OK
            # Serializar modelo
            model_bytes = joblib.dumps(model)
            
            # Subir a S3
            s3.put_object(
                Bucket='visionops-ml-models',
                Key='v-jepa2/model.pkl',
                Body=model_bytes
            )
            
            # Crear versión en SageMaker
            new_version = f"v-jepa2-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            sagemaker.create_model(
                ModelName=new_version,
                PrimaryContainer={
                    'Image': 'ECR_IMAGE_URI',
                    'ModelDataUrl': 's3://visionops-ml-models/v-jepa2/model.pkl'
                }
            )
            
            # Notificar éxito
            send_sns_notification(
                f"Retraining successful! New accuracy: {accuracy:.2%}",
                'visionops-alerts'
            )
        else:
            # Notificar falla
            send_sns_notification(
                f"Retraining failed. Accuracy {accuracy:.2%} < 0.78",
                'visionops-alerts'
            )
    
    except Exception as e:
        send_sns_notification(f"Retraining error: {str(e)}", 'visionops-alerts-critical')
```

**Configuration**:
- Trigger: EventBridge (cron: 0 22 ? * SUN)
- Runtime: Python 3.11
- Memory: 2048 MB (procesamiento ML)
- Timeout: 900 seconds (15 min)
- Layers: pandas, numpy, scikit-learn

**Cost**:
- Duración: 15 min = 900 sec
- 1 invocation/week = 4 invocations/month
- Cost: 4 × 900 × $0.0000166667 = $0.06/month

---

### 5. RDS PostgreSQL

**Purpose**: Persistencia de predicciones y datos de entrenamiento

```sql
-- Tablas principales

CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    embedding_id VARCHAR(255),
    model_version VARCHAR(50),
    prediction_label INTEGER,  -- 0-11
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_created_at (created_at)
);

CREATE TABLE ground_truth (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER,
    actual_label INTEGER,
    corrected_by VARCHAR(255),
    created_at TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    accuracy FLOAT,
    macro_f1 FLOAT,
    auc_roc FLOAT,
    evaluated_at TIMESTAMP,
    UNIQUE KEY unique_version_date (model_version, evaluated_at)
);

CREATE TABLE training_data (
    id SERIAL PRIMARY KEY,
    embedding BYTEA,  -- 1024-dim vector as binary
    label INTEGER,
    valid BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Configuration**:
- Engine: PostgreSQL 14
- Instance: db.t3.medium
  * vCPU: 2
  * Memory: 4 GB
  * Storage: 100 GB gp3
- Backup: Diario, retención 30 días
- Multi-AZ: Sí (automatic failover)
- Enhanced monitoring: Habilitado
- Performance Insights: Habilitado
- Subnet group: Private subnets (2 AZ)
- Security group: Solo desde Lambda + SageMaker

**Capacity Planning**:
- Rows por mes: ~1M predictions × 1 row = 1M rows
- Crecimiento anual: ~12M rows
- Storage: 100 GB es suficiente para 2 años

**Cost**:
- Instance: db.t3.medium = $0.166/hour × 730 = $121.18/month
- Storage: 100 GB = $0.02/GB-month × 100 = $2.00
- Estimated: $123.18/month

---

### 6. CloudWatch Monitoring

**Purpose**: Observabilidad y alertas

```
Log Groups:
  - /aws/lambda/visionops-inference
  - /aws/sagemaker/endpoints/v-jepa2-har-prod
  - /aws/rds/instance/visionops-db

Metrics (Custom):
  - Prediction latency (ms) - histogram
  - Prediction confidence - histogram
  - Accuracy vs ground truth - percent
  - Model drift (KS test) - boolean
  - RDS CPU utilization - percent
  - SageMaker endpoint invocations - count

Dashboards:
  1. "VisionOps Real-Time" (5-min granularity)
     - Predictions/minute
     - Confidence distribution
     - Latency (p50, p95, p99)
     - Errors/minute
  
  2. "Model Health" (1-hour granularity)
     - Accuracy vs time
     - Drift detection
     - Feature distribution
     - Class distribution
  
  3. "Infrastructure" (1-hour granularity)
     - CPU/Memory utilization
     - Network throughput
     - Costs (daily)

Alarms:
  - Latency p99 > 50ms → WARN
  - Accuracy < 75% → CRITICAL
  - Endpoint unhealthy → CRITICAL
  - RDS CPU > 80% → WARN
  - Lambda errors > 1% → WARN
  - Model drift detected → WARN
```

**Cost**:
- Logs: $0.50 per GB ingested
  - ~10 GB/month logs = $5.00
- Metrics: $0.30 per custom metric-month
  - 6 custom metrics × $0.30 = $1.80
- Alarms: $0.10 per alarm-month
  - 6 alarms × $0.10 = $0.60
- Estimated: $7.40/month

---

### 7. SNS Notifications

**Purpose**: Alertas vía email/SMS

```
Topics:
  - visionops-alerts (info/warning) → email
  - visionops-alerts-critical (critical) → email + SMS

Subscriptions:
  - MLOps Lead: [email + SMS]
  - DevOps Engineer: [email]
  - Manufacturing Manager: [email]
```

**Cost**:
- Emails: $0.02 per email (SNS to email)
- Estimated: 1000 emails/month = $20.00
- SMS: $0.00645 per SMS (optional)
- Estimated: $20.00/month

---

## Cost Breakdown

### Monthly Costs (Steady State)

| Service | Count | Unit Cost | Total |
|---------|-------|-----------|-------|
| **API Gateway** | 1M calls | $0.0000035 | $3.50 |
| **Lambda (Inference)** | 1M calls | $0.00003353 | $33.53 |
| **Lambda (Retraining)** | 4 calls | $0.0000167 | $0.06 |
| **SageMaker Endpoint** | 2 instances × 730h | $0.115/h | $168.10 |
| **SageMaker Auto-scaling** | 1 instance × 730h avg | $0.115/h | $84.05 |
| **RDS Instance** | 1 × 730h | $0.166/h | $121.18 |
| **RDS Storage** | 100 GB | $0.02/GB | $2.00 |
| **S3 (Models)** | 10 GB | $0.023/GB | $0.23 |
| **S3 (Data)** | 50 GB | $0.023/GB | $1.15 |
| **CloudWatch** | Logs + Metrics + Alarms | - | $7.40 |
| **SNS Notifications** | 1000 emails | $0.02 | $20.00 |
| **VPC + Security** | Nat Gateway, endpoints | - | $50.00 |
| **Misc** | Secrets Manager, KMS, etc | - | $30.00 |
| | | **TOTAL** | **$521.20** |

### Annual Costs

```
Monthly: $521.20
Annual: $521.20 × 12 = $6,254.40 USD
3-year: $6,254.40 × 3 = $18,763.20 USD
```

### Cost Optimization Strategies

1. **Reserved Instances** (SageMaker)
   - 1-year RI: ~30% discount
   - SageMaker RI: $168.10 → $117.67/month
   - Annual savings: $600

2. **Spot Instances** (Auto-scaling)
   - 70% discount on instances
   - Current: $84.05 → $25.21/month
   - Annual savings: $700

3. **Consolidated Billing**
   - Multiple AWS accounts → single bill
   - Potential 10% discount

4. **Volume Discounts**
   - API Gateway: $3.50 → $3.15 (1B threshold)
   - Savings: Minimal (<$100/year)

**With optimizations**:
```
Monthly: $521.20 - ($600/12) - ($700/12) = $417.53
Annual: $417.53 × 12 = $5,010.36 USD
Savings: $1,244/year
```

---

## Competitive Analysis

### AWS vs Azure vs GCP (Cost)

| Feature | AWS | Azure | GCP |
|---------|-----|-------|-----|
| ML Model Serving | $168 | $200 | $140 |
| Serverless Compute | $34 | $45 | $32 |
| Database (Managed) | $121 | $150 | $100 |
| **Monthly Total** | **$521** | **$650** | **$480** |
| **Annual Total** | **$6,254** | **$7,800** | **$5,760** |

**AWS Verdict**: Mid-range pricing, best feature set = best value.

---

## Deployment Cost (One-time)

| Item | Cost | Duration |
|------|------|----------|
| AWS setup & configuration | $2,000 | 2 weeks |
| Model packaging & testing | $1,500 | 2 weeks |
| Infrastructure as Code (IaC) | $1,000 | 1 week |
| Monitoring setup | $1,000 | 1 week |
| Security & compliance | $1,500 | 1 week |
| Documentation & training | $2,000 | 2 weeks |
| **TOTAL** | **$9,000** | **12 weeks** |

---

## ROI Analysis

### Assumptions
- Current manual process: 2 operators × 40 hours/week = 80 hours
- VisionOps reduces to: 1 operator × 10 hours/week = 10 hours
- **Savings: 70 hours/week**

- Salary cost: $30/hour (fully loaded)
- Weekly savings: 70 hours × $30 = $2,100
- Monthly savings: $2,100 × 4.33 = $9,093
- **Annual savings: $109,116**

### Payback Period
```
Total Cost Year 1: $9,000 (setup) + $6,254 (operation) = $15,254
Annual Savings: $109,116
Payback: 15,254 / 109,116 = 1.7 months
```

### 3-Year Total Cost of Ownership (TCO)
```
Year 1: $9,000 + $6,254 = $15,254
Year 2: $6,254
Year 3: $6,254
TOTAL 3-year: $27,762

Annual savings: $109,116 × 3 = $327,348
NET BENEFIT: $327,348 - $27,762 = $299,586
```

---

## Scaling Considerations

### Phase 1: Current (1M predictions/month)
- 2 SageMaker instances
- 1000 Lambda concurrency
- 100 GB RDS storage
- Monthly cost: $521

### Phase 2: Growth (10M predictions/month)
- 5 SageMaker instances (auto-scale)
- 5000 Lambda concurrency
- 500 GB RDS storage
- Monthly cost: ~$1,200

### Phase 3: Enterprise (100M predictions/month)
- 20 SageMaker instances (auto-scale)
- 10000 Lambda concurrency
- 2 TB RDS storage (or switch to DynamoDB)
- Monthly cost: ~$3,500

**Scaling is linear** – costs grow with volume but infrastructure handles it.

---

## Security & Compliance

### Encryption
- **In transit**: TLS 1.3 (API Gateway → Lambda → SageMaker)
- **At rest**: AES-256 (S3, RDS, EBS)
- **Key management**: AWS KMS (customer-managed keys)

### Access Control
- IAM roles with least-privilege principle
- API Gateway API key requirement
- VPC isolation (private subnets only)
- Security groups restrict traffic

### Compliance
- GDPR: Data retention policy (90 days in S3, 1 year in RDS, then delete)
- HIPAA: If patient data involved, enable additional logging
- SOC 2: AWS compliant

### Audit & Logging
- CloudTrail: All API calls logged
- VPC Flow Logs: Network traffic
- CloudWatch Logs: Application logs
- S3 Server Access Logs: Data access

---

## Disaster Recovery & High Availability

### RTO/RPO
- **RTO** (Recovery Time Objective): 5 minutes (multi-AZ failover)
- **RPO** (Recovery Point Objective): 1 minute (backup interval)

### Backup Strategy
- **RDS**: Automated daily backups (30-day retention)
- **S3**: Versioning enabled, cross-region replication (optional, +$0.02/GB)
- **Model artifacts**: Versioned in S3

### Disaster Recovery Plan
1. **RDS failure** → Automatic failover to standby (multi-AZ)
2. **SageMaker endpoint failure** → Auto-recreate from model in S3
3. **Complete AWS region failure** → Manual failover to secondary region (requires setup, ~1 hour)

**Optional**: Cross-region replication (add $200-300/month)

---

## Final Recommendation

**AWS is the optimal choice** for VisionOps due to:
1. ✓ Best price-to-performance ratio
2. ✓ Mature ML services (SageMaker)
3. ✓ High scalability
4. ✓ Team experience
5. ✓ Strong security & compliance

**Monthly Operating Cost**: $521 (can be reduced to $418 with optimizations)  
**ROI**: 1.7 months payback period  
**Risk**: Low (AWS is market leader with 99.99% SLA)

---

*Document Version: 1.0*  
*Date: 2026-06-13*  
*Reviewed by: [ML Lead]*  
*Approved by: [Engineering Manager]*
