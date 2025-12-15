
import React from 'react';
import { AWSCategory, Template } from './types';
import { Ec2Icon, LambdaIcon, AutoScalingGroupIcon, EcsIcon, EksIcon, AmiIcon, VpcIcon, SubnetIcon, LoadBalancerIcon, TargetGroupIcon, Route53Icon, ApiGatewayIcon, SqsIcon, S3Icon, EfsIcon, EbsIcon, RdsIcon, DynamoDbIcon, CloudWatchIcon, CloudTrailIcon, ConfigIcon, XRayIcon, IamRoleIcon, SecurityGroupIcon, WafIcon, SecretsManagerIcon, DatadogIcon, GrafanaIcon, PrometheusIcon, ElkIcon, StartNodeIcon, EndNodeIcon, GroupIcon, RectangleIcon, EllipseIcon, TextIcon, CloudFrontIcon, SnsIcon, DataIcon, ModelIcon, TrainingIcon, EvalIcon } from './components/Icons';
import { DL_CATEGORIES } from './dl-constants'; // Import DL categories

export const NODE_WIDTH = 180;
export const NODE_HEIGHT = 64;
export const WORKFLOW_NODE_SIZE = 64;

export const PIPELINE_COMMON_CODE = `import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Common Pipeline Steps
df = pd.read_csv('dataset.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)`;

const getCodeForModel = (importStmt: string, initStmt: string, isReg: boolean) => `
${PIPELINE_COMMON_CODE}

# Model Implementation
${importStmt}
model = ${initStmt}
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Metrics
from sklearn.metrics import ${isReg ? 'mean_squared_error, r2_score' : 'accuracy_score, classification_report'}
print("Eval Results:")
${isReg ? 'print(mean_squared_error(y_test, y_pred))' : 'print(classification_report(y_test, y_pred))'}
`;

const COLORS = {
    pipeline: 'bg-blue-100 border-blue-500 shadow-blue-500/20 dark:bg-blue-900/40 dark:border-blue-500',
    regression: 'bg-green-100 border-green-500 shadow-green-500/20 dark:bg-green-900/40 dark:border-green-500',
    classification: 'bg-orange-100 border-orange-500 shadow-orange-500/20 dark:bg-orange-900/40 dark:border-orange-500',
    unsupervised: 'bg-purple-100 border-purple-500 shadow-purple-500/20 dark:bg-purple-900/40 dark:border-purple-500',
    rl: 'bg-yellow-100 border-yellow-500 shadow-yellow-500/20 dark:bg-yellow-900/40 dark:border-yellow-500',
};

export const AWS_CATEGORIES: AWSCategory[] = [
  {
    name: 'Workflow',
    services: [
      { id: 'start', name: 'Start', icon: StartNodeIcon, defaultData: { label: 'Start' } },
      { id: 'end', name: 'End', icon: EndNodeIcon, defaultData: { label: 'End' } },
    ]
  },
  {
    name: 'Shapes & Layout',
    services: [
        { id: 'group', name: 'Group Outline', icon: GroupIcon, defaultData: { label: 'Group' } },
        { id: 'shape-rectangle', name: 'Rectangle', icon: RectangleIcon, defaultData: { label: 'Rectangle' } },
        { id: 'shape-ellipse', name: 'Ellipse', icon: EllipseIcon, defaultData: { label: 'Ellipse' } },
        { id: 'text', name: 'Text', icon: TextIcon, defaultData: { label: 'Text' } },
    ]
  },
  {
    name: 'Compute',
    services: [
      { id: 'ec2', name: 'EC2', icon: Ec2Icon, defaultData: { label: 'EC2 Instance', health: 'healthy', instanceFamily: 't2', instanceSize: 'micro', amiId: 'ami-0c55b159cbfafe1f0', securityGroup: 'sg-default', iamRole: 'none', subnetId: 'subnet-default', ebsSize: 8, ebsType: 'gp3' } },
      { id: 'lambda', name: 'Lambda', icon: LambdaIcon, defaultData: { label: 'Lambda Function', health: 'healthy', triggerType: 'API Gateway', runtime: 'Node.js 20.x', memorySize: 128, timeout: 3 } },
      { id: 'asg', name: 'Auto Scaling', icon: AutoScalingGroupIcon, defaultData: { label: 'ASG', health: 'healthy', minSize: "1", maxSize: "10", desiredCapacity: "2", scalingPolicy: 'manual' } },
      { id: 'ecs', name: 'ECS', icon: EcsIcon, defaultData: { label: 'ECS Cluster', health: 'healthy' } },
      { id: 'eks', name: 'EKS', icon: EksIcon, defaultData: { label: 'EKS Cluster', health: 'healthy' } },
      { id: 'ami', name: 'AMI', icon: AmiIcon, defaultData: { label: 'AMI', baseOs: 'Amazon Linux 2', version: '1.0.0' } },
    ],
  },
  {
    name: 'Networking & Delivery',
    services: [
      { id: 'vpc', name: 'VPC', icon: VpcIcon, defaultData: { label: 'VPC', cidr: '10.0.0.0/16' } },
      { id: 'subnet', name: 'Subnets', icon: SubnetIcon, defaultData: { label: 'Subnet' } },
      { id: 'alb', name: 'Load Balancer', icon: LoadBalancerIcon, defaultData: { label: 'ALB', health: 'healthy', listeners: 'HTTP:80' } },
      { id: 'tg', name: 'Target Group', icon: TargetGroupIcon, defaultData: { label: 'Target Group', health: 'healthy', targetType: 'instance', healthCheckPath: '/', port: 80, protocol: 'HTTP' } },
      { id: 'r53', name: 'Route 53', icon: Route53Icon, defaultData: { label: 'Route 53', hostedZone: 'example.com', recordType: 'A' } },
      { id: 'apigw', name: 'API Gateway', icon: ApiGatewayIcon, defaultData: { label: 'API Gateway', health: 'healthy', endpointType: 'REGIONAL' } },
      { id: 'cloudfront', name: 'CloudFront', icon: CloudFrontIcon, defaultData: { label: 'CloudFront', priceClass: 'PriceClass_100' } },
    ],
  },
  {
    name: 'Application Integration',
    services: [
        { id: 'sqs', name: 'SQS', icon: SqsIcon, defaultData: { label: 'SQS Queue', fifo: false, retentionPeriod: 4 } },
        { id: 'sns', name: 'SNS', icon: SnsIcon, defaultData: { label: 'SNS Topic', topicName: 'my-topic', fifo: false } },
    ],
  },
  {
    name: 'Storage',
    services: [
      { id: 's3', name: 'S3', icon: S3Icon, defaultData: { label: 'S3 Bucket', bucketName: 'my-bucket', versioning: false, encryption: 'SSE-S3', publicAccess: false, storageClass: 'Standard' } },
      { id: 'efs', name: 'EFS', icon: EfsIcon, defaultData: { label: 'EFS' } },
      { id: 'ebs', name: 'EBS', icon: EbsIcon, defaultData: { label: 'EBS Volume', ebsSize: 10, ebsType: 'gp3', ebsEncrypted: true } },
      { id: 'rds', name: 'RDS', icon: RdsIcon, defaultData: { label: 'RDS Database', health: 'healthy', engine: 'PostgreSQL', storageSize: 20, readReplicas: false, multiAZ: false } },
      { id: 'dynamodb', name: 'DynamoDB', icon: DynamoDbIcon, defaultData: { label: 'DynamoDB Table', health: 'healthy', partitionKey: 'id', billingMode: 'PROVISIONED' } },
    ],
  },
  {
    name: 'Monitoring & Logging',
    services: [
      { id: 'cloudwatch', name: 'CloudWatch', icon: CloudWatchIcon, defaultData: { label: 'CloudWatch', metric: 'CPUUtilization', threshold: 80 } },
      { id: 'cloudtrail', name: 'CloudTrail', icon: CloudTrailIcon, defaultData: { label: 'CloudTrail' } },
      { id: 'config', name: 'AWS Config', icon: ConfigIcon, defaultData: { label: 'AWS Config' } },
      { id: 'xray', name: 'X-Ray', icon: XRayIcon, defaultData: { label: 'X-Ray' } },
    ],
  },
  {
    name: 'Security & IAM',
    services: [
      { id: 'iam', name: 'IAM Role', icon: IamRoleIcon, defaultData: { label: 'IAM Role' } },
      { id: 'sg', name: 'Security Group', icon: SecurityGroupIcon, defaultData: { label: 'Security Group' } },
      { id: 'waf', name: 'WAF', icon: WafIcon, defaultData: { label: 'WAF' } },
      { id: 'secrets', name: 'Secrets Manager', icon: SecretsManagerIcon, defaultData: { label: 'Secrets Manager' } },
    ],
  },
  {
    name: 'Third-Party Tools',
    services: [
      { id: 'datadog', name: 'Datadog', icon: DatadogIcon, defaultData: { label: 'Datadog' } },
      { id: 'grafana', name: 'Grafana', icon: GrafanaIcon, defaultData: { label: 'Grafana' } },
      { id: 'prometheus', name: 'Prometheus', icon: PrometheusIcon, defaultData: { label: 'Prometheus' } },
      { id: 'elk', name: 'ELK Stack', icon: ElkIcon, defaultData: { label: 'ELK Stack' } },
    ],
  }
];

export const AI_CATEGORIES: AWSCategory[] = [
    {
        name: 'Workflow Control',
        services: [
            { id: 'start', name: 'Start Flow', icon: StartNodeIcon, defaultData: { label: 'Start' } },
            { id: 'end', name: 'End Flow', icon: EndNodeIcon, defaultData: { label: 'End' } },
            { id: 'group', name: 'Group', icon: GroupIcon, defaultData: { label: 'Pipeline Group' } },
            { id: 'text', name: 'Note', icon: TextIcon, defaultData: { label: 'Annotation' } },
        ]
    },
    {
        name: 'Supervised: Pipeline',
        services: [
            { id: 'ai-dataset', name: 'Load Dataset', icon: DataIcon, defaultData: { label: 'Dataset (CSV)', customTheme: COLORS.pipeline, code: "import pandas as pd\ndf = pd.read_csv('data.csv')" } },
            { id: 'ai-split', name: 'Train-Test Split', icon: DataIcon, defaultData: { label: 'Split Data', customTheme: COLORS.pipeline, code: "X_train, X_test, y_train, y_test = train_test_split(X, y)" } },
            { id: 'ai-scaler', name: 'Standard Scaler', icon: DataIcon, defaultData: { label: 'Scaler', customTheme: COLORS.pipeline, code: "scaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)" } },
        ]
    },
    {
        name: 'Regression Models',
        services: [
            { id: 'reg-linear', name: 'Linear Regression', icon: ModelIcon, defaultData: { label: 'Linear Reg', customTheme: COLORS.regression, code: getCodeForModel("from sklearn.linear_model import LinearRegression", "LinearRegression()", true) } },
            { id: 'reg-poly', name: 'Polynomial Reg', icon: ModelIcon, defaultData: { label: 'Polynomial Reg', customTheme: COLORS.regression, code: getCodeForModel("from sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.linear_model import LinearRegression", "make_pipeline(PolynomialFeatures(degree=2), LinearRegression())", true) } },
            { id: 'reg-ridge', name: 'Ridge', icon: ModelIcon, defaultData: { label: 'Ridge Reg', customTheme: COLORS.regression, code: getCodeForModel("from sklearn.linear_model import Ridge", "Ridge(alpha=1.0)", true) } },
            { id: 'reg-lasso', name: 'Lasso', icon: ModelIcon, defaultData: { label: 'Lasso Reg', customTheme: COLORS.regression, code: getCodeForModel("from sklearn.linear_model import Lasso", "Lasso(alpha=0.1)", true) } },
            { id: 'reg-elastic', name: 'ElasticNet', icon: ModelIcon, defaultData: { label: 'ElasticNet', customTheme: COLORS.regression, code: getCodeForModel("from sklearn.linear_model import ElasticNet", "ElasticNet(alpha=0.1, l1_ratio=0.5)", true) } },
            { id: 'reg-svr', name: 'SVR', icon: ModelIcon, defaultData: { label: 'SVR', customTheme: COLORS.regression, code: getCodeForModel("from sklearn.svm import SVR", "SVR(kernel='rbf')", true) } },
            { id: 'reg-dt', name: 'Decision Tree Reg', icon: ModelIcon, defaultData: { label: 'Decision Tree', customTheme: COLORS.regression, code: getCodeForModel("from sklearn.tree import DecisionTreeRegressor", "DecisionTreeRegressor(max_depth=5)", true) } },
            { id: 'reg-rf', name: 'Random Forest', icon: ModelIcon, defaultData: { label: 'RF Regressor', customTheme: COLORS.regression, code: getCodeForModel("from sklearn.ensemble import RandomForestRegressor", "RandomForestRegressor(n_estimators=100)", true) } },
            { id: 'reg-gb', name: 'Gradient Boosting', icon: ModelIcon, defaultData: { label: 'GB Regressor', customTheme: COLORS.regression, code: getCodeForModel("from sklearn.ensemble import GradientBoostingRegressor", "GradientBoostingRegressor(n_estimators=100)", true) } },
            { id: 'reg-xgb', name: 'XGBoost', icon: ModelIcon, defaultData: { label: 'XGBoost Reg', customTheme: COLORS.regression, code: getCodeForModel("from xgboost import XGBRegressor", "XGBRegressor(n_estimators=100)", true) } },
            { id: 'reg-lgbm', name: 'LightGBM', icon: ModelIcon, defaultData: { label: 'LightGBM Reg', customTheme: COLORS.regression, code: getCodeForModel("from lightgbm import LGBMRegressor", "LGBMRegressor()", true) } },
            { id: 'reg-cat', name: 'CatBoost', icon: ModelIcon, defaultData: { label: 'CatBoost Reg', customTheme: COLORS.regression, code: getCodeForModel("from catboost import CatBoostRegressor", "CatBoostRegressor(verbose=0)", true) } },
            { id: 'reg-sgd', name: 'SGD Regressor', icon: ModelIcon, defaultData: { label: 'SGD Regressor', customTheme: COLORS.regression, code: getCodeForModel("from sklearn.linear_model import SGDRegressor", "SGDRegressor()", true) } },
        ]
    },
    {
        name: 'Classification Models',
        services: [
            { id: 'clf-logistic', name: 'Logistic Regression', icon: ModelIcon, defaultData: { label: 'Logistic Reg', customTheme: COLORS.classification, code: getCodeForModel("from sklearn.linear_model import LogisticRegression", "LogisticRegression()", false) } },
            { id: 'clf-svm', name: 'SVM Classifier', icon: ModelIcon, defaultData: { label: 'SVC', customTheme: COLORS.classification, code: getCodeForModel("from sklearn.svm import SVC", "SVC(probability=True)", false) } },
            { id: 'clf-dt', name: 'Decision Tree', icon: ModelIcon, defaultData: { label: 'Decision Tree', customTheme: COLORS.classification, code: getCodeForModel("from sklearn.tree import DecisionTreeClassifier", "DecisionTreeClassifier()", false) } },
            { id: 'clf-rf', name: 'Random Forest', icon: ModelIcon, defaultData: { label: 'RF Classifier', customTheme: COLORS.classification, code: getCodeForModel("from sklearn.ensemble import RandomForestClassifier", "RandomForestClassifier()", false) } },
            { id: 'clf-knn', name: 'KNN', icon: ModelIcon, defaultData: { label: 'KNN', customTheme: COLORS.classification, code: getCodeForModel("from sklearn.neighbors import KNeighborsClassifier", "KNeighborsClassifier(n_neighbors=5)", false) } },
            { id: 'clf-nb', name: 'Naive Bayes', icon: ModelIcon, defaultData: { label: 'GaussianNB', customTheme: COLORS.classification, code: getCodeForModel("from sklearn.naive_bayes import GaussianNB", "GaussianNB()", false) } },
        ]
    },
    {
        name: 'Unsupervised & RL',
        services: [
            { id: 'unsup-kmeans', name: 'K-Means', icon: TrainingIcon, defaultData: { label: 'K-Means', customTheme: COLORS.unsupervised, code: "from sklearn.cluster import KMeans\nkmeans = KMeans(n_clusters=3)\nkmeans.fit(X)" } },
            { id: 'unsup-pca', name: 'PCA', icon: TrainingIcon, defaultData: { label: 'PCA', customTheme: COLORS.unsupervised, code: "from sklearn.decomposition import PCA\npca = PCA(n_components=2)\nX_pca = pca.fit_transform(X)" } },
            { id: 'rl-qlearn', name: 'Q-Learning', icon: TrainingIcon, defaultData: { label: 'Q-Agent', customTheme: COLORS.rl, code: "# Q-Learning implementation placeholder" } },
        ]
    }
];

export const DL_MODE_CATEGORIES = DL_CATEGORIES;

export const ALL_SERVICES = [...AWS_CATEGORIES.flatMap(c => c.services), ...AI_CATEGORIES.flatMap(c => c.services), ...DL_CATEGORIES.flatMap(c => c.services)];

export const SERVICE_RELATIONS: Record<string, string[]> = {
    'ec2': ['alb', 'asg', 'rds', 'ebs', 'cloudwatch', 'ami'],
    'lambda': ['apigw', 's3', 'dynamodb', 'cloudwatch', 'sqs', 'sns'],
    'alb': ['ec2', 'ecs', 'tg'],
    'rds': ['ec2', 'lambda'],
    's3': ['lambda', 'ec2', 'cloudtrail', 'cloudfront'],
    'apigw': ['lambda'],
    'sqs': ['lambda', 'ec2'],
    'sns': ['lambda', 'sqs'],
    'dynamodb': ['lambda', 'ec2'],
    'start': ['apigw', 'alb', 'ec2', 'cloudfront', 'ai-dataset'],
    'ami': ['asg', 'ec2'],
    'tg': ['ec2', 'lambda', 'ip'],
};

export const EC2_INSTANCE_TYPES = {
    't2': ['micro', 'small', 'medium', 'large'],
    't3': ['nano', 'micro', 'small', 'medium', 'large', 'xlarge', '2xlarge'],
    'm5': ['large', 'xlarge', '2xlarge', '4xlarge', '8xlarge'],
    'c5': ['large', 'xlarge', '2xlarge', '4xlarge', '9xlarge'],
    'r5': ['large', 'xlarge', '2xlarge', '4xlarge', '8xlarge'],
};

export const RDS_ENGINES = ['PostgreSQL', 'MySQL', 'MariaDB', 'Oracle', 'SQL Server', 'Aurora'];

export const ARCHITECTURE_TEMPLATES: Template[] = [
    {
        name: 'Comprehensive Cloud Architecture',
        description: 'A full-stack highly available architecture spanning Route53, CloudFront, ALB, Auto-Scaling, RDS, and S3.',
        nodes: [
            { id: 'start', type: 'start', position: { x: 50, y: 300 } },
            { id: 'r53', type: 'r53', position: { x: 180, y: 300 } },
            { id: 'cf', type: 'cloudfront', position: { x: 400, y: 300 } },
            { id: 'waf', type: 'waf', position: { x: 400, y: 150 } },
            { id: 's3_static', type: 's3', position: { x: 400, y: 500 }, data: { label: 'Static Assets' } },
            { id: 'alb', type: 'alb', position: { x: 650, y: 300 } },
            
            // App Layer - Target Group Container
            { id: 'tg_app', type: 'tg', position: { x: 900, y: 200 }, data: { label: 'App Target Group' } },
            { id: 'ec2_1', type: 'ec2', position: { x: 920, y: 250 }, data: { label: 'App Server 1' } },
            { id: 'ec2_2', type: 'ec2', position: { x: 1050, y: 250 }, data: { label: 'App Server 2' } },
            
            { id: 'asg', type: 'asg', position: { x: 900, y: 50 } },
            
            // Data Layer
            { id: 'rds_primary', type: 'rds', position: { x: 1300, y: 200 }, data: { label: 'Primary DB' } },
            { id: 'rds_replica', type: 'rds', position: { x: 1300, y: 400 }, data: { label: 'Read Replica' } },
            
            { id: 'end', type: 'end', position: { x: 1550, y: 300 } },
        ],
        connections: [
            { from: 'start', to: 'r53' },
            { from: 'r53', to: 'cf' },
            { from: 'cf', to: 's3_static' },
            { from: 'cf', to: 'alb' },
            { from: 'waf', to: 'cf' },
            { from: 'alb', to: 'tg_app' },
            // Implicit connections from TG to EC2 handled by container logic, but adding explicit for visuals
            { from: 'asg', to: 'tg_app' }, 
            { from: 'ec2_1', to: 'rds_primary' },
            { from: 'ec2_2', to: 'rds_primary' },
            { from: 'rds_primary', to: 'rds_replica' },
            { from: 'rds_primary', to: 'end' },
        ]
    },
    {
        name: 'VPC with Public/Private Subnets',
        description: 'A virtual private cloud layout with defined subnets.',
        nodes: [
             // VPC Boundary
            { id: 'vpc_shape', type: 'shape-rectangle', position: { x: 50, y: 50 }, data: { label: '', width: 800, height: 600, fillColor: 'transparent', strokeColor: '#2563eb', borderStyle: 'solid', strokeWidth: 4 } },
            { id: 'vpc_label', type: 'text', position: { x: 70, y: 70 }, data: { content: 'VPC: 10.0.0.0/16', fontSize: 20, color: '#2563eb' } },
            
            // Public Subnet
            { id: 'sub_pub', type: 'shape-rectangle', position: { x: 100, y: 150 }, data: { label: '', width: 700, height: 200, fillColor: 'rgba(34, 197, 94, 0.1)', strokeColor: '#22c55e', borderStyle: 'dashed', strokeWidth: 2 } },
            { id: 'pub_label', type: 'text', position: { x: 120, y: 170 }, data: { content: 'Public Subnet', fontSize: 16, color: '#22c55e' } },
            { id: 'igw', type: 'vpc', position: { x: 50, y: 200 }, data: { label: 'Internet Gateway' } },
            { id: 'nat', type: 'ec2', position: { x: 200, y: 220 }, data: { label: 'NAT Gateway' } },
            { id: 'bastion', type: 'ec2', position: { x: 500, y: 220 }, data: { label: 'Bastion Host' } },

            // Private Subnet
            { id: 'sub_priv', type: 'shape-rectangle', position: { x: 100, y: 400 }, data: { label: '', width: 700, height: 200, fillColor: 'rgba(239, 68, 68, 0.1)', strokeColor: '#ef4444', borderStyle: 'dashed', strokeWidth: 2 } },
            { id: 'priv_label', type: 'text', position: { x: 120, y: 420 }, data: { content: 'Private Subnet', fontSize: 16, color: '#ef4444' } },
            { id: 'app_srv', type: 'ec2', position: { x: 200, y: 470 }, data: { label: 'App Server' } },
            { id: 'db_srv', type: 'rds', position: { x: 500, y: 470 }, data: { label: 'Database' } },
        ],
        connections: [
            { from: 'igw', to: 'nat' },
            { from: 'igw', to: 'bastion' },
            { from: 'nat', to: 'app_srv' },
            { from: 'app_srv', to: 'db_srv' },
        ]
    },
    {
        name: 'Serverless API',
        description: 'API Gateway connected to a Lambda function and a DynamoDB table.',
        nodes: [
            { id: 'apigw_1', type: 'apigw', position: { x: 0, y: 100 } },
            { id: 'lambda_1', type: 'lambda', position: { x: 250, y: 100 } },
            { id: 'dynamodb_1', type: 'dynamodb', position: { x: 500, y: 100 } },
        ],
        connections: [
            { from: 'apigw_1', to: 'lambda_1' },
            { from: 'lambda_1', to: 'dynamodb_1' },
        ]
    },
    {
        name: 'SQS Processing for Java Backend',
        description: 'An asynchronous SQS queue processing architecture, ideal for Java backend developers.',
        nodes: [
            { id: 'sqs_apigw', type: 'apigw', position: { x: 0, y: 150 }, data: { label: 'Client API' } },
            { id: 'sqs_lambda_enqueue', type: 'lambda', position: { x: 200, y: 150 }, data: { label: 'Enqueue Request Lambda' } },
            { id: 'sqs_queue', type: 'sqs', position: { x: 450, y: 150 }, data: { label: 'Job Queue' } },
            { id: 'sqs_lambda_process', type: 'lambda', position: { x: 700, y: 150 }, data: { label: 'Process Message Lambda' } },
            { id: 'sqs_ec2_api', type: 'ec2', position: { x: 950, y: 50 }, data: { label: 'MCP Model Fetching API' } },
            { id: 'sqs_dynamodb', type: 'dynamodb', position: { x: 950, y: 250 }, data: { label: 'Results Table' } },
        ],
        connections: [
            { from: 'sqs_apigw', to: 'sqs_lambda_enqueue' },
            { from: 'sqs_lambda_enqueue', to: 'sqs_queue' },
            { from: 'sqs_queue', to: 'sqs_lambda_process' },
            { from: 'sqs_lambda_process', to: 'sqs_ec2_api' },
            { from: 'sqs_lambda_process', to: 'sqs_dynamodb' },
        ]
    }
];

// Updated AI/DL Templates to be Vertical (Top-Down)
export const AI_TEMPLATES: Template[] = [
    {
        name: 'Linear Regression Pipeline',
        description: 'Simple supervised learning pipeline for regression.',
        nodes: [
            { id: 'ai-start', type: 'start', position: { x: 400, y: 50 } },
            { id: 'ai-data', type: 'ai-dataset', position: { x: 400, y: 150 } },
            { id: 'ai-split', type: 'ai-split', position: { x: 400, y: 250 } },
            { id: 'ai-model', type: 'reg-linear', position: { x: 400, y: 350 } },
            { id: 'ai-end', type: 'end', position: { x: 400, y: 450 } },
        ],
        connections: [
            { from: 'ai-start', to: 'ai-data' },
            { from: 'ai-data', to: 'ai-split' },
            { from: 'ai-split', to: 'ai-model' },
            { from: 'ai-model', to: 'ai-end' },
        ]
    },
    {
        name: 'Random Forest Classification',
        description: 'A robust classification pipeline using Random Forest.',
        nodes: [
            { id: 'ai-start', type: 'start', position: { x: 400, y: 50 } },
            { id: 'ai-data', type: 'ai-dataset', position: { x: 400, y: 150 } },
            { id: 'ai-scaler', type: 'ai-scaler', position: { x: 400, y: 250 } },
            { id: 'ai-model', type: 'clf-rf', position: { x: 400, y: 350 } },
            { id: 'ai-end', type: 'end', position: { x: 400, y: 450 } },
        ],
        connections: [
            { from: 'ai-start', to: 'ai-data' },
            { from: 'ai-data', to: 'ai-scaler' },
            { from: 'ai-scaler', to: 'ai-model' },
            { from: 'ai-model', to: 'ai-end' },
        ]
    },
    {
        name: 'Vision Transformer (ViT) Classifier',
        description: 'Standard ViT architecture for ImageNet classification.',
        nodes: [
            { id: 'dl-start', type: 'start', position: { x: 400, y: 50 } },
            { id: 'aug', type: 'aug-transforms', position: { x: 400, y: 150 } },
            { id: 'vit', type: 'vit-base', position: { x: 400, y: 250 } },
            { id: 'train', type: 'train-basic', position: { x: 400, y: 350 } },
        ],
        connections: [
            { from: 'dl-start', to: 'aug' },
            { from: 'aug', to: 'vit' },
            { from: 'vit', to: 'train' },
        ]
    }
];
