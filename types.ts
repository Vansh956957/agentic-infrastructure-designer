
import React from 'react';

export interface Vector2D {
  x: number;
  y: number;
}

export type AWSServiceHealth = 'healthy' | 'warning' | 'critical';
export type TrafficLevel = 'low' | 'normal' | 'high';

export interface NodeData {
    label: string;
    health?: AWSServiceHealth;
    traffic?: TrafficLevel;
    // AI/DL Mode
    code?: string; // Default / PyTorch
    codeTF?: string; // TensorFlow / Keras alternative
    description?: string;
    customTheme?: string; 
    hyperparams?: Record<string, string | number>;
    
    // EC2
    instanceFamily?: string;
    instanceSize?: string;
    amiId?: string;
    keyPair?: string;
    securityGroup?: string;
    iamRole?: string;
    subnetId?: string;
    isPublicIp?: boolean;
    ebsSize?: number;
    ebsType?: string;
    ebsEncrypted?: boolean;
    // S3
    bucketName?: string;
    versioning?: boolean;
    storageClass?: string;
    encryption?: string;
    publicAccess?: boolean;
    // Lambda
    runtime?: string;
    memorySize?: number;
    timeout?: number;
    triggerType?: string;
    // RDS
    engine?: string;
    storageSize?: number;
    multiAZ?: boolean;
    // ALB
    listeners?: string;
    // TG
    targetType?: string;
    healthCheckPath?: string;
    port?: number;
    protocol?: string;
    // VPC
    cidr?: string;
    subnets?: string[];
    // CloudWatch
    metric?: string;
    threshold?: number;
    // General
    [key: string]: any;
}

export interface Node {
  id: string;
  type: string;
  position: Vector2D;
  data: NodeData;
  groupId?: string;
  isAnimating: boolean;
}

export interface Connection {
  id: string;
  fromNodeId: string;
  toNodeId: string;
  label?: string;
  isAnimating: boolean;
}

export interface AWSService {
  id: string;
  name: string;
  icon: React.FC<{ className?: string }>;
  defaultData: NodeData;
}

export interface AWSCategory {
    name: string;
    services: AWSService[];
}

export interface GroupData {
    label:string;
    borderStyle: 'dotted' | 'solid' | 'dashed';
    borderColor: string;
    backgroundColor: string;
    locked: boolean;
}

export interface Group {
  id: string;
  type: 'group';
  position: Vector2D;
  size: { width: number, height: number };
  data: GroupData;
  zIndex: number;
}

export interface ShapeData {
    fillColor: string;
    strokeColor: string;
    strokeWidth: number;
    borderStyle: 'solid' | 'dashed' | 'dotted';
}

export interface Shape {
    id: string;
    type: 'rectangle' | 'ellipse';
    position: Vector2D;
    size: { width: number, height: number };
    data: ShapeData;
    zIndex: number;
}

export interface TextNodeData {
    content: string;
    fontSize: number;
    color: string;
}

export interface TextNode {
    id: string;
    type: 'text';
    position: Vector2D;
    size: { width: number, height: number };
    data: TextNodeData;
    zIndex: number;
}

export interface AppState {
    nodes: Node[];
    connections: Connection[];
    groups: Group[];
    shapes: Shape[];
    textNodes: TextNode[];
    nextNodeId: number;
    nextConnectionId: number;
    nextGroupId: number;
    nextShapeId: number;
    nextTextNodeId: number;
}

export interface Snapshot {
    id: string;
    name: string;
    state: AppState;
}

export interface TemplateNode {
    id: string;
    type: string;
    position: { x: number, y: number };
    data?: Partial<NodeData>;
}
export interface TemplateConnection {
    from: string;
    to: string;
}
export interface Template {
    name: string;
    description: string;
    nodes: TemplateNode[];
    connections: TemplateConnection[];
}

export interface FixAction {
    type: 'add_node';
    nodeType: string;
    position: 'before' | 'after';
    label: string;
}

export interface ValidationIssue {
    nodeId: string;
    severity: 'warning' | 'error';
    message: string;
    missingComponent?: string; // e.g., 'vpc', 'subnet'
    fixAction?: FixAction;
}
