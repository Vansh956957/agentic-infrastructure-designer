
import { Node, Connection, ValidationIssue } from './types';

export const validateGraph = (nodes: Node[], connections: Connection[]): ValidationIssue[] => {
    const issues: ValidationIssue[] = [];

    const isConnectedToType = (nodeId: string, targetType: string, direction: 'incoming' | 'outgoing' | 'any' = 'any') => {
        return connections.some(c => {
            const isIncoming = c.toNodeId === nodeId;
            const isOutgoing = c.fromNodeId === nodeId;
            const otherNodeId = isIncoming ? c.fromNodeId : c.toNodeId;
            const otherNode = nodes.find(n => n.id === otherNodeId);
            
            if (!otherNode) return false;
            
            const typeMatch = otherNode.type === targetType;
            
            if (direction === 'incoming') return isIncoming && typeMatch;
            if (direction === 'outgoing') return isOutgoing && typeMatch;
            return typeMatch;
        });
    };

    const hasProperty = (node: Node, prop: string) => {
        return node.data[prop] !== undefined && node.data[prop] !== '' && node.data[prop] !== null;
    };

    nodes.forEach(node => {
        switch (node.type) {
            // --- NETWORKING ---
            case 'subnet':
                if (!isConnectedToType(node.id, 'vpc', 'any')) {
                    issues.push({
                        nodeId: node.id,
                        severity: 'error',
                        message: 'Subnet must be connected to a VPC.',
                        missingComponent: 'vpc',
                        fixAction: { type: 'add_node', nodeType: 'vpc', position: 'before', label: 'Add VPC' }
                    });
                }
                break;
            case 'igw':
                if (!isConnectedToType(node.id, 'vpc', 'any')) {
                     issues.push({ nodeId: node.id, severity: 'error', message: 'Internet Gateway must be attached to a VPC.', missingComponent: 'vpc', fixAction: { type: 'add_node', nodeType: 'vpc', position: 'after', label: 'Attach VPC' } });
                }
                break;
            case 'nat':
                if (!isConnectedToType(node.id, 'subnet', 'any')) {
                    issues.push({ nodeId: node.id, severity: 'error', message: 'NAT Gateway must be in a Subnet.', missingComponent: 'subnet', fixAction: { type: 'add_node', nodeType: 'subnet', position: 'before', label: 'Add Subnet' } });
                }
                break;
            
            // --- COMPUTE ---
            case 'ec2':
                if (!isConnectedToType(node.id, 'subnet', 'incoming') && !isConnectedToType(node.id, 'vpc', 'incoming')) {
                     issues.push({
                        nodeId: node.id,
                        severity: 'error',
                        message: 'EC2 requires a Subnet/VPC connection.',
                        missingComponent: 'subnet',
                        fixAction: { type: 'add_node', nodeType: 'subnet', position: 'before', label: 'Add Subnet' }
                    });
                }
                if (!isConnectedToType(node.id, 'sg', 'any')) {
                    issues.push({
                        nodeId: node.id,
                        severity: 'warning',
                        message: 'EC2 should have a Security Group.',
                        missingComponent: 'sg',
                        fixAction: { type: 'add_node', nodeType: 'sg', position: 'after', label: 'Add Security Group' }
                    });
                }
                break;
            
            case 'asg':
                if (!isConnectedToType(node.id, 'subnet', 'any') && !isConnectedToType(node.id, 'vpc', 'any')) {
                     issues.push({ nodeId: node.id, severity: 'error', message: 'Auto Scaling Group needs Subnets.', missingComponent: 'subnet', fixAction: { type: 'add_node', nodeType: 'subnet', position: 'before', label: 'Add Subnet' } });
                }
                break;

            case 'lambda':
                if (!hasProperty(node, 'runtime')) {
                     issues.push({ nodeId: node.id, severity: 'warning', message: 'Runtime not selected.' });
                }
                // Check for trigger (incoming connection from API GW, S3, SNS, SQS, EventBridge)
                const hasTrigger = connections.some(c => c.toNodeId === node.id);
                if (!hasTrigger) {
                     issues.push({
                        nodeId: node.id,
                        severity: 'warning',
                        message: 'Lambda has no trigger source.',
                        missingComponent: 'apigw',
                        fixAction: { type: 'add_node', nodeType: 'apigw', position: 'before', label: 'Add API Gateway' }
                     });
                }
                break;

            // --- DATABASE ---
            case 'rds':
                if (!isConnectedToType(node.id, 'subnet', 'any') && !isConnectedToType(node.id, 'vpc', 'any')) {
                     issues.push({ nodeId: node.id, severity: 'error', message: 'RDS requires a Subnet Group.', missingComponent: 'subnet', fixAction: { type: 'add_node', nodeType: 'subnet', position: 'before', label: 'Add Subnet Group' } });
                }
                break;
            
            // --- LOAD BALANCING ---
            case 'alb':
                if (!isConnectedToType(node.id, 'tg', 'outgoing')) {
                    issues.push({
                        nodeId: node.id,
                        severity: 'error',
                        message: 'Load Balancer must route to a Target Group.',
                        missingComponent: 'tg',
                        fixAction: { type: 'add_node', nodeType: 'tg', position: 'after', label: 'Add Target Group' }
                    });
                }
                // ALBs need subnets, usually indicated by incoming connections or placement
                if (!isConnectedToType(node.id, 'subnet', 'any') && !isConnectedToType(node.id, 'vpc', 'any')) {
                     issues.push({ nodeId: node.id, severity: 'warning', message: 'ALB needs Subnets.', missingComponent: 'subnet', fixAction: { type: 'add_node', nodeType: 'subnet', position: 'before', label: 'Add Subnet' } });
                }
                break;
            
            case 'tg':
                // Needs targets
                const hasTargets = connections.some(c => c.fromNodeId === node.id); // TG usually points to targets or targets reside in it. 
                // However, visually we often draw EC2 -> TG or TG -> EC2 depending on mental model. 
                // Let's assume TG -> EC2/Lambda based on flow.
                if (node.data.targetType === 'instance' && !isConnectedToType(node.id, 'ec2', 'any')) {
                     issues.push({ nodeId: node.id, severity: 'warning', message: 'Target Group has no registered instances.', missingComponent: 'ec2', fixAction: { type: 'add_node', nodeType: 'ec2', position: 'after', label: 'Add EC2 Instance' } });
                }
                break;

            // --- API & MESSAGING ---
            case 'apigw':
                if (!connections.some(c => c.fromNodeId === node.id)) {
                     issues.push({ nodeId: node.id, severity: 'warning', message: 'API Gateway has no integration.', missingComponent: 'lambda', fixAction: { type: 'add_node', nodeType: 'lambda', position: 'after', label: 'Add Lambda' } });
                }
                break;

            case 'sns':
                if (!connections.some(c => c.fromNodeId === node.id)) {
                     issues.push({ nodeId: node.id, severity: 'warning', message: 'SNS Topic has no subscribers.', missingComponent: 'sqs', fixAction: { type: 'add_node', nodeType: 'sqs', position: 'after', label: 'Add SQS Queue' } });
                }
                break;

            // --- STORAGE ---
            case 'ebs':
                if (!isConnectedToType(node.id, 'ec2', 'any')) {
                     issues.push({ nodeId: node.id, severity: 'warning', message: 'EBS Volume detached.', missingComponent: 'ec2', fixAction: { type: 'add_node', nodeType: 'ec2', position: 'before', label: 'Attach to EC2' } });
                }
                break;

            case 'efs':
                if (!isConnectedToType(node.id, 'subnet', 'any') && !isConnectedToType(node.id, 'vpc', 'any')) {
                     issues.push({ nodeId: node.id, severity: 'warning', message: 'EFS needs mount targets (Subnets).', missingComponent: 'subnet', fixAction: { type: 'add_node', nodeType: 'subnet', position: 'before', label: 'Add Subnet' } });
                }
                break;

            // --- CONTAINERS ---
            case 'ecs':
                 // Often needs ALB
                 if(node.data.launchType === 'FARGATE' || node.data.requiresALB) {
                     if(!isConnectedToType(node.id, 'alb', 'any')) {
                         issues.push({ nodeId: node.id, severity: 'warning', message: 'ECS Service requires Load Balancer.', missingComponent: 'alb', fixAction: { type: 'add_node', nodeType: 'alb', position: 'before', label: 'Add ALB' } });
                     }
                 }
                 break;
        }
    });

    return issues;
};
