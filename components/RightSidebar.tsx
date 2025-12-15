
import React, { useState, useEffect, useRef } from 'react';
import { Node, AWSServiceHealth, Connection, Group, GroupData, Shape, ShapeData, TextNode, TextNodeData, Vector2D, ValidationIssue, WorkflowStatus } from '../types';
import { LockIcon, AlignTopIcon, AlignMiddleIcon, AlignBottomIcon, AlignLeftIcon, AlignCenterIcon, AlignRightIcon, XIcon, StopIcon } from './Icons';
import { EC2_INSTANCE_TYPES, RDS_ENGINES, NODE_WIDTH, NODE_HEIGHT, PIPELINE_COMMON_CODE } from '../constants';

type Selection = { type: 'node', id: string } | { type: 'connection', id: string } | { type: 'group', id: string } | { type: 'shape', id: string } | { type: 'text', id: string };

type AlignmentDirection = 'top' | 'middle' | 'bottom' | 'left' | 'center' | 'right';
interface RightSidebarProps {
  selection: Selection[];
  node: Node | null;
  compareNode?: Node | null; // For comparison
  connection: Connection | null;
  group: Group | null;
  shape: Shape | null;
  textNode: TextNode | null;
  onUpdateNode: (nodeId: string, data: Partial<Node['data']>) => void;
  onUpdateConnection: (connId: string, data: Partial<Connection>) => void;
  onUpdateGroup: (groupId: string, data: Partial<GroupData>) => void;
  onUpdateShape: (shapeId: string, data: Partial<ShapeData>) => void;
  onUpdateTextNode: (textId: string, data: Partial<TextNodeData>) => void;
  onDeleteNode: (nodeId: string) => void;
  onDisconnectNode: (nodeId: string) => void;
  onDeleteConnection: (connId: string) => void;
  onDeleteGroup: (groupId: string) => void;
  onDeleteShape: (shapeId: string) => void;
  onDeleteTextNode: (textId: string) => void;
  onDeselect: () => void;
  onStartWorkflow: (nodeId: string) => void;
  onStopWorkflow?: () => void;
  workflowStatus?: WorkflowStatus;
  onAlign: (direction: AlignmentDirection) => void;
  isZenMode: boolean; // Kept for type compatibility but unused for visibility
  isOpen?: boolean; // New prop for visibility
  onClose: () => void; // New prop for closing
  compareMode?: boolean;
  setCompareMode?: (enabled: boolean) => void;
  nodes?: Node[]; 
  validationIssues?: ValidationIssue[];
  appMode?: 'aws' | 'ai' | 'dl';
  tutorialGlowButton?: 'compare' | 'tensorflow' | null;
  tutorialMessage?: string | null;
  onPauseWorkflow?: () => void;
}

const GeneralNodeConfig: React.FC<{node: Node, onUpdate: (data: any) => void, onStartWorkflow: () => void, onStopWorkflow?: () => void, workflowStatus?: WorkflowStatus}> = ({node, onUpdate, onStartWorkflow, onStopWorkflow, workflowStatus}) => (
    <>
         <div>
          <label className="block text-sm font-medium text-gray-500 dark:text-gray-400">Label</label>
          <input
            type="text"
            value={node.data.label}
            onChange={(e) => onUpdate({label: e.target.value})}
            className="mt-1 block w-full bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-md shadow-sm py-2 px-3 text-gray-900 dark:text-white focus:outline-none focus:ring-orange-500 focus:border-orange-500 sm:text-sm"
          />
        </div>
        {node.data.health !== undefined && <div>
          <label className="block text-sm font-medium text-gray-500 dark:text-gray-400">Health Status</label>
          <select
            value={node.data.health}
            onChange={(e) => onUpdate({health: e.target.value as AWSServiceHealth})}
            className="mt-1 block w-full bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-md shadow-sm py-2 px-3 text-gray-900 dark:text-white focus:outline-none focus:ring-orange-500 focus:border-orange-500 sm:text-sm"
            disabled={node.type === 'asg'}
          >
            <option value="healthy">Healthy</option>
            <option value="warning">Warning</option>
            <option value="critical">Critical</option>
          </select>
        </div>}
        {node.type === 'start' &&
            <button
                onClick={workflowStatus === 'running' ? onStopWorkflow : onStartWorkflow}
                className={`w-full mt-4 text-white font-bold py-2 px-4 rounded transition-colors shadow-sm flex items-center justify-center gap-2 ${
                    workflowStatus === 'running' 
                    ? 'bg-red-600 hover:bg-red-700 animate-pulse' 
                    : 'bg-green-600 hover:bg-green-700'
                }`}
            >
                {workflowStatus === 'running' && (
                    <>
                        <StopIcon className="w-4 h-4" />
                        Stop Workflow
                    </>
                )}
                {workflowStatus !== 'running' && (
                    <>
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd"></path></svg>
                        Run Workflow
                    </>
                )}
            </button>
        }
    </>
)

const CodeInspector: React.FC<{ 
    node: Node, 
    compareNode: Node | null, 
    compareMode: boolean, 
    appMode?: string,
    glowButton?: 'compare' | 'tensorflow' | null 
}> = ({ node, compareNode, compareMode, appMode, glowButton }) => {
    const [framework, setFramework] = useState<'pytorch' | 'tensorflow'>('pytorch');
    const highlightSpecific = true;
    const scrollRef = useRef<HTMLDivElement>(null);

    const currentCode = framework === 'pytorch' ? (node.data.code || '') : (node.data.codeTF || '# No TensorFlow code available');
    const compareCode = compareNode ? (framework === 'pytorch' ? compareNode.data.code : compareNode.data.codeTF) : undefined;

    // Auto-scroll to highlighted content when it appears
    useEffect(() => {
        if (compareMode && scrollRef.current) {
            const highlightedElement = scrollRef.current.querySelector('.bg-green-900\\/50');
            if (highlightedElement) {
                highlightedElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }
    }, [compareMode, compareNode]);

    const renderCodeBlock = (code: string, comparisonCode?: string, highlightType?: 'add' | 'remove') => {
        const lines = code.split('\n');
        const compLines = comparisonCode ? comparisonCode.split('\n') : [];
        const commonLines = PIPELINE_COMMON_CODE.split('\n');

        return (
            <div ref={scrollRef} className="flex-1 overflow-auto bg-gray-900 text-gray-100 p-3 rounded-md font-mono text-xs border border-gray-700 shadow-inner h-full min-h-[300px]">
                {lines.map((line, idx) => {
                    let className = "px-1 ";
                    
                    if (highlightType === 'add') {
                        if (!compLines.includes(line) && line.trim() !== '') {
                            className += "bg-green-900/50 text-green-200 block w-full";
                        }
                    } else if (highlightType === 'remove') {
                        if (!compLines.includes(line) && line.trim() !== '') {
                            className += "bg-red-900/50 text-red-200 block w-full";
                        }
                    } else {
                        if (highlightSpecific && !commonLines.includes(line) && line.trim() !== '') {
                            className += "bg-yellow-500/30 block w-full"; 
                        }
                    }

                    return (
                        <div key={idx} className={className} style={{ minHeight: '1.2em' }}>
                            {line || '\n'}
                        </div>
                    );
                })}
            </div>
        );
    };

    if (compareMode) {
        return (
            <div className="flex flex-col h-full overflow-hidden gap-2">
                <div className="grid grid-cols-2 gap-2 h-full">
                    <div className="flex flex-col h-full overflow-hidden">
                        <div className="text-xs font-bold text-gray-500 mb-1 flex justify-between">
                            <span>{node.data.label} (Base)</span>
                            <span className="w-2 h-2 rounded-full bg-red-500"></span>
                        </div>
                        {renderCodeBlock(currentCode, compareCode, 'remove')}
                    </div>
                    <div className="flex flex-col h-full overflow-hidden">
                        <div className="text-xs font-bold text-gray-500 mb-1 flex justify-between">
                            <span>{compareNode ? compareNode.data.label : 'Select Node...'}</span>
                            <span className="w-2 h-2 rounded-full bg-green-500"></span>
                        </div>
                        {compareNode ? renderCodeBlock(compareCode || '', currentCode, 'add') : (
                            <div className="flex items-center justify-center h-full bg-gray-100 dark:bg-gray-800 border border-dashed border-gray-400 rounded text-gray-500 text-center p-4">
                                Select another node to compare
                            </div>
                        )}
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="flex flex-col h-full overflow-hidden space-y-4">
             <div className="flex justify-between items-center pb-2 border-b border-gray-200 dark:border-gray-700">
                <h3 className="font-bold text-lg">{node.data.label}</h3>
                
                {/* Framework Toggle - Hidden for ML (appMode === 'ai') per request */}
                {node.data.codeTF && appMode !== 'ai' ? (
                    <div className="flex bg-gray-200 dark:bg-gray-700 rounded-md p-0.5">
                        <button 
                            onClick={() => setFramework('pytorch')}
                            className={`px-2 py-1 text-[10px] font-bold rounded ${framework === 'pytorch' ? 'bg-white dark:bg-gray-600 shadow text-orange-600 dark:text-white' : 'text-gray-500 dark:text-gray-400'}`}
                        >
                            PyTorch
                        </button>
                        <button 
                            onClick={() => setFramework('tensorflow')}
                            className={`px-2 py-1 text-[10px] font-bold rounded ${framework === 'tensorflow' ? 'bg-white dark:bg-gray-600 shadow text-orange-600 dark:text-white' : 'text-gray-500 dark:text-gray-400'} ${glowButton === 'tensorflow' ? 'ring-2 ring-orange-500 animate-pulse' : ''}`}
                        >
                            TensorFlow
                        </button>
                    </div>
                ) : (
                    <span className="text-xs bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded text-gray-600 dark:text-gray-300">PyTorch</span>
                )}
            </div>
            
            {node.data.description && (
                <div className="text-sm text-gray-600 dark:text-gray-400 italic">
                    {node.data.description}
                </div>
            )}

            {node.data.hyperparams && (
                <div className="bg-gray-50 dark:bg-gray-900 rounded p-2 border border-gray-200 dark:border-gray-700">
                    <h4 className="text-xs font-bold uppercase text-gray-500 mb-2">Hyperparameters</h4>
                    <div className="grid grid-cols-2 gap-2">
                        {Object.entries(node.data.hyperparams).map(([key, val]) => (
                            <div key={key} className="flex justify-between items-center text-xs">
                                <span className="text-gray-500">{key}:</span>
                                <span className="font-mono text-gray-800 dark:text-gray-200 bg-white dark:bg-gray-800 px-1 rounded border border-gray-200 dark:border-gray-600">{val}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <div className="flex-1 overflow-hidden flex flex-col">
                <h4 className="text-xs font-bold uppercase text-gray-500 mb-1">Implementation Template</h4>
                {renderCodeBlock(currentCode)}
            </div>

            <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                <button onClick={() => navigator.clipboard.writeText(currentCode)} className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded text-sm transition-colors shadow-sm">
                    Copy Code Template
                </button>
            </div>
        </div>
    );
};

const ActionSection: React.FC<{node: Node, issues: ValidationIssue[], onUpdate: (data: any) => void}> = ({node, issues, onUpdate}) => {
    const nodeIssues = issues.filter(i => i.nodeId === node.id);
    const hasError = nodeIssues.some(i => i.severity === 'error');
    
    return (
        <div className="bg-gray-100 dark:bg-gray-700/50 rounded-lg p-3 border border-gray-200 dark:border-gray-600 space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-600 pb-1">Actions & Status</h4>
            
            {/* Status Indicator */}
            <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${hasError ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`}></div>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    {hasError ? 'Configuration Issues' : 'Ready'}
                </span>
            </div>

            {/* Validation Messages */}
            {nodeIssues.length > 0 && (
                <div className="space-y-2">
                    {nodeIssues.map((issue, idx) => (
                        <div key={idx} className={`text-xs p-2 rounded border ${issue.severity === 'error' ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-700 dark:text-red-300' : 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800 text-yellow-700 dark:text-yellow-300'}`}>
                            {issue.message}
                        </div>
                    ))}
                </div>
            )}

            {/* Quick Actions */}
            <div className="grid grid-cols-2 gap-2">
                 {node.type === 'ec2' && (
                     <>
                        <button className="bg-green-600 hover:bg-green-700 text-white text-xs font-bold py-1 px-2 rounded" onClick={() => onUpdate({health: 'healthy'})}>Start</button>
                        <button className="bg-red-600 hover:bg-red-700 text-white text-xs font-bold py-1 px-2 rounded" onClick={() => onUpdate({health: 'critical'})}>Stop</button>
                     </>
                 )}
            </div>
        </div>
    )
}

// ... [ServiceSpecificConfig, ConfigInput, ConfigSelect, ConfigCheckbox, GroupConfig, ShapeConfig, TextConfig, AlignmentPanel components remain unchanged] ...
const ServiceSpecificConfig: React.FC<{node: Node, onUpdate: (data: any) => void}> = ({node, onUpdate}) => {
    switch(node.type) {
        case 'ec2':
            return <>
                <div className="flex space-x-2">
                    <ConfigSelect label="Family" value={node.data.instanceFamily} onChange={v => onUpdate({instanceFamily: v, instanceSize: EC2_INSTANCE_TYPES[v][0]})} options={Object.keys(EC2_INSTANCE_TYPES)} />
                    <ConfigSelect label="Size" value={node.data.instanceSize} onChange={v => onUpdate({instanceSize: v})} options={EC2_INSTANCE_TYPES[node.data.instanceFamily] || []} />
                </div>
                <ConfigInput label="AMI ID" value={node.data.amiId} onChange={v => onUpdate({amiId: v})} />
                <ConfigInput label="Security Group" value={node.data.securityGroup} onChange={v => onUpdate({securityGroup: v})} />
            </>
        case 'lambda':
            return <>
                <ConfigSelect label="Trigger" value={node.data.triggerType} onChange={v => onUpdate({triggerType: v})} options={['API Gateway', 'S3', 'EventBridge', 'SQS', 'DynamoDB']} />
                <ConfigSelect label="Runtime" value={node.data.runtime} onChange={v => onUpdate({runtime: v})} options={['Node.js 20.x', 'Python 3.12', 'Go 1.x', 'Java 21']} />
                <ConfigInput label="Memory (MB)" type="number" value={node.data.memorySize} onChange={v => onUpdate({memorySize: parseInt(v)})} />
                <ConfigInput label="Timeout (s)" type="number" value={node.data.timeout} onChange={v => onUpdate({timeout: parseInt(v)})} />
            </>
        case 'rds':
            return <>
                <ConfigSelect label="Engine" value={node.data.engine} onChange={v => onUpdate({engine: v})} options={RDS_ENGINES} />
                <ConfigInput label="Storage (GB)" type="number" value={node.data.storageSize} onChange={v => onUpdate({storageSize: parseInt(v)})} />
                <ConfigCheckbox label="Read Replicas" checked={node.data.readReplicas} onChange={v => onUpdate({readReplicas: v})} />
            </>
        case 'asg':
            return <>
                <ConfigSelect label="Scaling Policy" value={node.data.scalingPolicy} onChange={v => onUpdate({scalingPolicy: v})} options={['manual', 'traffic-based']} />
                <div className="grid grid-cols-3 gap-2">
                  <ConfigInput label="Min" type="number" value={node.data.minSize} onChange={v => onUpdate({minSize: v})} />
                  <ConfigInput label="Max" type="number" value={node.data.maxSize} onChange={v => onUpdate({maxSize: v})} />
                  <ConfigInput label="Desired" type="number" value={node.data.desiredCapacity} onChange={v => onUpdate({desiredCapacity: v})} />
                </div>
            </>
        case 'ami':
            return <>
                <ConfigInput label="Base OS" value={node.data.baseOs} onChange={v => onUpdate({baseOs: v})} />
                <ConfigInput label="Version" value={node.data.version} onChange={v => onUpdate({version: v})} />
            </>
        case 's3':
            return <>
                <ConfigCheckbox label="Versioning" checked={node.data.versioning} onChange={v => onUpdate({versioning: v})} />
                <ConfigCheckbox label="Encryption" checked={!!node.data.encryption} onChange={v => onUpdate({encryption: v ? 'SSE-S3' : undefined})} />
                <ConfigCheckbox label="Public Access" checked={node.data.publicAccess} onChange={v => onUpdate({publicAccess: v})} />
            </>
        case 'dynamodb':
            return <>
                 <ConfigInput label="Partition Key" value={node.data.partitionKey || 'id'} onChange={v => onUpdate({partitionKey: v})} />
                 <ConfigSelect label="Billing Mode" value={node.data.billingMode || 'PROVISIONED'} onChange={v => onUpdate({billingMode: v})} options={['PROVISIONED', 'PAY_PER_REQUEST']} />
                 <ConfigCheckbox label="Stream Enabled" checked={node.data.streamEnabled || false} onChange={v => onUpdate({streamEnabled: v})} />
            </>
        case 'sqs':
            return <>
                <ConfigCheckbox label="FIFO Queue" checked={node.data.fifo || false} onChange={v => onUpdate({fifo: v})} />
                <ConfigInput label="Retention (Days)" type="number" value={node.data.retentionPeriod || 4} onChange={v => onUpdate({retentionPeriod: parseInt(v)})} />
            </>
        case 'sns':
             return <>
                <ConfigInput label="Topic Name" value={node.data.topicName || 'my-topic'} onChange={v => onUpdate({topicName: v})} />
                <ConfigCheckbox label="FIFO Topic" checked={node.data.fifo || false} onChange={v => onUpdate({fifo: v})} />
            </>
        case 'cloudfront':
            return <>
                <ConfigInput label="Origin Domain" value={node.data.originDomain || ''} onChange={v => onUpdate({originDomain: v})} />
                <ConfigSelect label="Price Class" value={node.data.priceClass || 'PriceClass_100'} onChange={v => onUpdate({priceClass: v})} options={['PriceClass_100', 'PriceClass_200', 'PriceClass_All']} />
            </>
        case 'r53':
            return <>
                <ConfigInput label="Hosted Zone" value={node.data.hostedZone || 'example.com'} onChange={v => onUpdate({hostedZone: v})} />
                <ConfigSelect label="Record Type" value={node.data.recordType || 'A'} onChange={v => onUpdate({recordType: v})} options={['A', 'AAAA', 'CNAME', 'MX', 'TXT']} />
            </>
        case 'apigw':
            return <>
                <ConfigSelect label="Endpoint Type" value={node.data.endpointType || 'REGIONAL'} onChange={v => onUpdate({endpointType: v})} options={['REGIONAL', 'EDGE', 'PRIVATE']} />
            </>
        case 'tg':
             return <>
                <ConfigSelect label="Target Type" value={node.data.targetType || 'instance'} onChange={v => onUpdate({targetType: v})} options={['instance', 'ip', 'lambda', 'alb']} />
                <ConfigInput label="Health Check Path" value={node.data.healthCheckPath || '/'} onChange={v => onUpdate({healthCheckPath: v})} />
             </>
        default:
            return null;
    }
}

const ConfigInput: React.FC<{label: string, value: any, onChange: (val: any) => void, type?: string, step?: number}> = ({label, value, onChange, type="text", step}) => (
    <div>
        <label className="block text-sm font-medium text-gray-500 dark:text-gray-400">{label}</label>
        <input type={type} value={value} onChange={e => onChange(e.target.value)} step={step}
         className="mt-1 block w-full bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-md shadow-sm py-2 px-3 text-gray-900 dark:text-white focus:outline-none focus:ring-orange-500 focus:border-orange-500 sm:text-sm" />
    </div>
);

const ConfigSelect: React.FC<{label: string, value: any, onChange: (val: any) => void, options: string[] | {label: string, value: string}[]}> = ({label, value, onChange, options}) => (
    <div className="w-full">
        <label className="block text-sm font-medium text-gray-500 dark:text-gray-400">{label}</label>
        <select value={value} onChange={e => onChange(e.target.value)} className="mt-1 block w-full bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-md shadow-sm py-2 px-3 text-gray-900 dark:text-white focus:outline-none focus:ring-orange-500 focus:border-orange-500 sm:text-sm">
            {options.map(o => typeof o === 'string' 
                ? <option key={o} value={o}>{o}</option>
                : <option key={o.value} value={o.value}>{o.label}</option>
            )}
        </select>
    </div>
)

const ConfigCheckbox: React.FC<{label: string, checked: boolean, onChange: (val: boolean) => void}> = ({label, checked, onChange}) => (
    <div className="flex items-center pt-2">
        <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} className="h-4 w-4 bg-gray-50 dark:bg-gray-700 border-gray-300 dark:border-gray-600 rounded text-orange-500 focus:ring-orange-500" />
        <label className="ml-2 block text-sm text-gray-500 dark:text-gray-400">{label}</label>
    </div>
)

const GroupConfig: React.FC<{group: Group, onUpdate: (data: any) => void}> = ({group, onUpdate}) => (
    <>
        <ConfigInput label="Label" value={group.data.label} onChange={v => onUpdate({label: v})} />
        <ConfigSelect label="Border Style" value={group.data.borderStyle} onChange={v => onUpdate({borderStyle: v})} options={['dotted', 'dashed', 'solid']} />
        <ConfigInput label="Border Color" type="color" value={group.data.borderColor} onChange={v => onUpdate({borderColor: v})} />
        <ConfigInput label="Background Color" type="color" value={group.data.backgroundColor} onChange={v => onUpdate({backgroundColor: v})} />
        <ConfigCheckbox label="Lock Position & Size" checked={group.data.locked} onChange={v => onUpdate({locked: v})} />
    </>
)

const ShapeConfig: React.FC<{shape: Shape, onUpdate: (data: any) => void}> = ({shape, onUpdate}) => (
    <>
      <ConfigInput label="Fill Color" type="color" value={shape.data.fillColor} onChange={v => onUpdate({fillColor: v})} />
      <ConfigInput label="Stroke Color" type="color" value={shape.data.strokeColor} onChange={v => onUpdate({strokeColor: v})} />
      <ConfigInput label="Stroke Width" type="number" value={shape.data.strokeWidth} onChange={v => onUpdate({strokeWidth: parseInt(v)})} />
      <ConfigSelect label="Border Style" value={shape.data.borderStyle} onChange={v => onUpdate({borderStyle: v})} options={['solid', 'dashed', 'dotted']} />
    </>
)

const TextConfig: React.FC<{textNode: TextNode, onUpdate: (data: Partial<TextNodeData>) => void}> = ({textNode, onUpdate}) => (
    <>
        <div>
            <label className="block text-sm font-medium text-gray-500 dark:text-gray-400">Content</label>
            <textarea
                value={textNode.data.content}
                onChange={(e) => onUpdate({ content: e.target.value })}
                className="mt-1 block w-full bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-md shadow-sm py-2 px-3 text-gray-900 dark:text-white focus:outline-none focus:ring-orange-500 focus:border-orange-500 sm:text-sm"
                rows={3}
            />
        </div>
        <ConfigInput label="Font Size (px)" type="number" value={textNode.data.fontSize} onChange={v => onUpdate({ fontSize: parseInt(v) })} />
        <ConfigInput label="Color" type="color" value={textNode.data.color} onChange={v => onUpdate({ color: v })} />
    </>
)

const AlignmentPanel: React.FC<{onAlign: (direction: AlignmentDirection) => void}> = ({ onAlign }) => {
    const alignmentButtons: { title: string; direction: AlignmentDirection; icon: React.FC<{className?: string}> }[] = [
        { title: 'Align Left', direction: 'left', icon: AlignLeftIcon },
        { title: 'Align Center', direction: 'center', icon: AlignCenterIcon },
        { title: 'Align Right', direction: 'right', icon: AlignRightIcon },
        { title: 'Align Top', direction: 'top', icon: AlignTopIcon },
        { title: 'Align Middle', direction: 'middle', icon: AlignMiddleIcon },
        { title: 'Align Bottom', direction: 'bottom', icon: AlignBottomIcon },
    ];

    return (
        <div>
            <label className="block text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">Align Nodes</label>
            <div className="grid grid-cols-3 gap-2">
                {alignmentButtons.map(({ title, direction, icon: Icon }) => (
                    <button key={direction} title={title} onClick={() => onAlign(direction)} className="p-2 bg-gray-100 dark:bg-gray-700 rounded-md hover:bg-orange-500 hover:text-white dark:hover:bg-orange-500 transition-colors flex justify-center items-center text-gray-600 dark:text-gray-300">
                        <Icon className="w-5 h-5" />
                    </button>
                ))}
            </div>
        </div>
    )
}

export const RightSidebar: React.FC<RightSidebarProps> = ({ selection, node, compareNode, connection, group, shape, textNode, onUpdateNode, onUpdateConnection, onUpdateGroup, onUpdateShape, onUpdateTextNode, onDeleteNode, onDisconnectNode, onDeleteConnection, onDeleteGroup, onDeleteShape, onDeleteTextNode, onDeselect, onStartWorkflow, onStopWorkflow, onAlign, isZenMode, isOpen, onClose, compareMode, setCompareMode, nodes, validationIssues = [], appMode, tutorialGlowButton, tutorialMessage, workflowStatus, onPauseWorkflow }) => {
  
  const handleAutoArrangeTG = (tgNode: Node) => {
      if (!nodes) return;
      const tgRect = {
          x: tgNode.position.x,
          y: tgNode.position.y,
          w: 300,
          h: 200
      };
      
      const children = nodes.filter(n => {
          if (n.id === tgNode.id) return false;
          const cx = n.position.x + NODE_WIDTH/2;
          const cy = n.position.y + NODE_HEIGHT/2;
          return cx > tgRect.x && cx < tgRect.x + tgRect.w && cy > tgRect.y && cy < tgRect.y + tgRect.h;
      });

      const cols = 2;
      children.forEach((child, index) => {
          onUpdateNode(child.id, {} as any); // Trigger update
      });
  }

  const renderContent = () => {
    if (selection.length > 1) {
        return (
            <div className="space-y-4 flex-1 min-w-[200px]">
                <AlignmentPanel onAlign={onAlign} />
            </div>
        )
    }

    if (selection.length === 1) {
        const singleSelection = selection[0];
        switch (singleSelection.type) {
            case 'node':
                if (!node) return null;
                
                // --- CODE INSPECTOR FOR AI/DL MODE ---
                if (node.data.code) {
                    return (
                        <div className="flex flex-col h-full overflow-hidden">
                            {setCompareMode && (
                                <div className="mb-2 flex items-center justify-end relative">
                                    <button 
                                        onClick={() => setCompareMode(!compareMode)}
                                        className={`text-xs px-2 py-1 rounded font-bold border transition-all ${compareMode ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white dark:bg-gray-700 text-gray-600 dark:text-gray-300 border-gray-300 dark:border-gray-600'} ${tutorialGlowButton === 'compare' ? 'ring-2 ring-orange-500 animate-pulse shadow-lg shadow-orange-500/50' : ''}`}
                                    >
                                        {compareMode ? 'Disable Comparison' : 'Compare Code'}
                                    </button>
                                    {tutorialMessage && (
                                        <div className="absolute top-full right-0 mt-2 bg-blue-600 text-white text-[10px] p-2 rounded shadow-xl whitespace-nowrap animate-bounce z-50">
                                            <div className="absolute -top-1 right-3 w-2 h-2 bg-blue-600 transform rotate-45"></div>
                                            {tutorialMessage}
                                        </div>
                                    )}
                                </div>
                            )}
                            <CodeInspector 
                                node={node} 
                                compareNode={compareNode || null} 
                                compareMode={!!compareMode} 
                                appMode={appMode} 
                                glowButton={tutorialGlowButton}
                            />
                        </div>
                    );
                }

                // --- STANDARD PROPERTIES ---
                return (
                    <>
                      <div className="space-y-4 flex-1 overflow-y-auto pr-1 min-w-[200px]" onMouseDown={() => { if(onPauseWorkflow) onPauseWorkflow(); }}>
                          <ActionSection node={node} issues={validationIssues} onUpdate={(data) => onUpdateNode(node.id, data)} />
                          <GeneralNodeConfig node={node} onUpdate={(data) => onUpdateNode(node.id, data)} onStartWorkflow={() => onStartWorkflow(node.id)} onStopWorkflow={onStopWorkflow} workflowStatus={workflowStatus} />
                          <ServiceSpecificConfig node={node} onUpdate={(data) => onUpdateNode(node.id, data)} />
                          
                          {node.type === 'tg' && (
                              <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded text-sm" 
                                onClick={() => handleAutoArrangeTG(node)} title="This would arrange nodes spatially if enabled">
                                  Auto-Arrange Children
                              </button>
                          )}
                      </div>
                      <div className="mt-auto pt-4 border-t border-gray-200 dark:border-gray-700 min-w-[200px] space-y-2">
                          <button onClick={() => onDisconnectNode(node.id)} className="w-full bg-orange-600 hover:bg-orange-700 text-white font-bold py-2 px-4 rounded transition-colors shadow-sm text-sm">Remove Arrow Pairs</button>
                          <button onClick={() => onDeleteNode(node.id)} className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded transition-colors shadow-sm text-sm">Delete Component</button>
                      </div>
                    </>
                );
            case 'connection':
                if (!connection) return null;
                return (
                    <>
                      <div className="space-y-4 flex-1 min-w-[200px]">
                          <ConfigInput label="Label" value={connection.label || ''} onChange={v => onUpdateConnection(connection.id, {label: v})} />
                      </div>
                       <div className="mt-auto pt-4 border-t border-gray-200 dark:border-gray-700 min-w-[200px]">
                          <button onClick={() => onDeleteConnection(connection.id)} className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded transition-colors shadow-sm">Delete Connection</button>
                      </div>
                    </>
                );
            case 'group':
                if (!group) return null;
                return (
                     <>
                      <div className="space-y-4 flex-1 min-w-[200px]">
                          <GroupConfig group={group} onUpdate={(data) => onUpdateGroup(group.id, data)} />
                      </div>
                       <div className="mt-auto pt-4 border-t border-gray-200 dark:border-gray-700 min-w-[200px]">
                          <button onClick={() => onDeleteGroup(group.id)} className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded transition-colors shadow-sm">Delete Group</button>
                      </div>
                    </>
                )
            case 'shape':
                if (!shape) return null;
                return (
                    <>
                        <div className="space-y-4 flex-1 min-w-[200px]">
                            <ShapeConfig shape={shape} onUpdate={(data) => onUpdateShape(shape.id, data)} />
                        </div>
                        <div className="mt-auto pt-4 border-t border-gray-200 dark:border-gray-700 min-w-[200px]">
                            <button onClick={() => onDeleteShape(shape.id)} className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded transition-colors shadow-sm">Delete Shape</button>
                        </div>
                    </>
                )
            case 'text':
                if (!textNode) return null;
                return (
                    <>
                        <div className="space-y-4 flex-1 min-w-[200px]">
                            <TextConfig textNode={textNode} onUpdate={(data) => onUpdateTextNode(textNode.id, data)} />
                        </div>
                        <div className="mt-auto pt-4 border-t border-gray-200 dark:border-gray-700 min-w-[200px]">
                            <button onClick={() => onDeleteTextNode(textNode.id)} className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded transition-colors shadow-sm">Delete Text</button>
                        </div>
                    </>
                )
            default:
                return null;
        }
    }
    return null;
  }
    
  return (
        <aside className={`bg-white dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700/50 flex flex-col shadow-lg transition-all duration-300 ease-in-out overflow-hidden ${!isOpen ? 'w-0 p-0 border-none opacity-0' : compareMode ? 'w-[600px] p-4 opacity-100' : 'w-96 p-4 opacity-100'}`}>
            <div className="flex justify-between items-center mb-4 min-w-[200px] border-b border-gray-200 dark:border-gray-700 pb-2">
                <h2 className="text-lg font-bold text-gray-800 dark:text-white">
                    {node?.data.code ? 'Inspector' : 'Properties'}
                </h2>
                <button onClick={onClose} className="text-gray-500 hover:text-red-500 transition-colors p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700">
                    <XIcon className="w-5 h-5" />
                </button>
            </div>
            {renderContent() || (
                <div className="flex flex-col justify-center items-center text-gray-400 dark:text-gray-500 h-full min-w-[200px]">
                    <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    <p className="text-center">Select an item to see its properties.</p>
                </div>
            )}
        </aside>
    );
};
