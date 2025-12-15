
import React, { useRef, useState, useCallback, MouseEvent as ReactMouseEvent } from 'react';
import { Node, Vector2D, NodeData } from '../types';
import { ALL_SERVICES, NODE_WIDTH, NODE_HEIGHT, WORKFLOW_NODE_SIZE } from '../constants';
import { GearIcon, PlusIcon, TrashIcon, DisconnectIcon, XIcon, TextIcon } from './Icons';

interface DraggableNodeProps {
  node: Node;
  onMove: (nodeId: string, newPosition: Vector2D) => void;
  onMoveEnd: () => void;
  onStartConnection: (nodeId: string) => void;
  onEndConnection: (nodeId: string) => void;
  isSelected: boolean;
  onSelect: (nodeId: string, isMulti: boolean) => void;
  onAddRelated: (nodeId: string) => void;
  transformScale: number;
  isAnimating: boolean;
  onStartWorkflow: () => void;
  onStopWorkflow: () => void;
  isWorkflowRunning: boolean;
  onUpdateNode: (nodeId: string, data: Partial<NodeData>) => void;
  onDeleteNode: (nodeId: string) => void;
  onDisconnectNode: (nodeId: string) => void;
  isConnected?: boolean;
}

const healthColorMap = {
    healthy: 'bg-green-500',
    warning: 'bg-yellow-500',
    critical: 'bg-red-500',
};

const GRID_SNAP = 10;

const MiniPropertiesMenu: React.FC<{
    node: Node,
    onClose: () => void,
    onUpdate: (data: Partial<NodeData>) => void,
    onDelete: () => void,
    onDisconnect: () => void,
    onAddRelated: () => void,
    onViewCode: () => void
}> = ({ node, onClose, onUpdate, onDelete, onDisconnect, onAddRelated, onViewCode }) => {
    return (
        <div 
            className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 bg-white dark:bg-gray-800 shadow-xl rounded-lg p-3 border border-gray-200 dark:border-gray-700 z-[1000] w-64 flex flex-col gap-3"
            onMouseDown={e => e.stopPropagation()}
            onTouchStart={e => e.stopPropagation()}
        >
            <div className="flex justify-between items-center border-b border-gray-100 dark:border-gray-700 pb-2">
                <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Quick Edit</span>
                <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"><XIcon className="w-4 h-4"/></button>
            </div>
            
            <div>
                <label className="block text-[10px] font-semibold text-gray-400 mb-1 uppercase">Label</label>
                <input 
                    type="text" 
                    value={node.data.label} 
                    onChange={(e) => onUpdate({label: e.target.value})}
                    className="w-full text-sm border rounded px-2 py-1.5 bg-gray-50 dark:bg-gray-900 border-gray-300 dark:border-gray-600 text-gray-800 dark:text-white focus:ring-2 focus:ring-orange-500 outline-none transition-colors"
                />
            </div>

            {node.data.health !== undefined && node.type !== 'asg' && (
                <div>
                    <label className="block text-[10px] font-semibold text-gray-400 mb-1 uppercase">Health Status</label>
                    <div className="flex bg-gray-100 dark:bg-gray-900 p-1 rounded-md">
                        {(['healthy', 'warning', 'critical'] as const).map((status) => (
                            <button
                                key={status}
                                onClick={() => onUpdate({ health: status })}
                                className={`flex-1 py-1 text-[10px] font-bold uppercase rounded-sm transition-all ${
                                    node.data.health === status 
                                    ? (status === 'healthy' ? 'bg-green-500 text-white shadow-sm' : status === 'warning' ? 'bg-yellow-500 text-white shadow-sm' : 'bg-red-500 text-white shadow-sm')
                                    : 'text-gray-500 dark:text-gray-400 hover:bg-white dark:hover:bg-gray-700'
                                }`}
                            >
                                {status}
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {node.type.startsWith('bu-') && (
                 <div>
                    <label className="block text-[10px] font-semibold text-gray-400 mb-1 uppercase">Traffic Level</label>
                    <div className="flex bg-gray-100 dark:bg-gray-900 p-1 rounded-md">
                        {(['low', 'normal', 'high'] as const).map((level) => (
                            <button
                                key={level}
                                onClick={() => onUpdate({ traffic: level })}
                                className={`flex-1 py-1 text-[10px] font-bold uppercase rounded-sm transition-all ${
                                    node.data.traffic === level 
                                    ? (level === 'high' ? 'bg-red-500 text-white shadow-sm' : level === 'normal' ? 'bg-blue-500 text-white shadow-sm' : 'bg-green-500 text-white shadow-sm')
                                    : 'text-gray-500 dark:text-gray-400 hover:bg-white dark:hover:bg-gray-700'
                                }`}
                            >
                                {level}
                            </button>
                        ))}
                    </div>
                </div>
            )}
            
             {node.type === 'tg' && (
                 <button 
                    onClick={onAddRelated}
                    className="w-full py-1.5 text-xs bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-800/30 rounded-md hover:bg-blue-100 dark:hover:bg-blue-900/40"
                 >
                     + Add Child Component
                 </button>
            )}

            {node.data.code && (
                <button
                    onClick={onViewCode}
                    className="w-full py-1.5 text-xs bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 border border-indigo-200 dark:border-indigo-800/30 rounded-md hover:bg-indigo-100 dark:hover:bg-indigo-900/40 flex items-center justify-center gap-1"
                >
                    <span>{ } View Code Template</span>
                </button>
            )}

            <div className="flex gap-2 pt-1">
                 <button 
                    onClick={onDisconnect}
                    className="flex-1 flex items-center justify-center gap-1.5 text-xs bg-orange-50 dark:bg-orange-900/20 text-orange-600 dark:text-orange-400 hover:bg-orange-100 dark:hover:bg-orange-900/40 py-2 px-2 rounded-md transition-colors border border-orange-200 dark:border-orange-800/30 font-medium"
                    title="Disconnect All Arrows"
                >
                    <DisconnectIcon className="w-3.5 h-3.5" />
                    Disconnect
                </button>
                <button 
                    onClick={onDelete}
                    className="flex-1 flex items-center justify-center gap-1.5 text-xs bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 hover:bg-red-100 dark:hover:bg-red-900/40 py-2 px-2 rounded-md transition-colors border border-red-200 dark:border-red-800/30 font-medium"
                    title="Delete Component"
                >
                    <TrashIcon className="w-3.5 h-3.5" />
                    Delete
                </button>
            </div>
        </div>
    );
};

export const DraggableNode: React.FC<DraggableNodeProps> = ({
  node,
  onMove,
  onMoveEnd,
  onStartConnection,
  onEndConnection,
  isSelected,
  onSelect,
  onAddRelated,
  transformScale,
  isAnimating,
  onStartWorkflow,
  onStopWorkflow,
  isWorkflowRunning,
  onUpdateNode,
  onDeleteNode,
  onDisconnectNode,
  isConnected = true
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [showMiniMenu, setShowMiniMenu] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  
  // Refs for drag calculations to avoid state updates causing re-renders/lag
  const dragStartMouse = useRef({ x: 0, y: 0 });
  const dragStartNodePos = useRef({ x: 0, y: 0 });

  const nodeRef = useRef<HTMLDivElement>(null);
  const hoverTimerRef = useRef<number | null>(null);

  const ServiceIcon = ALL_SERVICES.find(s => s.id === node.type)?.icon;

  // Handle Hover Logic
  const handleMouseEnter = useCallback(() => {
    setIsHovered(true);
    
    if (node.type === 'start') {
        if (!isConnected) return;
        const duration = 2000; 
        hoverTimerRef.current = window.setTimeout(() => {
            if (isWorkflowRunning) {
                onStopWorkflow();
            } else {
                onStartWorkflow();
            }
        }, duration);
    } 
  }, [node.type, isWorkflowRunning, onStartWorkflow, onStopWorkflow, isConnected]);

  const handleMouseLeave = useCallback(() => {
    setIsHovered(false);
    if (hoverTimerRef.current) {
        clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = null;
    }
  }, []);

  const handleMouseDown = useCallback((e: ReactMouseEvent) => {
    if (e.button !== 0) return;
    e.stopPropagation(); // Stop event bubbling
    
    // Select the node
    onSelect(node.id, e.shiftKey);
    
    // Record start positions for delta calculation
    dragStartMouse.current = { x: e.clientX, y: e.clientY };
    dragStartNodePos.current = { x: node.position.x, y: node.position.y };
    
    setIsDragging(true);
    setShowMiniMenu(false); 
  }, [node.id, node.position, onSelect]);

  const handleTouchStart = useCallback((e: React.TouchEvent) => {
      e.stopPropagation();
      const touch = e.touches[0];
      onSelect(node.id, false);
      
      dragStartMouse.current = { x: touch.clientX, y: touch.clientY };
      dragStartNodePos.current = { x: node.position.x, y: node.position.y };
      
      setIsDragging(true);
      setShowMiniMenu(false);
  }, [node.id, node.position, onSelect]);
  
  const handleMouseMove = useCallback((e: MouseEvent | TouchEvent) => {
    if (isDragging) {
        e.preventDefault();
        
        let clientX, clientY;
        if ('touches' in e) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = (e as MouseEvent).clientX;
            clientY = (e as MouseEvent).clientY;
        }

        // Calculate delta adjusted by zoom scale
        const dx = (clientX - dragStartMouse.current.x) / transformScale;
        const dy = (clientY - dragStartMouse.current.y) / transformScale;

        // Apply delta to initial position
        const newX = dragStartNodePos.current.x + dx;
        const newY = dragStartNodePos.current.y + dy;

        // Snap to grid
        const snappedX = Math.round(newX / GRID_SNAP) * GRID_SNAP;
        const snappedY = Math.round(newY / GRID_SNAP) * GRID_SNAP;

        onMove(node.id, { x: snappedX, y: snappedY });
    }
  }, [isDragging, node.id, onMove, transformScale]);
  
  const handleMouseUp = useCallback(() => {
    if (isDragging) {
      onMoveEnd();
    }
    setIsDragging(false);
  }, [isDragging, onMoveEnd]);

  React.useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      window.addEventListener('touchmove', handleMouseMove, { passive: false });
      window.addEventListener('touchend', handleMouseUp);
    } else {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleMouseMove);
      window.removeEventListener('touchend', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleMouseMove);
      window.removeEventListener('touchend', handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp]);

  const renderNodeMetadata = () => {
      const { data } = node;
      let details = [];

      switch(node.type) {
          case 'ec2':
              if(data.instanceFamily && data.instanceSize) details.push(`${data.instanceFamily}.${data.instanceSize}`);
              break;
          case 'lambda':
              if(data.runtime) details.push(data.runtime);
              if(data.memorySize) details.push(`${data.memorySize}MB`);
              break;
          case 'rds':
              if(data.engine) details.push(data.engine);
              if(data.storageSize) details.push(`${data.storageSize}GB`);
              break;
          case 'asg':
              details.push(`Min:${data.minSize} Max:${data.maxSize}`);
              break;
           case 'ami':
              if(data.baseOs) details.push(data.baseOs);
              break;
           case 's3':
              if(data.versioning) details.push('Ver:On');
              if(data.encryption) details.push('Enc:On');
              break;
           case 'dynamodb':
              if(data.billingMode) details.push(data.billingMode);
              break;
      }
      
      if (details.length === 0) return null;

      return (
          <div className="mt-0.5 flex flex-wrap gap-x-1 text-[9px] leading-tight text-gray-500 dark:text-gray-400 font-mono">
              {details.map((detail, i) => (
                  <span key={i} className="bg-gray-100 dark:bg-gray-700/50 px-1 rounded">{detail}</span>
              ))}
          </div>
      )
  }


  const healthIndicatorColor = node.data.health ? healthColorMap[node.data.health] : 'bg-gray-300 dark:bg-gray-600';
  const isWorkflowNode = node.type === 'start' || node.type === 'end';
  const isContainerNode = node.type === 'tg'; // Target Group acts as a container
  
  // Dimensions based on type
  let width = NODE_WIDTH;
  let height = NODE_HEIGHT;
  if (isWorkflowNode) {
      width = WORKFLOW_NODE_SIZE;
      height = WORKFLOW_NODE_SIZE;
  } else if (isContainerNode) {
      width = 300;
      height = 200;
  }
  
  const nodeSizeStyle = { width: `${width}px`, height: `${height}px` };
  
  // Base classes
  let baseClasses = `absolute flex group cursor-pointer transition-all duration-150 shadow-sm hover:shadow-md`;
  let themeClasses = '';
  
  // Dynamic glow classes based on health/traffic
  let glowClasses = '';
  if (node.data.health === 'critical') glowClasses = 'shadow-[0_0_10px_rgba(239,68,68,0.6)] border-red-500';
  else if (node.data.health === 'warning') glowClasses = 'shadow-[0_0_10px_rgba(234,179,8,0.6)] border-yellow-500';
  else if (node.data.traffic === 'high') glowClasses = 'shadow-[0_0_15px_rgba(239,68,68,0.4)] border-red-500 animate-pulse';

  // Support for AI Mode Custom Themes
  if (node.data.customTheme) {
      themeClasses = node.data.customTheme;
  } else if (isWorkflowNode) {
      themeClasses = 'rounded-full justify-center items-center border-0';
  } else if (isContainerNode) {
      themeClasses = 'bg-orange-50/50 dark:bg-orange-900/10 border-2 border-dashed border-orange-300 dark:border-orange-700 rounded-lg items-start pt-2 pl-2';
      baseClasses = `absolute flex group cursor-pointer transition-all duration-150 hover:border-orange-500`; // Remove default shadow/bg logic
  } else {
      themeClasses = `bg-white dark:bg-gray-750 border border-gray-200 dark:border-gray-600 rounded-md hover:border-gray-300 dark:hover:border-gray-500 ${glowClasses}`;
  }
    
  const selectedClasses = isSelected 
    ? (isWorkflowNode ? 'ring-4 ring-orange-500/50' : (isContainerNode ? 'border-orange-500 ring-1 ring-orange-500' : 'ring-2 ring-orange-500 border-orange-500 dark:border-orange-500')) 
    : '';
    
  // Important: Explicitly handle animation prop for Tutorial Glow
  const animatingClasses = isAnimating 
    ? (isWorkflowNode ? 'node-animating' : 'node-animating ring-4 ring-orange-400 shadow-[0_0_20px_rgba(249,115,22,0.6)]') 
    : '';
    
  // Ring Animation for Start Node Gesture
  const circleRadius = 34; 
  const circumference = 2 * Math.PI * circleRadius;
  const ringColor = isWorkflowRunning ? '#ef4444' : '#22c55e';
  const ringDuration = '2s';
  
  const showStartRing = node.type === 'start' && isHovered && isConnected;
  
  // EC2 Contextual Actions
  const showEC2Actions = isHovered && node.type === 'ec2' && node.data.health !== 'healthy';

  return (
    <div
      ref={nodeRef}
      className={`${baseClasses} ${themeClasses} ${selectedClasses} ${animatingClasses}`}
      style={{ left: node.position.x, top: node.position.y, ...nodeSizeStyle, zIndex: isAnimating ? 20 : 10 }}
      onMouseDown={handleMouseDown}
      onTouchStart={handleTouchStart}
      onMouseUp={() => onEndConnection(node.id)}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
        {showMiniMenu && (
            <MiniPropertiesMenu 
                node={node} 
                onClose={() => setShowMiniMenu(false)} 
                onUpdate={(data) => onUpdateNode(node.id, data)}
                onDelete={() => { onDeleteNode(node.id); setShowMiniMenu(false); }}
                onDisconnect={() => { onDisconnectNode(node.id); setShowMiniMenu(false); }}
                onAddRelated={() => onAddRelated(node.id)}
                onViewCode={() => { 
                    onSelect(node.id, false); 
                    setShowMiniMenu(false); 
                }}
            />
        )}

        {/* EC2 Contextual Actions */}
        {showEC2Actions && (
             <div className="absolute -top-10 left-0 w-full flex justify-center space-x-2 z-50">
                  <button onClick={(e) => { e.stopPropagation(); onUpdateNode(node.id, { health: 'healthy' }) }} 
                    className="bg-green-500 hover:bg-green-600 text-white text-[10px] font-bold px-2 py-1 rounded shadow-lg animate-bounce transition-transform transform hover:scale-110">
                    Start
                  </button>
                  <button onClick={(e) => { e.stopPropagation(); /* Reboot logic */ }} 
                    className="bg-yellow-500 hover:bg-yellow-600 text-white text-[10px] font-bold px-2 py-1 rounded shadow-lg animate-bounce transition-transform transform hover:scale-110" style={{ animationDelay: '0.1s' }}>
                    Reboot
                  </button>
                  <button onClick={(e) => { e.stopPropagation(); onUpdateNode(node.id, { health: 'critical' }) }} 
                    className="bg-red-500 hover:bg-red-600 text-white text-[10px] font-bold px-2 py-1 rounded shadow-lg animate-bounce transition-transform transform hover:scale-110" style={{ animationDelay: '0.2s' }}>
                    Stop
                  </button>
             </div>
        )}

        {isWorkflowNode ? (
             <>
                {node.type === 'start' && (
                    <svg className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 overflow-visible pointer-events-none" width="76" height="76">
                        <circle 
                            cx="38" cy="38" r={circleRadius} 
                            fill="none" 
                            stroke={ringColor} 
                            strokeWidth="4" 
                            strokeDasharray={circumference}
                            strokeDashoffset={showStartRing ? 0 : circumference}
                            strokeLinecap="round"
                            style={{ transition: showStartRing ? `stroke-dashoffset ${ringDuration} linear` : 'stroke-dashoffset 0.2s ease-out' }}
                        />
                    </svg>
                )}

                {ServiceIcon && <ServiceIcon className="w-10 h-10 text-gray-600 dark:text-gray-200 relative z-10" />}
                 <div className={`absolute -bottom-6 text-xs font-bold text-gray-600 dark:text-gray-300 w-32 text-center truncate`}>
                    {node.data.label}
                </div>
             </>
        ) : isContainerNode ? (
            // TARGET GROUP CONTAINER LAYOUT
            <>
                <div className="flex items-center space-x-2 mb-2 pointer-events-none">
                     {ServiceIcon && <ServiceIcon className="w-6 h-6 text-orange-600 dark:text-orange-400" />}
                     <span className="text-xs font-bold text-orange-700 dark:text-orange-300 uppercase tracking-wide">{node.data.label}</span>
                </div>
                {/* Connection point hint */}
                <div className="absolute -right-1 top-1/2 w-2 h-2 bg-orange-400 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"></div>
            </>
        ) : node.data.customTheme ? (
             // AI NODE LAYOUT (Simple text centered)
             <div className="flex-1 flex items-center justify-center p-2 text-center overflow-hidden">
                <span className="text-xs font-bold truncate leading-tight pointer-events-none">
                    {node.data.label}
                </span>
             </div>
        ) : (
            // STANDARD NODE LAYOUT
            <>
                <div className="flex-shrink-0 w-12 flex items-center justify-center bg-gray-50 dark:bg-gray-800 border-r border-gray-200 dark:border-gray-600 rounded-l-md overflow-hidden">
                     {ServiceIcon && <ServiceIcon className="w-7 h-7 text-gray-600 dark:text-gray-300" />}
                </div>
                
                <div className="flex-1 min-w-0 flex flex-col justify-center px-3 py-1 overflow-hidden">
                     <span className="text-xs font-bold text-gray-800 dark:text-gray-100 truncate block leading-tight mb-0.5">
                        {node.data.label}
                     </span>
                     <span className="text-[10px] text-gray-500 dark:text-gray-400 truncate block leading-tight uppercase tracking-wide font-semibold">
                         {node.type}
                     </span>
                     {renderNodeMetadata()}
                </div>

                <div className={`w-1.5 rounded-r-md h-full flex-shrink-0 ${healthIndicatorColor}`}></div>
            </>
        )}
        
        {/* Action Buttons */}
        <div 
          className="node-action-button"
          style={{ top: '50%', right: '-10px', transform: 'translateY(-50%)', opacity: 0 }}
          title="Connect"
          onMouseDown={(e) => { e.stopPropagation(); onStartConnection(node.id); }}
          onTouchStart={(e) => { e.stopPropagation(); onStartConnection(node.id); }}
        >
          <div className="w-1.5 h-1.5 bg-white rounded-full"></div>
        </div>

        <div
          title="Settings & Properties"
          className="node-action-button"
          style={{ top: '-10px', right: '5px' }}
          onClick={(e) => { e.stopPropagation(); setShowMiniMenu(!showMiniMenu); }}
        >
          <GearIcon className="w-3 h-3" />
        </div>
        <div
          title="Add Related"
          className="node-action-button"
          style={{ bottom: '-10px', right: '5px' }}
          onClick={(e) => { e.stopPropagation(); onAddRelated(node.id); }}
        >
          <PlusIcon className="w-3 h-3" />
        </div>
    </div>
  );
};
