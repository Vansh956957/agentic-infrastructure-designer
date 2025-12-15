
import React, { useState, useRef, useCallback, MouseEvent as ReactMouseEvent, useEffect } from 'react';
import { Node as CanvasNode, Connection, Vector2D, AWSService, Group, GroupData, Shape, ShapeData, TextNode, TextNodeData, NodeData, Node } from '../types';
import { DraggableNode } from './DraggableNode';
import { Connector } from './Connector';
import { ALL_SERVICES, SERVICE_RELATIONS, EC2_INSTANCE_TYPES, RDS_ENGINES } from '../constants';
import { GroupNode } from './GroupNode';
import { ShapeNode } from './ShapeNode';
import { TextNode as TextNodeComponent } from './TextNode';
import { XIcon } from './Icons';

type Selection = { type: 'node', id: string } | { type: 'connection', id: string } | { type: 'group', id: string } | { type: 'shape', id: string } | { type: 'text', id: string };

interface CanvasProps {
  nodes: CanvasNode[];
  connections: Connection[];
  groups: Group[];
  shapes: Shape[];
  textNodes: TextNode[];
  onAddNode: (service: AWSService, position: Vector2D, dataOverrides?: Partial<NodeData>) => Node;
  onAddGroup: (position: Vector2D) => void;
  onAddShape: (shapeType: 'rectangle' | 'ellipse', position: Vector2D) => void;
  onAddTextNode: (position: Vector2D) => void;
  onAddRelatedNode: (sourceNode: CanvasNode, service: AWSService) => void;
  onUpdateNodePosition: (nodeId: string, newPosition: Vector2D) => void;
  onUpdateGroupPosition: (groupId: string, delta: Vector2D) => void;
  onUpdateGroup: (groupId: string, data: Partial<GroupData> & {size?: Group['size']}) => void;
  onUpdateShape: (shapeId: string, data: Partial<ShapeData> & {size?: Shape['size'], position?: Shape['position']}) => void;
  onUpdateTextNode: (textId: string, data: Partial<TextNodeData> & {size?: TextNode['size'], position?: TextNode['position']}) => void;
  onUpdateNodeData: (nodeId: string, data: Partial<NodeData>) => void;
  onDeleteNode: (nodeId: string) => void;
  onDisconnectNode: (nodeId: string) => void;
  onStartWorkflow: (nodeId: string) => void;
  onStopWorkflow: () => void;
  workflowRunningNodeId: string | null;
  onAddConnection: (fromNodeId: string, toNodeId: string) => void;
  selection: Selection[];
  onSelect: (selection: Selection | null, additive: boolean) => void;
  isSelectingForExport: boolean;
  isMultiSelectMode: boolean;
  onExportArea: (area: {x: number, y: number, width: number, height: number}) => void;
  onMultiSelect: (area: {x: number, y: number, width: number, height: number}) => void;
  onCancelExportSelection: () => void;
  animationState: { activeNodes: Set<string>, activeConnections: Set<string>, alertConnections: Set<string> };
  onInteractionEnd: () => void;
  transform: { x: number, y: number, k: number };
  setTransform: React.Dispatch<React.SetStateAction<{ x: number, y: number, k: number }>>;
}

const AddRelatedServiceMenu: React.FC<{
    node: CanvasNode;
    onAdd: (service: AWSService) => void;
    onClose: () => void;
}> = ({ node, onAdd, onClose }) => {
    const menuRef = useRef<HTMLDivElement>(null);
    const relatedServiceIds = SERVICE_RELATIONS[node.type] || [];
    const relatedServices = ALL_SERVICES.filter(s => relatedServiceIds.includes(s.id));

    React.useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (menuRef.current && !menuRef.current.contains(event.target as globalThis.Node)) {
                onClose();
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [onClose]);
    
    if (relatedServices.length === 0) {
        return null;
    }

    return (
        <div
            ref={menuRef}
            className="absolute bg-white dark:bg-gray-700 rounded-md shadow-lg p-2 z-50 text-sm border border-gray-200 dark:border-gray-600"
            style={{ left: node.position.x + 160, top: node.position.y }}
        >
            <p className="text-xs text-gray-500 dark:text-gray-400 px-2 pb-1 border-b border-gray-200 dark:border-gray-600 mb-1">Add related:</p>
            {relatedServices.map(service => (
                <button
                    key={service.id}
                    onClick={() => onAdd(service)}
                    className="flex items-center w-full text-left p-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200"
                >
                    <service.icon className="w-5 h-5 mr-2" />
                    {service.name}
                </button>
            ))}
        </div>
    );
};

const AdvancedServiceConfigMenu: React.FC<{
    node: CanvasNode;
    onImport: (config: any) => void;
    onClose: () => void;
}> = ({ node, onImport, onClose }) => {
    // ... (Code omitted for brevity, same as previous)
    const menuRef = useRef<HTMLDivElement>(null);
    const [config, setConfig] = useState<any>({ ...node.data });
    const handleChange = (key: string, value: any) => setConfig((prev: any) => ({ ...prev, [key]: value }));

    return (
        <div 
            ref={menuRef}
            className="absolute bg-white dark:bg-gray-800 rounded-lg shadow-2xl z-50 text-sm border border-gray-200 dark:border-gray-700 w-80 max-h-[500px] overflow-y-auto flex flex-col"
            style={{ left: node.position.x + 190, top: Math.max(10, node.position.y - 100) }}
            onMouseDown={e => e.stopPropagation()}
        >
            <div className="flex justify-between items-center p-3 border-b border-gray-200 dark:border-gray-700 sticky top-0 bg-white dark:bg-gray-800 z-10">
                <h3 className="font-bold text-gray-800 dark:text-white flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-orange-500"></span>
                    {node.data.label} Config
                </h3>
                <button onClick={onClose} className="text-gray-400 hover:text-gray-600"><XIcon className="w-4 h-4" /></button>
            </div>
            <div className="p-4">
                <p className="text-xs text-gray-500 mb-2">Properties</p>
                {Object.keys(config).filter(k => !['label', 'health', 'traffic', 'customTheme', 'code', 'hyperparams'].includes(k)).map(key => (
                    <div key={key} className="mb-2">
                        <label className="block text-[10px] uppercase text-gray-500 mb-1">{key}</label>
                        <input 
                            type="text" 
                            className="w-full bg-gray-50 dark:bg-gray-900 border border-gray-300 dark:border-gray-600 rounded p-1.5"
                            value={config[key]}
                            onChange={e => handleChange(key, e.target.value)}
                        />
                    </div>
                ))}
                 {Object.keys(config).filter(k => !['label', 'health', 'traffic', 'customTheme', 'code', 'hyperparams'].includes(k)).length === 0 && (
                     <p className="text-gray-400 italic">No specific properties to configure.</p>
                 )}
            </div>
            <div className="p-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 sticky bottom-0">
                <button onClick={() => onImport(config)} className="w-full bg-orange-600 hover:bg-orange-700 text-white font-bold py-2 px-4 rounded shadow-md">Update</button>
            </div>
        </div>
    );
}

const SelectionTool: React.FC<{
    onSelect: (rect: {x: number, y: number, width: number, height: number}) => void,
    onCancel: () => void,
    getCanvasCoordinates: (e: ReactMouseEvent) => Vector2D,
    color?: string
}> = ({ onSelect, onCancel, getCanvasCoordinates, color = '#f90' }) => {
    const [startPos, setStartPos] = useState<Vector2D | null>(null);
    const [endPos, setEndPos] = useState<Vector2D | null>(null);

    const handleMouseDown = (e: ReactMouseEvent) => {
        setStartPos(getCanvasCoordinates(e));
        setEndPos(getCanvasCoordinates(e));
    };

    const handleMouseMove = (e: ReactMouseEvent) => {
        if (startPos) {
            setEndPos(getCanvasCoordinates(e));
        }
    };

    const handleMouseUp = () => {
        if (startPos && endPos) {
            const rect = {
                x: Math.min(startPos.x, endPos.x),
                y: Math.min(startPos.y, endPos.y),
                width: Math.abs(startPos.x - endPos.x),
                height: Math.abs(startPos.y - endPos.y),
            };
            if (rect.width > 5 && rect.height > 5) {
                onSelect(rect);
            } else {
                onCancel();
            }
        }
        setStartPos(null);
        setEndPos(null);
    };

    const selectionRectStyle = startPos && endPos ? {
        left: Math.min(startPos.x, endPos.x),
        top: Math.min(startPos.y, endPos.y),
        width: Math.abs(startPos.x - endPos.x),
        height: Math.abs(startPos.y - endPos.y),
        borderColor: color,
        backgroundColor: color === '#f90' ? 'rgba(255, 153, 0, 0.2)' : 'rgba(59, 130, 246, 0.2)' // Orange vs Blue tint
    } : {};
    
    return (
        <div 
            className="selection-overlay"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
        >
           {startPos && endPos && <div className="selection-rectangle" style={selectionRectStyle} />}
        </div>
    )
}


export const Canvas: React.FC<CanvasProps> = ({
  nodes,
  connections,
  groups,
  shapes,
  textNodes,
  onAddNode,
  onAddGroup,
  onAddShape,
  onAddTextNode,
  onAddRelatedNode,
  onUpdateNodePosition,
  onUpdateGroupPosition,
  onUpdateGroup,
  onUpdateShape,
  onUpdateTextNode,
  onUpdateNodeData,
  onDeleteNode,
  onDisconnectNode,
  onStartWorkflow,
  onStopWorkflow,
  workflowRunningNodeId,
  onAddConnection,
  selection,
  onSelect,
  isSelectingForExport,
  isMultiSelectMode,
  onExportArea,
  onMultiSelect,
  onCancelExportSelection,
  animationState,
  onInteractionEnd,
  transform,
  setTransform,
}) => {
  const [isPanning, setIsPanning] = useState(false);
  const [startPanPosition, setStartPanPosition] = useState({ x: 0, y: 0 });
  const [isConnecting, setIsConnecting] = useState<string | null>(null);
  const [tempConnectorEnd, setTempConnectorEnd] = useState<Vector2D | null>(null);
  const [relatedMenuNode, setRelatedMenuNode] = useState<CanvasNode | null>(null);
  const canvasRef = useRef<HTMLDivElement>(null);

  const getCanvasCoordinates = useCallback((e: ReactMouseEvent | MouseEvent | React.DragEvent): Vector2D => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left - transform.x) / transform.k,
      y: (e.clientY - rect.top - transform.y) / transform.k,
    };
  }, [transform]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    if(isSelectingForExport || isMultiSelectMode) return;
    const zoomFactor = 1.1;
    const newScale = e.deltaY < 0 ? transform.k * zoomFactor : transform.k / zoomFactor;
    const clampedScale = Math.max(0.1, Math.min(newScale, 5));

    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const newX = mouseX - (mouseX - transform.x) * (clampedScale / transform.k);
    const newY = mouseY - (mouseY - transform.y) * (clampedScale / transform.k);

    setTransform({ x: newX, y: newY, k: clampedScale });
  }, [transform, isSelectingForExport, isMultiSelectMode, setTransform]);

  const handleMouseDown = useCallback((e: ReactMouseEvent) => {
    if (isSelectingForExport || isMultiSelectMode) return;
    // Pan with middle mouse button OR primary button if clicking on canvas background
    if ((e.button === 1 || (e.button === 0 && e.target === e.currentTarget.firstChild))) {
      e.preventDefault();
      setIsPanning(true);
      setStartPanPosition({ x: e.clientX - transform.x, y: e.clientY - transform.y });
    }
  }, [transform, isSelectingForExport, isMultiSelectMode]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (isPanning) {
      const newX = e.clientX - startPanPosition.x;
      const newY = e.clientY - startPanPosition.y;
      setTransform((prev) => ({ ...prev, x: newX, y: newY }));
    }
    if (isConnecting) {
      setTempConnectorEnd(getCanvasCoordinates(e));
    }
  }, [isPanning, startPanPosition, isConnecting, getCanvasCoordinates, setTransform]);

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
    if (isConnecting) {
      setIsConnecting(null);
      setTempConnectorEnd(null);
    }
  }, [isConnecting]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const serviceId = e.dataTransfer.getData('application/aws-service');
    const service = ALL_SERVICES.find((s) => s.id === serviceId);
    if (service) {
      const position = getCanvasCoordinates(e);
      if(service.id === 'group') {
        onAddGroup(position);
      } else if (service.id === 'shape-rectangle') {
        onAddShape('rectangle', position);
      } else if (service.id === 'shape-ellipse') {
        onAddShape('ellipse', position);
      } else if (service.id === 'text') {
        onAddTextNode(position);
      }
      else {
        onAddNode(service, position);
      }
    }
  }, [getCanvasCoordinates, onAddNode, onAddGroup, onAddShape, onAddTextNode]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };
  
  const startConnection = useCallback((nodeId: string) => {
    setIsConnecting(nodeId);
    const fromNode = nodes.find(n => n.id === nodeId);
    if (fromNode) {
        // Center start of connection
        const isWorkflow = fromNode.type === 'start' || fromNode.type === 'end';
        const width = isWorkflow ? 64 : 180;
        const height = 64;
        setTempConnectorEnd({ x: fromNode.position.x + width/2, y: fromNode.position.y + height/2 });
    }
  }, [nodes]);

  const endConnection = useCallback((nodeId: string) => {
    if (isConnecting) {
        onAddConnection(isConnecting, nodeId);
    }
    setIsConnecting(null);
    setTempConnectorEnd(null);
  }, [isConnecting, onAddConnection]);
  
  const handleCanvasClick = (e: ReactMouseEvent) => {
      // Only deselect if we clicked the background and NOT in selection mode
      if ((e.target === canvasRef.current || e.target === canvasRef.current?.firstChild) && !isMultiSelectMode) {
          onSelect(null, false);
          setRelatedMenuNode(null);
      }
  }

  React.useEffect(() => {
    if (isPanning || isConnecting) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    } else {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isPanning, isConnecting, handleMouseMove, handleMouseUp]);
  
  const fromNodeForTempConnection = isConnecting ? nodes.find(n => n.id === isConnecting) : null;
  const isWorkflowFrom = fromNodeForTempConnection && (fromNodeForTempConnection.type === 'start' || fromNodeForTempConnection.type === 'end');
  const fromSize = isWorkflowFrom ? {w: 64, h: 64} : {w: 180, h: 64};
  const fromPos = fromNodeForTempConnection ? { x: fromNodeForTempConnection.position.x, y: fromNodeForTempConnection.position.y } : {x:0, y:0};

  const sortedGroups = [...groups].sort((a, b) => (a.size.width * a.size.height) - (b.size.width * b.size.height));

  const sortedNodes = [...nodes].sort((a, b) => {
      if (a.type === 'tg' && b.type !== 'tg') return -1;
      if (a.type !== 'tg' && b.type === 'tg') return 1;
      return 0;
  });
  
  const handleConfigUpdate = (config: any) => {
      if (!relatedMenuNode) return;
      onUpdateNodeData(relatedMenuNode.id, config);
      setRelatedMenuNode(null);
  };

  return (
    <div
      ref={canvasRef}
      className={`w-full h-full bg-gray-50 dark:bg-gray-800 relative ${isMultiSelectMode ? 'cursor-crosshair' : (isPanning ? 'cursor-grabbing' : 'cursor-grab')}`}
      style={{
        backgroundImage: 'radial-gradient(circle at 1px 1px, #cbd5e1 1px, transparent 0)',
        backgroundSize: '20px 20px',
      }}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onClick={handleCanvasClick}
    >
      <style>{`.dark .bg-gray-800 { background-image: radial-gradient(circle at 1px 1px, #4a5568 1px, transparent 0) !important; }`}</style>

      <div
        id="canvas-content-wrapper"
        className="w-full h-full absolute top-0 left-0"
        style={{ transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.k})`, transformOrigin: '0 0' }}
      >
        {isSelectingForExport && 
          <SelectionTool 
            onSelect={onExportArea}
            onCancel={onCancelExportSelection}
            getCanvasCoordinates={getCanvasCoordinates}
          />
        }
        
        {isMultiSelectMode && 
          <SelectionTool 
            onSelect={onMultiSelect}
            onCancel={() => {}} // Do nothing on cancel, just stay in mode
            getCanvasCoordinates={getCanvasCoordinates}
            color="#3b82f6" // Blue for multi-select
          />
        }

        {shapes.map((shape) => (
            <ShapeNode
                key={shape.id}
                shape={shape}
                onUpdate={onUpdateShape}
                onSelect={(id, isMulti) => onSelect({ type: 'shape', id }, isMulti)}
                isSelected={selection.some(s => s.type === 'shape' && s.id === shape.id)}
                transformScale={transform.k}
                onInteractionEnd={onInteractionEnd}
            />
        ))}
        
        {textNodes.map((textNode) => (
            <TextNodeComponent
                key={textNode.id}
                textNode={textNode}
                onUpdate={onUpdateTextNode}
                onSelect={(id, isMulti) => onSelect({ type: 'text', id }, isMulti)}
                isSelected={selection.some(s => s.type === 'text' && s.id === textNode.id)}
                transformScale={transform.k}
                onInteractionEnd={onInteractionEnd}
            />
        ))}

        {sortedGroups.map((group) => (
            <GroupNode
                key={group.id}
                group={group}
                onMove={onUpdateGroupPosition}
                onUpdate={onUpdateGroup}
                onSelect={(id, isMulti) => onSelect({ type: 'group', id }, isMulti)}
                isSelected={selection.some(s => s.type === 'group' && s.id === group.id)}
                transformScale={transform.k}
                onInteractionEnd={onInteractionEnd}
            />
        ))}

        <svg id="connectors-svg" className="w-full h-full absolute top-0 left-0 pointer-events-none" style={{ transformOrigin: '0 0' }}>
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="8" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#9ca3af" className="dark:fill-gray-600" />
                </marker>
                 <marker id="arrowhead-highlight" markerWidth="10" markerHeight="7" refX="8" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#f97316" className="neon-glow" />
                </marker>
                <marker id="arrowhead-danger" markerWidth="10" markerHeight="7" refX="8" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444" className="neon-glow-red" />
                </marker>
            </defs>
            <g>
              {connections.map((conn) => (
                  <Connector 
                      key={conn.id} 
                      connection={conn} 
                      nodes={nodes} 
                      isSelected={selection.some(s => s.type === 'connection' && s.id === conn.id)}
                      onSelect={(id, isMulti) => onSelect(id ? {type: 'connection', id} : null, isMulti)}
                      isAnimating={animationState.activeConnections.has(conn.id)}
                      isAlertState={animationState.alertConnections.has(conn.id)}
                  />
              ))}
              {isConnecting && tempConnectorEnd && (
                  <path 
                      d={`M ${fromPos.x + fromSize.w/2} ${fromPos.y + fromSize.h/2} L ${tempConnectorEnd.x} ${tempConnectorEnd.y}`} 
                      className="connector-path connector-path-highlight"
                      style={{markerEnd: 'url(#arrowhead-highlight)'}}
                  />
              )}
            </g>
        </svg>

        {sortedNodes.map((node) => (
          <DraggableNode
            key={node.id}
            node={node}
            onMove={onUpdateNodePosition}
            onStartConnection={startConnection}
            onEndConnection={endConnection}
            isSelected={selection.some(s => s.type === 'node' && s.id === node.id)}
            onSelect={(id, isMulti) => onSelect({ type: 'node', id }, isMulti)}
            onAddRelated={() => setRelatedMenuNode(node)}
            transformScale={transform.k}
            isAnimating={animationState.activeNodes.has(node.id)}
            onMoveEnd={onInteractionEnd}
            onStartWorkflow={() => onStartWorkflow(node.id)}
            onStopWorkflow={onStopWorkflow}
            isWorkflowRunning={workflowRunningNodeId === node.id}
            onUpdateNode={onUpdateNodeData}
            onDeleteNode={onDeleteNode}
            onDisconnectNode={onDisconnectNode}
            isConnected={node.type === 'start' ? connections.some(c => c.fromNodeId === node.id) : true}
          />
        ))}
        {relatedMenuNode && (
             <AdvancedServiceConfigMenu 
                node={relatedMenuNode}
                onImport={handleConfigUpdate}
                onClose={() => setRelatedMenuNode(null)}
            />
        )}
      </div>
    </div>
  );
};