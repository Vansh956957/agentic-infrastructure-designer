
import React from 'react';
import { Connection, Node, Vector2D } from '../types';
import { NODE_WIDTH, NODE_HEIGHT, WORKFLOW_NODE_SIZE } from '../constants';

interface ConnectorProps {
  connection: Connection;
  nodes: Node[];
  isSelected: boolean;
  onSelect: (connectionId: string | null, isMulti: boolean) => void;
  isAnimating: boolean;
  isAlertState?: boolean;
}

const getNodeDimensions = (node: Node) => {
    const isWorkflow = node.type === 'start' || node.type === 'end';
    return {
        width: isWorkflow ? WORKFLOW_NODE_SIZE : NODE_WIDTH,
        height: isWorkflow ? WORKFLOW_NODE_SIZE : NODE_HEIGHT
    };
};

// Calculates the four connection points on the bounding box of a node
const getConnectionPoints = (node: Node): { top: Vector2D, right: Vector2D, bottom: Vector2D, left: Vector2D } => {
    const { width, height } = getNodeDimensions(node);
    return {
        top: { x: node.position.x + width / 2, y: node.position.y },
        right: { x: node.position.x + width, y: node.position.y + height / 2 },
        bottom: { x: node.position.x + width / 2, y: node.position.y + height },
        left: { x: node.position.x, y: node.position.y + height / 2 },
    };
};

// Finds the pair of connection points with the minimum distance
const getBestConnectionPoints = (fromNode: Node, toNode: Node) => {
    const fromPoints = getConnectionPoints(fromNode);
    const toPoints = getConnectionPoints(toNode);
    let minDistance = Infinity;
    let bestPoints: { start: Vector2D, end: Vector2D, fromSide: string, toSide: string } = {
        start: fromPoints.right,
        end: toPoints.left,
        fromSide: 'right',
        toSide: 'left'
    };

    for (const [fromSide, fromPoint] of Object.entries(fromPoints)) {
        for (const [toSide, toPoint] of Object.entries(toPoints)) {
            const dx = fromPoint.x - toPoint.x;
            const dy = fromPoint.y - toPoint.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < minDistance) {
                minDistance = distance;
                bestPoints = { start: fromPoint, end: toPoint, fromSide, toSide };
            }
        }
    }
    return bestPoints;
};

export const Connector: React.FC<ConnectorProps> = ({ connection, nodes, isSelected, onSelect, isAnimating, isAlertState }) => {
  const fromNode = nodes.find((n) => n.id === connection.fromNodeId);
  const toNode = nodes.find((n) => n.id === connection.toNodeId);

  if (!fromNode || !toNode) {
    return null;
  }
  
  const { start, end, fromSide, toSide } = getBestConnectionPoints(fromNode, toNode);

  const dx = end.x - start.x;
  const dy = end.y - start.y;
  
  let controlX1 = start.x;
  let controlY1 = start.y;
  let controlX2 = end.x;
  let controlY2 = end.y;

  const offset = Math.max(Math.abs(dx), Math.abs(dy)) * 0.4;
  
  if (fromSide === 'right') controlX1 += offset;
  if (fromSide === 'left') controlX1 -= offset;
  if (fromSide === 'top') controlY1 -= offset;
  if (fromSide === 'bottom') controlY1 += offset;

  if (toSide === 'right') controlX2 += offset;
  if (toSide === 'left') controlX2 -= offset;
  if (toSide === 'top') controlY2 -= offset;
  if (toSide === 'bottom') controlY2 += offset;


  const pathData = `M${start.x},${start.y} C${controlX1},${controlY1} ${controlX2},${controlY2} ${end.x},${end.y}`;

  // For label positioning
  const t = 0.5; // Midpoint of the Bezier curve
  const midX = Math.pow(1 - t, 3) * start.x + 3 * Math.pow(1 - t, 2) * t * controlX1 + 3 * (1 - t) * t * t * controlX2 + Math.pow(t, 3) * end.x;
  const midY = Math.pow(1 - t, 3) * start.y + 3 * Math.pow(1 - t, 2) * t * controlY1 + 3 * (1 - t) * t * t * controlX2 + Math.pow(t, 3) * end.y;
  
  // Angle for label rotation
  const p0 = {x: start.x, y: start.y};
  const p1 = {x: controlX1, y: controlY1};
  const p2 = {x: controlX2, y: controlY2};
  const p3 = {x: end.x, y: end.y};

  const dxt = 3 * (1 - t) * (1 - t) * (p1.x - p0.x) + 6 * (1 - t) * t * (p2.x - p1.x) + 3 * t * t * (p3.x - p2.x);
  const dyt = 3 * (1 - t) * (1 - t) * (p1.y - p0.y) + 6 * (1 - t) * t * (p2.y - p1.y) + 3 * t * t * (p3.y - p2.y);
  const angle = Math.atan2(dyt, dxt) * (180 / Math.PI);

  const pathClasses = [
      'connector-path',
      isSelected && 'connector-path-highlight',
      isAnimating && 'connector-path-animating',
      isAlertState && 'connector-path-alert'
  ].filter(Boolean).join(' ');
  
  const markerEnd = isAlertState 
    ? 'url(#arrowhead-danger)' 
    : (isSelected || isAnimating ? 'url(#arrowhead-highlight)' : 'url(#arrowhead)');
  
  return (
    <g onClick={(e) => { e.stopPropagation(); onSelect(connection.id, e.shiftKey) }} className="pointer-events-auto">
      <style>
      {`
        .connector-path-alert {
            stroke: #ef4444 !important;
            stroke-dasharray: 10, 5;
            animation: march 0.5s linear infinite;
        }
      `}
      </style>
      <path d={pathData} className="connector-interaction-layer" />
      <path d={pathData} className={pathClasses} style={{ markerEnd }} />
      {connection.label && (
        <text
            x={midX}
            y={midY - 8}
            fill={isSelected ? '#f97316' : '#6b7280'}
            fontSize="12"
            textAnchor="middle"
            transform={`rotate(${angle} ${midX} ${midY})`}
            style={{pointerEvents: 'none'}}
            className="fill-gray-500 dark:fill-gray-400"
        >
            {connection.label}
        </text>
      )}
    </g>
  );
};
